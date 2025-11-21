import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.layer_norm import LayerNorm2d

class SimpleGate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
            Split the input in 2, along the channel axis
            Return multiplication of these 2 parts
        """
        part1, part2 = torch.chunk(x, 2, dim=1)
        output = torch.mul(part1, part2)
        return output

class NAFNetUpBlock(nn.Module):
    def __init__(self, channels):
        """NAFNet upsampling block
            conv 1x1 (chan, 2 * chan)
            pixelshuffle(2)
        """
        super(NAFNetUpBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        return x

class NAFNetDownBlock(nn.Module):
    def __init__(self, channels):
        """NAFNet downsampling block
            conv with stride 2, **mind the padding**
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return x

class SCA(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Simplified channel attention module
            adaptiveavgpool to get 1x1 feature map
            conv 1x1 projection layer
        """
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def forward(self, x):
        att_weights = self.sca(x)
        x = self.adaptive_pool(att_weights)
        return att_weights

class NAFNetBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        ffn_channel = FFN_Expand * c
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sca = SCA(in_channels=dw_channel // 2, out_channels=dw_channel // 2)
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)



    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class GeneralizedNAFNet(nn.Module):
    def __init__(self, NAFNetBlock, NAFNetDownBlock, NAFNetUpBlock, img_channel=3, width=36, middle_blk_num=1,  enc_blk_nums=[1,2,2,2], dec_blk_nums=[2,2,2,1]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFNetBlock(chan) for _ in range(num)]))
            self.downs.append(NAFNetDownBlock(chan))
            chan = chan * 2
        self.middle_blks = nn.Sequential(*[NAFNetBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(NAFNetUpBlock(chan))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFNetBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    

# class NAFNetBlock(nn.Module):
#     def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
#         super().__init__()
#         dw_channel  = c * DW_Expand
#         ffn_channel = FFN_Expand * c

#         # Token mixing
#         self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)  # depthwise
#         self.sg    = SimpleGate()  # halves channels
#         self.sca   = SCA(in_channels=dw_channel // 2, out_channels=dw_channel // 2)  # channel attention on halved channels
#         self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

#         # FFN
#         self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

#         # Norms, dropout, scales
#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)
#         self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#         self.beta  = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

#     def forward(self, inp):
#         x = inp
#         print(f"in                     : {tuple(x.shape)}")             # [B, c, H, W]

#         x = self.norm1(x)
#         print(f"after norm1            : {tuple(x.shape)}")             # [B, c, H, W]

#         x = self.conv1(x)
#         print(f"after conv1 (1x1)      : {tuple(x.shape)}")             # [B, c*DW_Expand, H, W]

#         x = self.conv2(x)
#         print(f"after dconv (3x3 dw)   : {tuple(x.shape)}")             # [B, c*DW_Expand, H, W]

#         x = self.sg(x)
#         print(f"after SimpleGate       : {tuple(x.shape)}")             # [B, (c*DW_Expand)/2, H, W]

#         att = self.sca(x)
#         print(f"SCA weights            : {tuple(att.shape)}")           # matches x shape in your SCA

#         x = x * att
#         print(f"after channel att mul  : {tuple(x.shape)}")             # [B, (c*DW_Expand)/2, H, W]

#         x = self.conv3(x)
#         print(f"after conv3 (1x1)      : {tuple(x.shape)}")             # [B, c, H, W]

#         x = self.dropout1(x)
#         print(f"after dropout1         : {tuple(x.shape)}")             # [B, c, H, W]

#         y = inp + x * self.beta
#         print(f"residual y             : {tuple(y.shape)}")             # [B, c, H, W]

#         x = self.norm2(y)
#         print(f"after norm2            : {tuple(x.shape)}")             # [B, c, H, W]

#         x = self.conv4(x)
#         print(f"after conv4 (1x1)      : {tuple(x.shape)}")             # [B, c*FFN_Expand, H, W]

#         x = self.sg(x)
#         print(f"after SimpleGate (FFN) : {tuple(x.shape)}")             # [B, (c*FFN_Expand)/2, H, W]

#         x = self.conv5(x)
#         print(f"after conv5 (1x1)      : {tuple(x.shape)}")             # [B, c, H, W]

#         x = self.dropout2(x)
#         print(f"after dropout2         : {tuple(x.shape)}")             # [B, c, H, W]

#         out = y + x * self.gamma
#         print(f"out                    : {tuple(out.shape)}")           # [B, c, H, W]
#         return out
if __name__ == "__main__":
    # Sanity checks for NAFNet components
    # block = SimpleGate()
    # x = torch.rand(16, 64, 256, 256)
    # assert block(x).shape == torch.Size([16, 32, 256, 256])

    # block = NAFNetUpBlock(64)
    # x = torch.rand(16, 64, 256, 256)
    # assert block(x).shape == torch.Size([16, 32, 512, 512])

    # block = NAFNetDownBlock(64)
    # x = torch.rand(16, 64, 256, 256)
    # assert block(x).shape == torch.Size([16, 128, 128, 128])

    # block = SCA(64, 32)
    # x = torch.rand(16, 64, 256, 256)
    # assert block(x).shape == torch.Size([16, 32, 256, 256])

    # block = NAFNetBlock(6)
    # x = torch.rand(16, 6, 256, 256)
    # assert block(x).shape == torch.Size([16, 6, 256, 256])

    model = GeneralizedNAFNet(NAFNetBlock, NAFNetDownBlock, NAFNetUpBlock, img_channel=6, width=16, enc_blk_nums=[1,1,1,1], dec_blk_nums=[1,1,1,1])#, enc_blk_nums=[1,2,2,2], dec_blk_nums=[2,2,2,1])
    print(model)
    x = torch.rand(16, 6, 256, 256)
    assert model(x).shape == torch.Size([16, 6, 256, 256])