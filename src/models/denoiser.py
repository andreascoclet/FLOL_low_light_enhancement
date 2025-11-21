import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.denoiser_layers.snr_map_kornia import SNRMap
from src.models.denoiser_layers.snr_interaction import SNRInteraction
from src.models.denoiser_layers.freq_branch import FreqStage
from src.models.denoiser_layers.spatial_branch import GeneralizedNAFNet
from src.models.utils.layer_norm import LayerNorm2d



class UpBlock(nn.Module):
    def __init__(self, channels: int) -> None:     
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class Denoiser(nn.Module):
    def __init__(self, in_channels: int = 6, channels: int = 16, num_fre_blocks: int = 4) -> None:        
        super().__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(channels * 2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(channels * 4),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm2d(channels * 4),
            nn.ReLU(inplace=True)
        )

        # Branches
        self.freq_branch = FreqStage(channels=channels * 4, num_fre_blocks=num_fre_blocks)
        #self.spatial_branch = GeneralizedNAFNet(in_channels=channels * 4, width=channels)
        self.spatial_branch = GeneralizedNAFNet(in_channels=channels * 4, width=channels, with_relu=True)
        self.snr_map = SNRMap()
        self.snr_interaction = SNRInteraction()

        # Decoder
        self.up_1 = UpBlock(channels * 4)
        self.up_2 = UpBlock(channels * 2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm2d(channels * 4),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)

        # Learnable residual parameters
        self.beta = nn.Parameter(torch.zeros((1, 3, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels * 4, 1, 1)), requires_grad=True)


    def forward(self, x: torch.Tensor, x_concat: torch.Tensor, x_lol: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc_1 = self.conv1(x_concat)
        enc_2 = self.conv2(enc_1)
        enc_3 = self.conv3(enc_2)
        x_enc = self.conv4(enc_3)

        # SNR map from x_lol
        snr_map = self.snr_map(x_lol)

        # Spatial and frequency branches
        o_s = self.spatial_branch(x_enc)
        o_f = self.freq_branch(x_enc) + x_enc * self.gamma

        # SNR interaction
        features = self.snr_interaction(o_s, o_f, snr_map)

        # Decoder
        res = features + x_enc
        dec = self.conv5(res)
        res = dec + enc_3
        dec = self.up_1(res)
        res = dec + enc_2
        dec = self.up_2(res)
        dec = self.conv6(dec)
        dec = self.conv7(dec)

        # Residual connection
        output = x + dec * self.beta
        return output


if __name__ == "__main__":
    denoiser = Denoiser(in_channels=6, channels=16, num_fre_blocks=4)
    x = torch.rand(16, 3, 256, 256)
    x_concat = torch.rand(16, 6, 256, 256)
    x_lol = torch.rand(16, 3, 256, 256)
    output = denoiser(x, x_concat, x_lol)
    assert output.shape == torch.Size([16, 3, 256, 256])
