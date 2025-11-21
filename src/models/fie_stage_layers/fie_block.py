import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.utils.layer_norm import LayerNorm2d
from src.models.utils.fre_block import FreBlock

class SimpleGate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        part1, part2 = torch.chunk(x, 2, dim=1)
        output = torch.mul(part1, part2)
        return output

class FieBlock(nn.Module):
    def __init__(self, channels: int = 16, DW_Expand: int = 2, FFN_Expand: int = 2, drop_out_rate: float = 0.7) -> None:        
        super().__init__()
        dw_channel = channels * DW_Expand
        ffn_channel = FFN_Expand * channels
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.sg = SimpleGate()
        self.fre_block = FreBlock(channels=channels)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.fre_block(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.dropout2(x)
        return y + x * self.gamma

if __name__ == "__main__":
    block = FieBlock(channels=16)
    x = torch.rand(16, 16, 256, 256)
    assert block(x).shape == torch.Size([16, 16, 256, 256])