import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.fie_stage_layers.fie_block import FieBlock


class FieStage(nn.Module):
    def __init__(self, in_channels: int = 3, channels: int = 16, num_fie_blocks: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, padding=0, stride=1, bias=True)
        self.fie_blocks = nn.Sequential(*[FieBlock(channels=channels) for _ in range(num_fie_blocks)])
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        inp = x
        # First path: Downsample, process, upsample
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=None)
        x = self.conv1(x)
        x = self.fie_blocks(x)
        x = self.conv2(x)
        module_ampl = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=None)
        # Second path: FFT processing
        fft_out = torch.fft.fft2(inp, norm='backward')
        phi = torch.angle(fft_out)
        module = torch.abs(fft_out)
        ratio = module / (module_ampl.abs() + 1e-6)
        real_img = ratio * torch.cos(phi)
        imag_img = ratio * torch.sin(phi)
        img = torch.complex(real_img, imag_img)
        x_lol = torch.fft.ifft2(img, s=(H, W), norm='backward').real
        x_concat = torch.cat([inp, x_lol], dim=1)
        return x_concat, x_lol

if __name__ == "__main__":
    fie_stage = FieStage(channels=16)
    x = torch.rand(16, 3, 256, 256)
    x_concat, x_lol = fie_stage(x)
    assert x_concat.shape == torch.Size([16, 6, 256, 256])
    assert x_lol.shape == torch.Size([16, 3, 256, 256])