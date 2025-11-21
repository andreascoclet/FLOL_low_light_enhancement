import torch
import torch.nn as nn
import torch.nn.functional as F


class FreBlock(nn.Module):
    def __init__(self, channels: int, in_channels: int | None = None, drop_out_rate: float = 0.7) -> None:
        super().__init__()
        # Default: if in_channels is None, use channels
        if in_channels is None:
            in_channels = channels
        self.conv1 = nn.Conv2d(in_channels, out_channels=channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        inp = x
        x = self.conv1(x)
        fft_out = torch.fft.rfft2(x, norm='backward')
        phi = torch.angle(fft_out)
        module = torch.abs(fft_out)
        phi = self.conv3(F.leaky_relu(self.conv2(phi), negative_slope=0.1))
        module = self.conv5(F.leaky_relu(self.conv4(module), negative_slope=0.1))
        real_img = module * torch.cos(phi)
        imag_img = module * torch.sin(phi)
        img = torch.complex(real_img, imag_img)
        x = torch.fft.irfft2(img, s=(H, W), norm='backward')
        x = self.dropout1(x)
        return inp + x * self.beta

if __name__ == "__main__":
    block = FreBlock(channels=16)
    x = torch.rand(16, 16, 256, 256)
    assert block(x).shape == torch.Size([16, 16, 256, 256])