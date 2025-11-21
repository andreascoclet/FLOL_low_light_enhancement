import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.denoiser import Denoiser
from src.models.fie_stage import FieStage


class FlolPlus(nn.Module):
    def __init__(self, channels=16, num_fie_blocks=4, num_fre_blocks=4):
        super().__init__()
        self.fie_stage = FieStage(in_channels=3, channels=channels, num_fie_blocks=num_fie_blocks)
        self.denoiser = Denoiser(in_channels=6, channels=channels, num_fre_blocks=num_fre_blocks)

    def forward(self, x):
        inp = x
        x_concat, x_lol = self.fie_stage(x)
        out = self.denoiser(inp, x_concat, x_lol)
        return out, x_lol

if __name__ == "__main__":
    flol_plus = FlolPlus(channels=16, num_fie_blocks=3, num_fre_blocks=4)
    x = torch.rand(16, 3, 256, 256)
    output, _ = flol_plus(x)
    assert output.shape == torch.Size([16, 3, 256, 256])