import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.utils.fre_block import FreBlock


class FreqStage(nn.Module):
    def __init__(self, channels: int = 6, num_fre_blocks: int = 4) -> None:
        super().__init__()
        self.fre_blocks = nn.Sequential(*[FreBlock(channels=channels) for _ in range(num_fre_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
         x = self.fre_blocks(x)
         return x
        

if __name__ == "__main__":
    freq_stage = FreqStage(channels=6, num_fre_blocks=4)
    x = torch.rand(16, 6, 256, 256)
    x = freq_stage(x)
    assert x.shape == torch.Size([16, 6, 256, 256])


