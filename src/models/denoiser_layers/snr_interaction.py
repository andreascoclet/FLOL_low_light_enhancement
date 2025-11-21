import torch
import torch.nn as nn
import torch.nn.functional as F

class SNRInteraction(nn.Module):
    def __init__(self) -> None:        
        super().__init__()

    def forward(self, o_s: torch.Tensor, o_f: torch.Tensor, snr_map: torch.Tensor) -> torch.Tensor:
        # Get  dimensions of o_s and o_f
        _, _, H_prime, W_prime = o_s.shape
        # Resize snr_map to match o_s and o_f spatial dimensions
        snr_map = F.interpolate(snr_map, size=(H_prime, W_prime), mode='nearest')#, align_corners=False)
        # Computepy features: F = O_s * snr_map + O_f * (1 - snr_map)
        features = o_s * snr_map + o_f * (1 - snr_map)
        return features

if __name__ == "__main__":
    sir_interaction = SNRInteraction()
    x = torch.rand(16, 3, 256, 256)
    o_s = torch.rand(16, 16, 128, 128)
    o_f = torch.rand(16, 16, 128, 128)
    snr_map = torch.rand(16, 1, 256, 256)
    features = sir_interaction(o_s, o_f, snr_map)
    assert features.shape == torch.Size([16, 16, 128, 128])