import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

class SNRMap(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.5):
        super(SNRMap, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, x_lol):
        gray = x_lol
        free_noise_gray = kornia.filters.gaussian_blur2d(gray, (self.kernel_size, self.kernel_size), (self.sigma, self.sigma))
        gray = gray[:, 0:1, :, :] * 0.299 + gray[:, 1:2, :, :] * 0.587 + gray[:, 2:3, :, :] * 0.114
        free_noise_gray = free_noise_gray[:, 0:1, :, :] * 0.299 + free_noise_gray[:, 1:2, :, :] * 0.587 + free_noise_gray[:, 2:3, :, :] * 0.114
        noise = torch.abs(gray - free_noise_gray)
        epsilon = 1e-4
        mask = torch.div(free_noise_gray,noise + epsilon)
        batch_size, _, height, width = mask.shape
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1).repeat(1, 1, height, width)
        mask = mask / (mask_max + epsilon)
        mask = torch.clamp(mask, min=0.0, max=1.0)
        mask = F.interpolate(mask, size=(gray.shape[2], gray.shape[3]), mode='nearest')
        return mask.float()
    

if __name__ == "__main__":
    # Simple sanity test
    x_lol = torch.rand(2, 3, 256, 256)  # [batch_size, 3, H, W]
    snr_map = SNRMap(kernel_size=5, sigma=1.5)

    assert x_lol.shape == (2, 3, 256, 256), f"Expected input shape [2, 3, 256, 256], got {x_lol.shape}"

    with torch.no_grad():
        mask = snr_map(x_lol)

    assert mask.shape == (2, 1, 256, 256), f"Expected output shape [2, 1, 256, 256], got {mask.shape}"
    assert mask.min() >= 0 and mask.max() <= 1, f"Mask values out of range [0, 1]: min={mask.min()}, max={mask.max()}"
