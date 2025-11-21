import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.5):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.conv = nn.Conv2d(1, 1, kernel_size, stride=1, padding=kernel_size//2, bias=None, groups=1)
        self.weights_init()

    def weights_init(self):
        size = self.kernel_size
        sigma = self.sigma
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        k = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
        k = k / k.sum() 
        k_tensor = torch.from_numpy(k).float().unsqueeze(0).unsqueeze(0)  
        for name, param in self.named_parameters():
            if 'weight' in name:
                param.data.copy_(k_tensor)
                param.requires_grad = False

    def forward(self, x):
        x_blurred = self.conv(x)
        return x_blurred

class SNRMap(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.5):
        super(SNRMap, self).__init__()
        self.blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, x_lol):
        gray = x_lol[:, 0:1, :, :] * 0.299 + x_lol[:, 1:2, :, :] * 0.587 + x_lol[:, 2:3, :, :] * 0.114
        free_noise_gray = self.blur(gray)
        #flip the two previous lines
        noise = torch.abs(gray - free_noise_gray)
        epsilon = 1e-4
        mask = gray / (noise + epsilon)
        batch_size, _, height, width = mask.shape
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1).repeat(1, 1, height, width)
        mask = mask / (mask_max + epsilon)
        mask = torch.clamp(mask, min=0.0, max=1.0)
        mask = F.interpolate(mask, size=(gray.shape[2], gray.shape[3]), mode='nearest')
        return mask.float()

if __name__ == "__main__":
    x_lol = torch.rand(2, 3, 256, 256)  
    snr_map = SNRMap(kernel_size=5, sigma=1.5)
    assert x_lol.shape == (2, 3, 256, 256), f"Expected input shape [2, 3, 256, 256], got {x_lol.shape}"
    with torch.no_grad():
        mask = snr_map(x_lol)
    assert mask.shape == (2, 1, 256, 256), f"Expected output shape [2, 1, 256, 256], got {mask.shape}"
    assert mask.min() >= 0 and mask.max() <= 1, f"Mask values out of range [0, 1]: min={mask.min()}, max={mask.max()}"
