import os
import math 
import requests
import numpy as np 
import torch, torch.nn as nn
import torch.nn.functional as F
from urllib.parse import urlencode
from torch.optim.lr_scheduler import _LRScheduler
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm
from math import log10, sqrt 


def test_model(model, device, test_dataloader):
    PSNRs = []
    SSIMs = []
    MSEs = []
    for inputs, targets in tqdm(test_dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs).detach().cpu().numpy()
        targets = targets.numpy()
        inputs = inputs.detach().cpu().numpy()
        outputs.shape = 3, 720, 1280 #3, 512, 512
        targets.shape = 3, 720, 1280 #3, 512, 512
        inputs.shape = 3, 720, 1280 #3, 512, 512
        outputs = np.transpose(outputs, axes=[1, 2, 0])
        inputs = np.transpose(inputs, axes=[1, 2, 0])
        targets = np.transpose(targets, axes=[1, 2, 0])
        l_2 = mse(outputs, targets)
        MSEs.append(l_2)
        psnr = 10 * np.mean(np.log10(l_2 +  1e-8))
        PSNRs.append(psnr)
        sim = ssim(outputs, targets, channel_axis=-1, data_range=1)
        SSIMs.append(sim)
    print(f"Mean SSIM: {np.mean(SSIMs)}")
    print(f"Mean PSNR: {np.mean(PSNRs)}")
    print(f"Mean MSE: {np.mean(MSEs)}")
    return np.mean(MSEs), np.mean(PSNRs), np.mean(SSIMs)


def get_position_from_periods(iteration, cumulative_period):
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=(1, ),
                 eta_min=1e-6,
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


def get_scheduler(optimizer,eta_min=1e-6):
    return CosineAnnealingRestartLR(optimizer, periods = [10] * 20,
                                    restart_weights = [1, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05],
                                    eta_min=eta_min)
    
# 

class PSNRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-12

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
        psnr = 10 * torch.mean(torch.log10(mse + self.eps))
        return psnr



class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) Loss."""

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred, target):
        channel = pred.size(1)
        window = torch.ones((channel, 1, self.window_size, self.window_size), device=pred.device, dtype=pred.dtype)
        window /= self.window_size ** 2
        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )

        #return 1 - ssim_map.mean()
        return ssim_map.mean()