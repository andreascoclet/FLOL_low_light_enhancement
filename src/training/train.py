import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import lpips
import pyrallis
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from src.models.model import FlolPlus
from src.datasets.flol_datasets import get_datasets
from src.utils.utils import get_scheduler, PSNRLoss, SSIMLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

@dataclass
class TrainConfig:
    data_root: str = "data"
    batch_size: int = 32
    num_workers: int = 16
    seed: int = 42
    heldout_samples: int = 50
    channels: int = 64
    num_fie_blocks: int = 3
    num_fre_blocks: int = 4
    lr: float = 8e-4
    eta_min: float = 1e-6
    num_epochs: int = 200
    patience: int = 16
    min_delta: float = 5e-4
    grad_clip: float = 0.05
    scheduler_step: bool = True
    lambda_lpips: float = 0.1
    l1_weight: float = 1 
    checkpoints_path: str = "./checkpoints"
    tensorboard_log_dir: str = "./runs"
    run_name: str = "flol-plus-run"

    def __post_init__(self):
        import uuid
        self.name = (
            f"{self.run_name}"
            f"-lr_{self.lr}"
            f"-channels_{self.channels}"
            f"-patience_{self.patience}"
            f"-lambda_{self.lambda_lpips}"
            f"-seed_{self.seed}"
            f"_{str(uuid.uuid4())[:8]}"
        )
        self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
        self.tensorboard_log_dir = os.path.join(self.tensorboard_log_dir, self.name)

        print(f"\n\n{self.name}\n\n")

@pyrallis.wrap()
def train(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}\n")

    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    start_time = datetime.now()
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training started at: {start_str}\n")

    writer = SummaryWriter(log_dir=cfg.tensorboard_log_dir)

    ROOT = cfg.data_root

    train_ds, eval_ds = get_datasets(ROOT, mode="train_eval")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = FlolPlus(
        channels=cfg.channels,
        num_fie_blocks=cfg.num_fie_blocks,
        num_fre_blocks=cfg.num_fre_blocks,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params/1e6:.3f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))#, weight_decay=0.0001)
    # scheduler = get_scheduler(optimizer, cfg.eta_min)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=cfg.eta_min)

    l1 = nn.L1Loss()
    lpips_fn = lpips.LPIPS(net="vgg")
    if device.type == "cuda":
        lpips_fn.cuda()
    lpips_fn.eval()

    psnr_metric = PSNRLoss()
    ssim_metric = SSIMLoss()

    def to_lpips(img):
        return img * 2.0 - 1.0

    os.makedirs(cfg.checkpoints_path, exist_ok=True)
    with open(os.path.join(cfg.checkpoints_path, "config.yaml"), "w") as f:
        pyrallis.dump(cfg, f)

    best_loss = float('inf')
    best_psnr = 0
    best_ssim = 0
    epochs_no_improve = 0




    for epoch in range(cfg.num_epochs):
        model.train()
        train_total_vals = []
        train_l1_dist_vals = []
        train_l1_inter_vals = []
        train_lpips_vals = []
        train_psnr_vals = []
        train_ssim_vals = []

        pbar = tqdm(train_loader,
                    desc=f"Epoch [{epoch+1}/{cfg.num_epochs}] Train",
                    ncols=200,
                    leave=True)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            output, x_lol = model(inputs)

            l1_dist = l1(output, targets)
            l1_inter = l1(x_lol, targets)
            lpips_val = lpips_fn(to_lpips(output), to_lpips(targets)).mean()
            total_loss = (
                cfg.l1_weight * l1_dist +
                cfg.l1_weight * l1_inter +
                cfg.lambda_lpips * lpips_val
            )

            psnr_val = psnr_metric(output, targets)
            ssim_val = ssim_metric(output, targets)

            train_total_vals.append(total_loss.item())
            train_l1_dist_vals.append(l1_dist.item())
            train_l1_inter_vals.append(l1_inter.item())
            train_lpips_vals.append(lpips_val.item())
            train_psnr_vals.append(psnr_val.item())
            train_ssim_vals.append(ssim_val.item())

            total_loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            if (batch_idx % 16 == 0) or (batch_idx == len(train_loader) - 1):
                pbar.set_postfix({
                    "Total": f"{np.mean(train_total_vals):.4f}",
                    "L1_dist": f"{np.mean(train_l1_dist_vals):.4f}",
                    "L1_inter": f"{np.mean(train_l1_inter_vals):.4f}",
                    "LPIPS": f"{np.mean(train_lpips_vals):.4f}",
                    "PSNR": f"{np.mean(train_psnr_vals):.4f}",
                    "SSIM": f"{np.mean(train_ssim_vals):.4f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                })

        train_total = np.mean(train_total_vals)
        train_l1_dist = np.mean(train_l1_dist_vals)
        train_l1_inter = np.mean(train_l1_inter_vals)
        train_lpips = np.mean(train_lpips_vals)
        train_psnr = np.mean(train_psnr_vals)
        train_ssim = np.mean(train_ssim_vals)

        model.eval()
        eval_total_vals = []
        eval_l1_dist_vals = []
        eval_l1_inter_vals = []
        eval_lpips_vals = []
        eval_psnr_vals = []
        eval_ssim_vals = []

        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                output, x_lol = model(inputs)

                l1_dist = l1(output, targets)
                l1_inter = l1(x_lol, targets)
                lpips_val = lpips_fn(to_lpips(output), to_lpips(targets)).mean()
                total_loss = (
                    cfg.l1_weight * l1_dist +
                    cfg.l1_weight * l1_inter +
                    cfg.lambda_lpips * lpips_val
                )

                psnr_val = psnr_metric(output, targets)
                ssim_val = ssim_metric(output, targets)

                eval_total_vals.append(total_loss.item())
                eval_l1_dist_vals.append(l1_dist.item())
                eval_l1_inter_vals.append(l1_inter.item())
                eval_lpips_vals.append(lpips_val.item())
                eval_psnr_vals.append(psnr_val.item())
                eval_ssim_vals.append(ssim_val.item())

        eval_total = np.mean(eval_total_vals)
        eval_l1_dist = np.mean(eval_l1_dist_vals)
        eval_l1_inter = np.mean(eval_l1_inter_vals)
        eval_lpips = np.mean(eval_lpips_vals)
        eval_psnr = np.mean(eval_psnr_vals)
        eval_ssim = np.mean(eval_ssim_vals)



        # if (best_psnr - eval_psnr ) > cfg.min_delta:
        if (best_loss - eval_total) > cfg.min_delta:
            best_loss = eval_total
            best_psnr = eval_psnr
            best_ssim = eval_ssim
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(cfg.checkpoints_path, "best.pt"))
        else:
            epochs_no_improve += 1

        torch.save(model.state_dict(), os.path.join(cfg.checkpoints_path, "last.pt"))
        with open(os.path.join(cfg.checkpoints_path, "metrics.txt"), "w") as f:
            f.write(f"Training started: {start_str}\n\n") 
            f.write(f"Trainable parameters: {num_params/1e6:.3f}M ({num_params:,})\n\n")
            f.write(f"Best PSNR: {best_psnr:.4f}\nBest SSIM: {best_ssim:.4f}\n\nLast PSNR: {eval_psnr:.4f}\nLast SSIM: {eval_ssim:.4f}\n")

        log_str = (
            f"\nEpoch {epoch+1}: "
            f"train_total={train_total:.4f}, train_L1_dist={train_l1_dist:.4f}, train_L1_inter={train_l1_inter:.4f}, "
            f"train_LPIPS={train_lpips:.4f}, train_PSNR={train_psnr:.4f}, train_SSIM={train_ssim:.4f}, "
            f"val_total={eval_total:.4f}, val_L1_dist={eval_l1_dist:.4f}, val_L1_inter={eval_l1_inter:.4f}, "
            f"val_LPIPS={eval_lpips:.4f}, val_PSNR={eval_psnr:.4f}, val_SSIM={eval_ssim:.4f}, "
            f"patience={epochs_no_improve}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}\n\n\n"
        )
        tqdm.write(log_str)

        writer.add_scalars("Total_Loss", {"train": train_total, "val": eval_total}, epoch)
        writer.add_scalars("L1_Distortion", {"train": train_l1_dist, "val": eval_l1_dist}, epoch)
        writer.add_scalars("L1_Intermediate", {"train": train_l1_inter, "val": eval_l1_inter}, epoch)
        writer.add_scalars("LPIPS", {"train": train_lpips, "val": eval_lpips}, epoch)
        writer.add_scalars("PSNR", {"train": train_psnr, "val": eval_psnr}, epoch)
        writer.add_scalars("SSIM", {"train": train_ssim, "val": eval_ssim}, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        if cfg.scheduler_step:
            scheduler.step()

        if epochs_no_improve >= cfg.patience:
            tqdm.write(f"Early stopping {epoch+1} epoch ")
            break

    writer.add_scalar("best_val_PSNR", best_psnr)
    writer.flush()
    writer.close()

    print(f" finished run {cfg.name}")

if __name__ == "__main__":
    train()