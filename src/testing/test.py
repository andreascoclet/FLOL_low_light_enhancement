# src/testing/test.py
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pyrallis
from dataclasses import dataclass
import yaml
from PIL import Image
from piq import brisque


# Local imports
from src.models.model import FlolPlus
from src.datasets.flol_datasets import get_datasets
from src.utils.utils import PSNRLoss, SSIMLoss


@dataclass
class TestConfig:
    checkpoint_dir: str = "checkpoints/flol-plus-run_uhd_ll-lr_0.0004-channels_16-patience_32-lambda_0.1-seed_42_98660ab6"
    results_dir: str = "results"
    batch_size: int = 8
    num_vis: int = 4
    seed: int = 42


def tensor_to_np(img_tensor):
    """[C,H,W] in [0,1] → HWC uint8 numpy (RGB)"""
    return (img_tensor.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def hstack_images(img_list, gap=10):
    """Horizontally stack a list of images with a gap (in pixels)."""
    h, w = img_list[0].shape[:2]
    total_w = w * len(img_list) + gap * (len(img_list) - 1)
    canvas = np.full((h, total_w, 3), 255, dtype=np.uint8)
    x = 0
    for img in img_list:
        canvas[:, x:x + w] = img
        x += w + gap
    return canvas


def vstack_images(img_list, gap=10):
    """Vertically stack a list of images with a gap (in pixels)."""
    h, w = img_list[0].shape[:2]
    total_h = h * len(img_list) + gap * (len(img_list) - 1)
    canvas = np.full((total_h, w, 3), 255, dtype=np.uint8)
    y = 0
    for img in img_list:
        canvas[y:y + h, :] = img
        y += h + gap
    return canvas


@pyrallis.wrap()
def test(cfg: TestConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}\n")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    ckpt_dir = Path(cfg.checkpoint_dir)
    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {ckpt_dir}")

    # Load config
    cfg_path = ckpt_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {ckpt_dir}")
    with open(cfg_path, "r") as f:
        train_cfg = yaml.safe_load(f)
    print(f"Loaded config from {cfg_path}")

    # Rebuild model
    model = FlolPlus(
        channels=train_cfg["channels"],
        num_fie_blocks=train_cfg["num_fie_blocks"],
        num_fre_blocks=train_cfg["num_fre_blocks"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params/1e6:.3f}M")
    
    # Load best weights
    best_pt = ckpt_dir / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found in {ckpt_dir}")
    model.load_state_dict(torch.load(best_pt, map_location=device))
    model.eval()
    print(f"Loaded weights from {best_pt}")

    # Results directory
    results_root = Path(cfg.results_dir) / ckpt_dir.name
    results_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {results_root}")

    # Metrics
    psnr_loss = PSNRLoss()
    ssim_loss = SSIMLoss()

    quant_sets = {
        "LOL-v2 Real"       : ("eval", "lol-v2-real"),   # Real_captured/Test only
        "LOL-v2 Syn"        : ("eval", "lol-v2-syn"),    # Synthetic/Test only
        "LOL-v2"            : ("eval", "lol-v2"),        # Real + Syn
        "LSRW"              : ("eval", "lsrw"),          # LSRW Eval
        "LOL-v2_LSRW"       : ("eval", "both"),           # LOL-v2 Real + LOL-v2 Syn + LSRW
        "UHD-LL_test"       : ("uhd_ll_test", None),     # unchanged
    }

    quant_results = {}

    print("\n=== Quantitative evaluation ===")
    for name, (mode, sub) in quant_sets.items():
        dataset = get_datasets(
            root_dir=train_cfg["data_root"],
            mode=mode,
            subfolder=sub
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False
        )
        psnrs, ssims = [], []

        # Random indices for visualisation
        indices = np.random.choice(len(dataset), cfg.num_vis, replace=False)
        vis_samples = []

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=f"Eval {name}", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                output, _ = model(inputs)
                psnrs.append(-psnr_loss(output, targets).item())
                ssims.append(ssim_loss(output, targets).item())

        mean_psnr = np.mean(psnrs)
        mean_ssim = np.mean(ssims)
        quant_results[name] = (mean_psnr, mean_ssim)
        print(f"{name:12} PSNR: {mean_psnr:.3f} | SSIM: {mean_ssim:.4f}")

        # Collect the 4 samples
        for i in indices:
            inp, gt = dataset[i]
            with torch.no_grad():
                pred, _ = model(inp.unsqueeze(0).to(device))
                pred = pred.squeeze(0).cpu()
            vis_samples.append((inp, pred, gt))

        # Build 3×4 grid
        inputs_np   = [tensor_to_np(inp)   for inp, _, _   in vis_samples]
        predicts_np = [tensor_to_np(pred)  for _, pred, _ in vis_samples]
        gts_np      = [tensor_to_np(gt)    for _, _, gt    in vis_samples]

        row_input   = hstack_images(inputs_np)
        row_predict = hstack_images(predicts_np)
        row_gt      = hstack_images(gts_np)

        grid = vstack_images([row_input, row_predict, row_gt])
        Image.fromarray(grid).save(
            results_root / f"visualization_quant_{name.replace('-', '_')}.png"
        )

    TEST_SUBS = ["DICM", "LIME", "MEF", "NPE", "VV"]
    brisque_scores = {}
    print("\n=== Qualitative evaluation (BRISQUE) ===")
    for sub in TEST_SUBS:
        dataset = get_datasets(
            root_dir=train_cfg["data_root"],
            mode="test_lol_sub",
            subfolder=sub
        )
        if len(dataset) == 0:
            print(f"{sub}: empty")
            continue

        indices = np.random.choice(len(dataset), min(cfg.num_vis, len(dataset)), replace=False)
        vis_samples = []
        brisques = []

        for i, (inp, _) in enumerate(dataset):
            with torch.no_grad():
                pred, _ = model(inp.unsqueeze(0).to(device))
                pred_img = pred.clamp(0, 1)

            # Try BRISQUE, skip on error
            try:
                score = brisque(pred_img).item()
                brisques.append(score)
            except Exception as e:
                print(f"[BRISQUE] Skipping {sub} sample {i}: {e}")
                continue

            # Store visualization samples
            if i in indices:
                vis_samples.append((inp, pred.squeeze(0).cpu()))

        mean_brisque = np.mean(brisques)
        brisque_scores[sub] = mean_brisque
        print(f"{sub:6} BRISQUE: {mean_brisque:.2f}")

        inputs_np   = [tensor_to_np(inp)   for inp, _   in vis_samples]
        predicts_np = [tensor_to_np(pred)  for _, pred  in vis_samples]

        row_input   = hstack_images(inputs_np)
        row_predict = hstack_images(predicts_np)

        grid = vstack_images([row_input, row_predict])
        Image.fromarray(grid).save(
            results_root / f"visualization_qual_{sub}.png"
        )

    with open(results_root / "quantitative.txt", "w") as f:
        f.write("Dataset      | PSNR   | SSIM\n")
        f.write("-" * 30 + "\n")
        for name, (psnr, ssim) in quant_results.items():
            f.write(f"{name:12} | {psnr:6.3f} | {ssim:.4f}\n")

    with open(results_root / "qualitative_brisque.txt", "w") as f:
        f.write("Subset | BRISQUE\n")
        f.write("-" * 20 + "\n")
        for sub, score in brisque_scores.items():
            f.write(f"{sub:6} | {score:6.2f}\n")

    print(f"\nAll grid PNGs and tables saved to: {results_root}")
    print("\n=== FINAL RESULTS ===")
    print("Quantitative:")
    for k, v in quant_results.items():
        print(f"  {k:12}: PSNR={v[0]:.3f}, SSIM={v[1]:.4f}")
    print("Qualitative BRISQUE:")
    for k, v in brisque_scores.items():
        print(f"  {k:6}: {v:.2f}")


if __name__ == "__main__":
    test()