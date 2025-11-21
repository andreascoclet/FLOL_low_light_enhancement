#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2



train_transform = A.Compose(
    [
        A.RandomCrop(256, 256, p=1),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.OneOf(
            [
                A.Rotate(limit=(90, 90), p=0.5),
                A.Rotate(limit=(180, 180), p=0.5),
                A.Rotate(limit=(270, 270), p=0.5),
            ],
            p=0.5, 
        ),
        ToTensorV2(),
    ],
    additional_targets={"image1": "image"},
)

val_test_transform = A.Compose(
    [A.RandomCrop(256, 256, p=1.0),
    ToTensorV2()],
    additional_targets={"image1": "image"},
)

class LowLightDataset(Dataset):
    EXT_MAP = {
        "DICM":   (".jpg", ".jpeg"),
        "LIME":   (".bmp",),
        "low":    (".jpeg", ".bmp"),
        "MEF":    (".png",),
        "NPE":    (".jpg", ".bmp"),
        "VV":     (".jpg",),
        "UHD-LL": (".jpg",),
        "default": (".png", ".jpg", ".jpeg", ".bmp"),
    }

    def __init__(self,
                 root_dir: str,
                 mode: str = "train",
                 subfolder: Optional[str] = None,   # for the 7 test-sets
                 transform: Optional[A.Compose] = None,
                 seed: int = 42):
        
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.root = Path(root_dir).expanduser().resolve()
        self.mode = mode.lower()
        self.subfolder = subfolder
        self.transform = transform
        self.pairs: List[Tuple[Path, Optional[Path]]] = []
        self._build_pairs()

    def _glob(self, folder: Path, exts: Tuple[str, ...]) -> List[Path]:
        return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]

    def _paired(self, low_dir: Path, high_dir: Path, exts: Tuple[str, ...]) -> List[Tuple[Path, Path]]:
        pairs = []
        low_files = self._glob(low_dir, exts)
        # print(f"[DEBUG] {low_dir} → {len(low_files)} files")
        for low in low_files:
            high = high_dir / low.name
            if high.exists():
                pairs.append((low, high))
            else:
                print(f"  [MISSING] {high}")
                pass
        # print(f"[DEBUG] Paired: {len(pairs)} / {len(low_files)}")
        return sorted(pairs)
    
    def _paired_real_captured(self, low_dir: Path, high_dir: Path) -> List[Tuple[Path, Path]]:
        """Pair lowXXXXX.png with normalXXXXX.png (strip prefixes)."""
        pairs = []
        low_files = [p for p in low_dir.iterdir() if p.suffix.lower() == ".png"]
        for low in sorted(low_files):
            # low00001.png → 00001
            num = low.stem.replace("low", "")
            high = high_dir / f"normal{num}.png"
            if high.exists():
                pairs.append((low, high))
        return pairs
    
    def _repeat(self, pairs: List[Tuple[Path, Path]], target: int) -> List[Tuple[Path, Path]]:
        """
        Randomly resample (with replacement) LOL-v2 pairs until we have exactly `target` pairs.
        """
        if len(pairs) == 0:
            return []
        rng = random.Random()                # <-- same seed as the dataset
        return rng.choices(pairs, k=target)

    def _build_pairs(self):
        if self.mode == "train":

            # for cam in ("Huawei", "Nikon"):
            #             lsrw_low  = self.root / "LSRW" / "Train" / cam / "low"
            #             lsrw_high = self.root / "LSRW" / "Train" / cam / "high"
            #             self.pairs.extend(self._paired(lsrw_low, lsrw_high, (".jpg",)))

            lolv2_real_pairs = []
            lolv2_syn_pairs = []
            lsrw_huawei_pairs = []
            lsrw_nikon_pairs = []
            uhd_ll_pairs = []

            lsrw_huawei_low  = self.root / "LSRW" / "Train" / "Huawei" / "low"
            lsrw_huawei_high = self.root / "LSRW" / "Train" / "Huawei" / "high"
            lsrw_huawei_pairs.extend(self._paired(lsrw_huawei_low, lsrw_huawei_high, (".jpg",)))

            lsrw_nikon_low  = self.root / "LSRW" / "Train" / "Nikon" / "low"
            lsrw_nikon_high = self.root / "LSRW" / "Train" / "Nikon" / "high"
            lsrw_nikon_pairs.extend(self._paired(lsrw_nikon_low, lsrw_nikon_high, (".jpg",)))

            real_low  = self.root / "LOL-v2" / "Real_captured" / "Train" / "Low"
            real_high = self.root / "LOL-v2" / "Real_captured" / "Train" / "Normal"
            if real_low.exists() and real_high.exists():
                lolv2_real_pairs.extend(self._paired_real_captured(real_low, real_high))

            syn_low  = self.root / "LOL-v2" / "Synthetic" / "Train" / "Low"
            syn_high = self.root / "LOL-v2" / "Synthetic" / "Train" / "Normal"
            lolv2_syn_pairs.extend(self._paired(syn_low, syn_high, (".png",)))
            


            lolv2_pairs = self._repeat(lolv2_real_pairs, len(lsrw_nikon_pairs)) + \
                                self._repeat(lolv2_syn_pairs, len(lsrw_nikon_pairs))
            
            lsrw_pairs = lsrw_nikon_pairs  + \
                                self._repeat(lsrw_huawei_pairs, len(lsrw_nikon_pairs))
            
            #uhd_ll_pairs = self._repeat(uhd_ll_pairs)

            self.pairs = lolv2_pairs[:] + lsrw_pairs[:]

            del (
                lolv2_real_pairs,
                lolv2_syn_pairs,
                lsrw_huawei_pairs,
                lsrw_nikon_pairs,
                lsrw_pairs,
                lolv2_pairs
            )

        elif self.mode == "eval":
            if not self.subfolder:
                raise ValueError("subfolder must be provided for mode='eval' (use 'lol-v2', 'lol-v2-real', 'lol-v2-syn', 'lsrw', or 'both')")
            sub = self.subfolder.lower()

            # ---------- LOL-v2 Real only ----------
            if sub in{"lol-v2-real", "both", "lol-v2"}:
                low_dir  = self.root / "LOL-v2" / "Real_captured" / "Test" / "Low"
                high_dir = self.root / "LOL-v2" / "Real_captured" / "Test" / "Normal"
                if low_dir.exists() and high_dir.exists():
                    self.pairs.extend(self._paired_real_captured(low_dir, high_dir))

            # ---------- LOL-v2 Synthetic only ----------
            if sub in {"lol-v2-syn", "both", "lol-v2"}:   # "lol-v2" also includes syn
                syn_low  = self.root / "LOL-v2" / "Synthetic" / "Test" / "Low"
                syn_high = self.root / "LOL-v2" / "Synthetic" / "Test" / "Normal"
                self.pairs.extend(self._paired(syn_low, syn_high, (".png",)))

            # ---------- LSRW ----------
            if sub in {"lsrw", "both"}:
                for cam in ("Huawei", "Nikon"):
                    lsrw_low  = self.root / "LSRW" / "Eval" / cam / "low"
                    lsrw_high = self.root / "LSRW" / "Eval" / cam / "high"
                    self.pairs.extend(self._paired(lsrw_low, lsrw_high, (".jpg",)))

        elif self.mode == "test_lol_sub":
            if not self.subfolder:
                raise ValueError("subfolder must be set for mode='test_lol_sub'")
            low_dir = self.root / "Test_Quality_v2" / self.subfolder
            if not low_dir.exists():
                return
            exts = self.EXT_MAP.get(self.subfolder, self.EXT_MAP["default"])
            for low_path in self._glob(low_dir, exts):
                self.pairs.append((low_path, None))
        
        elif self.mode == "uhd_ll_train":
            low_dir = self.root / "UHD-LL_train" / "input"
            gt_dir  = self.root / "UHD-LL_train" / "gt"
            if low_dir.exists():
                exts = self.EXT_MAP.get("UHD-LL", self.EXT_MAP["default"])
                self.pairs.extend(self._paired(low_dir, gt_dir, exts))

        elif self.mode == "uhd_ll_test":
            low_dir = self.root / "UHD-LL_test" / "input"
            gt_dir  = self.root / "UHD-LL_test" / "gt"
            if low_dir.exists():
                exts = self.EXT_MAP.get("UHD-LL", self.EXT_MAP["default"])
                self.pairs.extend(self._paired(low_dir, gt_dir, exts))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        low_path, high_path = self.pairs[idx]
        low_img  = np.array(Image.open(low_path).convert("RGB"))
        high_img = np.array(Image.open(high_path).convert("RGB")) if high_path else np.zeros_like(low_img)

        if self.transform:
            low_img  = low_img.astype(np.float32) / 255.0
            high_img = high_img.astype(np.float32) / 255.0
            aug = self.transform(image=low_img, image1=high_img)
            low_img, high_img = aug["image"], aug["image1"]
        else:
            low_img  = torch.from_numpy(low_img.transpose(2, 0, 1)).float() / 255.0
            high_img = torch.from_numpy(high_img.transpose(2, 0, 1)).float() / 255.0

        return low_img, high_img


def get_datasets(
    root_dir: str,
    mode: str = "train",
    subfolder: Optional[str] = None,
) -> Union[LowLightDataset, Tuple[LowLightDataset, LowLightDataset]]:
    mode = mode.lower()

    if mode == "train_eval":
        train_ds = LowLightDataset(root_dir, mode="train", transform=train_transform)
        eval_ds  = LowLightDataset(root_dir, mode="eval", subfolder="both", transform=val_test_transform)
        return train_ds, eval_ds

    if mode == "train":
        return LowLightDataset(root_dir, mode="train", transform=train_transform)

    if mode == "eval":
        if not subfolder:
            raise ValueError(
                "subfolder required for mode='eval'. "
                "Options: 'lol-v2', 'lol-v2-real', 'lol-v2-syn', 'lsrw', 'both'"
            )
        return LowLightDataset(root_dir, mode="eval", subfolder=subfolder, transform=val_test_transform)

    if mode == "test_lol_sub":
        return LowLightDataset(root_dir, mode="test_lol_sub", subfolder=subfolder, transform=val_test_transform)

    if mode == "uhd_ll_test":
        return LowLightDataset(root_dir, mode="uhd_ll_test", transform=val_test_transform)
    
    if mode == "uhd_ll_train_test":
        return  LowLightDataset(root_dir, mode="uhd_ll_train", transform=val_test_transform)  , LowLightDataset(root_dir, mode="uhd_ll_test", transform=val_test_transform)

    raise ValueError(f"Unknown mode: {mode}")



if __name__ == "__main__":
    ROOT = "data"
    TEST_SUBS = ["DICM", "LIME", "MEF", "NPE", "VV"]

    print("\n" + "="*70)
    print(" LowLightDataset – Full Verification")
    print("="*70 + "\n")

    train_ds, eval_ds = get_datasets(ROOT, mode="train_eval")
    print(f"Train (LOL-v2 + LSRW)         : {len(train_ds):>6} pairs")
    print(f"Eval  (LOL-v2 + LSRW)         : {len(eval_ds):>6} pairs")

    train_only = get_datasets(ROOT, mode="train")
    print(f"Train only                    : {len(train_only):>6}")

    print("\n--- Evaluation Subsets ---")
    eval_lol_real = get_datasets(ROOT, mode="eval", subfolder="lol-v2-real")
    eval_lol_syn  = get_datasets(ROOT, mode="eval", subfolder="lol-v2-syn")
    eval_lol      = get_datasets(ROOT, mode="eval", subfolder="lol-v2")
    eval_lsrw     = get_datasets(ROOT, mode="eval", subfolder="lsrw")
    eval_both     = get_datasets(ROOT, mode="eval", subfolder="both")

    print(f"Eval LOL-v2 Real              : {len(eval_lol_real):>6}")
    print(f"Eval LOL-v2 Syn               : {len(eval_lol_syn):>6}")
    print(f"Eval LOL-v2 (Real+Syn)        : {len(eval_lol):>6}")
    print(f"Eval LSRW                     : {len(eval_lsrw):>6}")
    print(f"Eval (LOL-v2 + LSRW)          : {len(eval_both):>6}")

    print("\n--- UHD-LL Dataset ---")
    uhd_train_ds, uhd_test_ds = get_datasets(ROOT, mode="uhd_ll_train_test")
    print(f"UHD-LL Train                  : {len(uhd_train_ds):>6} pairs")
    print(f"UHD-LL Test                   : {len(uhd_test_ds):>6} pairs")
    print(f"UHD-LL Total                  : {len(uhd_train_ds) + len(uhd_test_ds):>6} pairs")

    uhd_test_only = get_datasets(ROOT, mode="uhd_ll_test")
    print(f"UHD-LL Test (single)          : {len(uhd_test_only):>6} pairs")

    print("\n--- Qualitative Test Subsets ---")
    for sub in TEST_SUBS:
        ds = get_datasets(ROOT, mode="test_lol_sub", subfolder=sub)
        print(f" Test/{sub:<6}                 : {len(ds):>6} images")

    assert len(train_ds) > 0
    assert len(eval_ds) > 0
    assert len(eval_lol_real) > 0
    assert len(eval_lol_syn) > 0
    assert len(uhd_train_ds) > 0
    assert len(uhd_test_ds) > 0

    print("\n" + "SUCCESS: All datasets loaded and verified correctly!")
    print("="*70)