"""Build ``data/eurosat_rgb.npz`` (images + fixed split) from EuroSAT RGB.

Examples
--------
Download through torchvision (about 90 MB, from the torchgeo mirror on Hugging Face):
    python -m eurofm.prepare_data --source torchvision --root data/raw

Use an already extracted copy (e.g. a Kaggle dataset):
    python -m eurofm.prepare_data --source folder --root /kaggle/input/eurosat-dataset

Tiny synthetic stand-in, only for smoke tests:
    python -m eurofm.prepare_data --source synthetic --out data/synthetic.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .data import CLASS_NAMES, build_split, find_class_root


def load_folder(root: Path) -> tuple[np.ndarray, np.ndarray]:
    class_root = find_class_root(root)
    images, labels = [], []
    for label, name in enumerate(CLASS_NAMES):
        files = sorted(p for p in (class_root / name).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif"})
        for f in files:
            with Image.open(f) as im:
                images.append(np.asarray(im.convert("RGB").resize((64, 64))))
            labels.append(label)
        print(f"  {name:22s} {len(files):5d} images")
    return np.stack(images).astype(np.uint8), np.asarray(labels, dtype=np.int64)


def make_synthetic(per_class: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Colour + stripe-orientation patterns: learnable, but not trivially so. Smoke tests only."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:64, 0:64]
    images, labels = [], []
    for c in range(len(CLASS_NAMES)):
        base = rng.uniform(40, 215, size=3)
        angle = np.pi * c / len(CLASS_NAMES)
        stripes = np.sin((np.cos(angle) * xx + np.sin(angle) * yy) / 3.0)[..., None]
        for _ in range(per_class):
            img = base + 35 * stripes + rng.normal(0, 25, size=(64, 64, 3))
            images.append(np.clip(img, 0, 255).astype(np.uint8))
            labels.append(c)
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["torchvision", "folder", "synthetic"], default="torchvision")
    ap.add_argument("--root", default="data/raw", help="download dir (torchvision) or extracted dataset dir (folder)")
    ap.add_argument("--out", default="data/eurosat_rgb.npz")
    ap.add_argument("--test-per-class", type=int, default=200)
    ap.add_argument("--synthetic-per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.source == "synthetic":
        images, labels = make_synthetic(args.synthetic_per_class, args.seed)
        test_per_class = min(args.test_per_class, args.synthetic_per_class // 3)
    else:
        root = Path(args.root)
        if args.source == "torchvision":
            from torchvision.datasets import EuroSAT

            EuroSAT(root=str(root), download=True)
        print(f"Reading images from {root} ...")
        images, labels = load_folder(root)
        test_per_class = args.test_per_class

    test_idx, pool_idx = build_split(labels, test_per_class, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=images,
        labels=labels,
        test_idx=test_idx,
        pool_idx=pool_idx,
        class_names=np.asarray(CLASS_NAMES),
    )
    print(f"Saved {out}: {len(labels)} images, test={len(test_idx)}, pool={len(pool_idx)}")


if __name__ == "__main__":
    main()
