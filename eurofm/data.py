"""EuroSAT loading, the fixed train-pool/test split, and k-shot sampling.

Everything downstream reads one compact file, ``data/eurosat_rgb.npz``:

    images     uint8 [N, 64, 64, 3]
    labels     int64 [N]
    test_idx   int64 [n_test]    fixed test set (same for every method)
    pool_idx   int64 [n_pool]    labelled pool that k-shot training sets are drawn from

Build it once with ``python -m eurofm.prepare_data``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Folder names used by the official EuroSAT release, in the canonical (alphabetical) order.
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Natural-language names used in text prompts (CLIP-style zero-shot and VLM prompts).
CLASS_DESCRIPTIONS = {
    "AnnualCrop": "annual crop land",
    "Forest": "forest",
    "HerbaceousVegetation": "herbaceous vegetation land",
    "Highway": "highway or road",
    "Industrial": "industrial buildings",
    "Pasture": "pasture land",
    "PermanentCrop": "permanent crop land",
    "Residential": "residential buildings",
    "River": "river",
    "SeaLake": "sea or lake",
}

DEFAULT_DATA = Path(os.environ.get("EUROFM_DATA", "data/eurosat_rgb.npz"))


@dataclass
class EuroSAT:
    images: np.ndarray
    labels: np.ndarray
    test_idx: np.ndarray
    pool_idx: np.ndarray
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def load(path: str | Path = DEFAULT_DATA) -> EuroSAT:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m eurofm.prepare_data` first.")
    z = np.load(path, allow_pickle=False)
    return EuroSAT(
        images=z["images"],
        labels=z["labels"].astype(np.int64),
        test_idx=z["test_idx"].astype(np.int64),
        pool_idx=z["pool_idx"].astype(np.int64),
        class_names=[str(c) for c in z["class_names"]],
    )


def find_class_root(root: str | Path) -> Path:
    """Return the directory that directly contains the 10 EuroSAT class folders.

    Works for the torchvision layout (``eurosat/2750/<Class>``), the Zenodo layout
    (``EuroSAT_RGB/<Class>``) and most Kaggle mirrors.
    """
    root = Path(root)
    for cand in [root, *sorted(p.parent for p in root.rglob("AnnualCrop") if p.is_dir())]:
        if all((cand / c).is_dir() for c in CLASS_NAMES):
            return cand
    raise FileNotFoundError(f"Could not find the EuroSAT class folders under {root}")


def build_split(labels: np.ndarray, test_per_class: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split: ``test_per_class`` images of each class go to the test set, the rest to the pool."""
    rng = np.random.default_rng(seed)
    test, pool = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        test.append(idx[:test_per_class])
        pool.append(idx[test_per_class:])
    return np.sort(np.concatenate(test)), np.sort(np.concatenate(pool))


def sample_kshot(ds: EuroSAT, k: int | None, seed: int) -> np.ndarray:
    """Draw ``k`` labelled images per class from the pool (``k=None`` means the whole pool).

    For a fixed seed the draws are nested: the 5-shot set contains the 1-shot set, and so on,
    so learning curves differ only by the amount of data, not by which images were drawn.
    """
    if k is None:
        return ds.pool_idx.copy()
    rng = np.random.default_rng(1000 + seed)
    pool_labels = ds.labels[ds.pool_idx]
    out = []
    for c in range(ds.num_classes):
        members = ds.pool_idx[pool_labels == c]
        out.append(rng.permutation(members)[:k])
    return np.sort(np.concatenate(out))


def subsample_test(ds: EuroSAT, per_class: int | None) -> np.ndarray:
    """First ``per_class`` test images of each class (deterministic); ``None`` keeps the full test set."""
    if per_class is None:
        return ds.test_idx
    test_labels = ds.labels[ds.test_idx]
    return np.sort(np.concatenate([ds.test_idx[test_labels == c][:per_class] for c in range(ds.num_classes)]))


def parse_budgets(text: str) -> list[int | None]:
    """'1,5,10,all' -> [1, 5, 10, None]."""
    out: list[int | None] = []
    for tok in text.split(","):
        tok = tok.strip().lower()
        if tok:
            out.append(None if tok == "all" else int(tok))
    return out


def budget_name(k: int | None) -> str:
    return "all" if k is None else str(k)
