"""Run each frozen encoder once over all EuroSAT images and cache the features.

    python -m eurofm.extract_features --encoders dinov3_vitl16_sat dinov3_vitl16_web clip_vitb16

Features go to ``features/<encoder>.npz``; linear probes and zero-shot evaluation read them,
so every later experiment on that encoder takes seconds instead of re-running the network.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import data
from .common import count_params, device, device_name
from .encoders import ENCODERS, load_encoder

FEATURE_DIR = Path(os.environ.get("EUROFM_FEATURES", "features"))


def feature_path(key: str) -> Path:
    return FEATURE_DIR / f"{key}.npz"


def load_features(key: str) -> tuple[np.ndarray, dict]:
    path = feature_path(key)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m eurofm.extract_features --encoders {key}` first.")
    z = np.load(path)
    return z["features"].astype(np.float32), json.loads(str(z["meta"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    ap.add_argument("--data", default=str(data.DEFAULT_DATA))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-pretrained", action="store_true", help="random weights (offline smoke test only)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ds = data.load(args.data)
    dev = device()
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    for key in args.encoders:
        out = feature_path(key)
        if out.exists() and not args.overwrite:
            print(f"{out} exists, skipping (use --overwrite to redo)")
            continue
        print(f"== {key}: {ENCODERS[key]['desc']}")
        enc = load_encoder(key, pretrained=not args.no_pretrained, device=dev)
        enc.encode_images(ds.images[: args.batch_size], args.batch_size)  # warm-up (cuDNN autotune, lazy init)
        t0 = time.perf_counter()
        feats = enc.encode_images(ds.images, args.batch_size)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        meta = dict(
            encoder=key,
            feature_dim=int(feats.shape[1]),
            total_params=count_params(enc.model)[0],
            ms_per_image=1000 * seconds / len(ds.images),
            image_size=enc.image_size,
            pretrained=not args.no_pretrained,
            device=device_name(),
        )
        np.savez(out, features=feats.astype(np.float16), meta=json.dumps(meta))
        print(f"   {feats.shape} in {seconds:.1f}s ({meta['ms_per_image']:.2f} ms/img) -> {out}")
        del enc
        if dev.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
