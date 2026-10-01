"""Zero-shot classification with image-text encoders: no labelled images at all.

    python -m eurofm.zero_shot --encoders clip_vitb16 siglip2_vitb16 remoteclip_vitb32

Each class is described by an ensemble of prompts ("a satellite photo of {forest}", ...);
an image is assigned to the class whose averaged text embedding is most similar.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from . import data
from .common import completed_runs, device, log_run, run_key
from .encoders import ENCODERS, load_encoder
from .extract_features import load_features

TEMPLATES = [
    "a satellite photo of {}.",
    "a satellite image of {}.",
    "an aerial photo of {}.",
    "a centered satellite photo of {}.",
    "a remote sensing image of {}.",
    "a Sentinel-2 satellite image showing {}.",
]

TEXT_ENCODERS = [k for k, v in ENCODERS.items() if v["lib"] == "open_clip"]


def class_text_embeddings(enc, class_names: list[str]) -> torch.Tensor:
    rows = []
    for name in class_names:
        desc = data.CLASS_DESCRIPTIONS[name]
        e = enc.encode_texts([t.format(desc) for t in TEMPLATES])
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.mean(0)
        rows.append(e / e.norm())
    return torch.stack(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", default=TEXT_ENCODERS, choices=TEXT_ENCODERS)
    ap.add_argument("--data", default=str(data.DEFAULT_DATA))
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    ds = data.load(args.data)
    done = completed_runs()
    for key in args.encoders:
        if run_key("zero_shot", key, "0", 0) in done:
            continue
        feats, meta = load_features(key)
        enc = load_encoder(key, pretrained=not args.no_pretrained, device=device())
        text = class_text_embeddings(enc, ds.class_names).cpu().numpy()
        img = feats[ds.test_idx]
        img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-8)
        pred = (img @ text.T).argmax(1)
        log_run(
            method="zero_shot",
            model=key,
            k="0",
            seed=0,
            y_true=ds.labels[ds.test_idx],
            y_pred=pred,
            total_params=meta["total_params"],
            trainable_params=0,
            ms_per_image=meta["ms_per_image"],
            extra={"templates": len(TEMPLATES), "pretrained": meta["pretrained"]},
        )


if __name__ == "__main__":
    main()
