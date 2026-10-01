"""Gradio demo: upload a satellite tile, get land-cover probabilities.

Uses a frozen encoder plus the linear head exported by
``python -m eurofm.linear_probe --encoders dinov3_vitl16_sat --budgets all --seeds 0 --export-head``.

    python app/app.py                         # launch the demo
    python app/app.py --export-examples       # write one test image per class to app/examples/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eurofm import data  # noqa: E402
from eurofm.common import RESULTS_DIR, device  # noqa: E402
from eurofm.encoders import ENCODERS, load_encoder  # noqa: E402

EXAMPLES = ROOT / "app" / "examples"


class Classifier:
    def __init__(self, encoder: str, pretrained: bool = True):
        head_path = RESULTS_DIR / "heads" / f"{encoder}.npz"
        if not head_path.exists():
            raise FileNotFoundError(f"{head_path} not found - export it with eurofm.linear_probe --export-head")
        head = np.load(head_path)
        self.coef, self.intercept = head["coef"], head["intercept"]
        self.class_names = [str(c) for c in head["class_names"]]
        self.enc = load_encoder(encoder, pretrained=pretrained, device=device())

    def __call__(self, image: Image.Image) -> dict[str, float]:
        # EuroSAT tiles are 64x64 px at 10 m/px; resize uploads the same way the training data was seen
        tile = np.asarray(image.convert("RGB").resize((64, 64), Image.BICUBIC))[None]
        feats = self.enc.encode_images(tile, batch_size=1)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        logits = feats @ self.coef.T + self.intercept
        probs = torch.softmax(torch.from_numpy(logits[0]), 0).numpy()
        return {f"{n} ({data.CLASS_DESCRIPTIONS[n]})": float(p) for n, p in zip(self.class_names, probs)}


def export_examples() -> None:
    ds = data.load()
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    for c, name in enumerate(ds.class_names):
        idx = ds.test_idx[ds.labels[ds.test_idx] == c][0]
        Image.fromarray(ds.images[idx]).save(EXAMPLES / f"{name}.png")
    print(f"wrote {len(ds.class_names)} examples to {EXAMPLES}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov3_vitl16_sat", choices=list(ENCODERS))
    ap.add_argument("--export-examples", action="store_true")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    if args.export_examples:
        export_examples()
        return

    import gradio as gr

    clf = Classifier(args.encoder)
    examples = sorted(str(p) for p in EXAMPLES.glob("*.png"))
    demo = gr.Interface(
        fn=clf,
        inputs=gr.Image(type="pil", label="Sentinel-2 RGB tile"),
        outputs=gr.Label(num_top_classes=5, label="Land cover"),
        title="EuroSAT land-cover classifier",
        description=(
            f"{ENCODERS[args.encoder]['desc']} (frozen) + a linear head trained on EuroSAT. "
            "Works best on ~640 m x 640 m Sentinel-2 RGB tiles."
        ),
        examples=examples or None,
        flagging_mode="never",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
