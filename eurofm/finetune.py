"""Supervised training on k labelled images per class.

Modes
-----
scratch : ResNet-18 trained from random init at native 64x64 (the "no pre-training" reference).
lora    : frozen DINOv3 backbone + LoRA adapters on the attention layers + linear head.
full    : full fine-tuning of a DINOv3 backbone (expensive; optional).

    python -m eurofm.finetune --mode scratch --budgets 1,5,10,50,100,500
    python -m eurofm.finetune --mode lora --encoders dinov3_vitl16_sat --budgets 5,50,500

Satellite tiles have no canonical "up", so augmentation is random flips + 90-degree rotations.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn

from . import data
from .common import Timer, completed_runs, count_params, device, log_run, run_key, seed_everything
from .encoders import ENCODERS, load_encoder

LORA_ENCODERS = [k for k, v in ENCODERS.items() if v["lib"] == "timm" and k.startswith("dinov3")]

DEFAULTS = {
    #          lr      wd     bs  epochs min_steps max_steps
    "scratch": (2e-3, 5e-4, 64, 100, 500, 3000),
    "lora": (1e-3, 1e-4, 32, 10, 100, 600),
    "full": (2e-5, 0.05, 32, 10, 100, 600),
}


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random horizontal/vertical flips and 90-degree rotations of a uint8 NHWC batch."""
    n = x.shape[0]
    flip_h = torch.rand(n, device=x.device) < 0.5
    flip_v = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip_h.view(-1, 1, 1, 1), x.flip(2), x)
    x = torch.where(flip_v.view(-1, 1, 1, 1), x.flip(1), x)
    rot = torch.randint(0, 4, (n,), device=x.device)
    out = x.clone()
    for r in range(1, 4):
        m = rot == r
        if m.any():
            out[m] = torch.rot90(x[m], r, dims=(1, 2))
    return out


class ScratchNet(nn.Module):
    """timm ResNet-18 at 64x64, normalised with EuroSAT pool statistics."""

    def __init__(self, num_classes: int, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        import timm

        self.net = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.permute(0, 3, 1, 2).float().div(255.0)
        return self.net((x - self.mean) / self.std)


class EncoderClassifier(nn.Module):
    """Pretrained encoder (optionally with LoRA adapters) + linear head."""

    def __init__(self, enc, num_classes: int):
        super().__init__()
        self.enc = enc
        self.backbone = enc.model  # registered so .parameters() / .to() see it
        self.head = nn.Linear(enc.feature_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc.forward_features(self.enc.preprocess(images)).float())


def build_model(mode: str, key: str | None, num_classes: int, ds: data.EuroSAT, pretrained: bool, lora_rank: int):
    if mode == "scratch":
        pool = ds.images[ds.pool_idx].reshape(-1, 3).astype(np.float64) / 255.0
        return ScratchNet(num_classes, pool.mean(0), pool.std(0))

    enc = load_encoder(key, pretrained=pretrained)
    if hasattr(enc.model, "set_grad_checkpointing"):
        enc.model.set_grad_checkpointing(True)
    if mode == "lora":
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.1, target_modules=r".*\.attn\.(qkv|proj)")
        enc.model = get_peft_model(enc.model, cfg)
    else:  # full fine-tuning
        for p in enc.model.parameters():
            p.requires_grad_(True)
    return EncoderClassifier(enc, num_classes)


@torch.no_grad()
def predict(model: nn.Module, images: np.ndarray, dev: torch.device, batch_size: int = 128) -> tuple[np.ndarray, float]:
    model.eval()
    preds = []
    with Timer() as t:
        for i in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[i : i + batch_size]).to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                preds.append(model(batch).argmax(1).cpu())
    return torch.cat(preds).numpy(), 1000 * t.seconds / len(images)


def train_one(model: nn.Module, x: np.ndarray, y: np.ndarray, dev: torch.device, hp: dict, seed: int) -> float:
    seed_everything(seed)
    model.to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["wd"])
    bs = min(hp["bs"], len(x))
    steps = int(np.clip(hp["epochs"] * math.ceil(len(x) / bs), hp["min_steps"], hp["max_steps"]))
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    x_t = torch.from_numpy(x).to(dev)
    y_t = torch.from_numpy(y).to(dev)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    t0 = time.perf_counter()
    step = 0
    while step < steps:
        order = torch.randperm(len(x), generator=gen).to(dev)
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(augment(x_t[idx])), y_t[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0 or step == steps:
                print(f"    step {step}/{steps} loss {loss.item():.4f}")
            if step >= steps:
                break
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=list(DEFAULTS), required=True)
    ap.add_argument("--encoders", nargs="+", default=["dinov3_vitl16_sat"], choices=LORA_ENCODERS)
    ap.add_argument("--budgets", default=None, help="default: 1,5,10,50,100,500 (scratch) / 5,50,500 (lora, full)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data", default=str(data.DEFAULT_DATA))
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--no-pretrained", action="store_true", help="random weights (offline smoke test only)")
    args = ap.parse_args()

    lr, wd, bs, epochs, min_steps, max_steps = DEFAULTS[args.mode]
    hp = dict(
        lr=args.lr or lr,
        wd=wd,
        bs=args.batch_size or bs,
        epochs=epochs,
        min_steps=min(min_steps, args.max_steps or min_steps),
        max_steps=args.max_steps or max_steps,
    )
    budgets = data.parse_budgets(args.budgets or ("1,5,10,50,100,500" if args.mode == "scratch" else "5,50,500"))
    ds = data.load(args.data)
    dev = device()
    done = completed_runs()
    keys = ["resnet18_scratch"] if args.mode == "scratch" else args.encoders

    for key in keys:
        for k in budgets:
            for seed in args.seeds:
                kname = data.budget_name(k)
                if run_key(args.mode, key, kname, seed) in done:
                    continue
                train_idx = data.sample_kshot(ds, k, seed)
                print(f"== {args.mode} | {key} | k={kname} | seed={seed} | n_train={len(train_idx)}")
                seed_everything(seed)
                model = build_model(args.mode, key, ds.num_classes, ds, not args.no_pretrained, args.lora_rank)
                total, trainable = count_params(model)
                train_s = train_one(model, ds.images[train_idx], ds.labels[train_idx], dev, hp, seed)
                pred, ms = predict(model, ds.images[ds.test_idx], dev)
                log_run(
                    method=args.mode,
                    model=key,
                    k=kname,
                    seed=seed,
                    y_true=ds.labels[ds.test_idx],
                    y_pred=pred,
                    total_params=total,
                    trainable_params=trainable,
                    train_seconds=train_s,
                    ms_per_image=ms,
                    extra={**hp, "n_train": int(len(train_idx)), "lora_rank": args.lora_rank if args.mode == "lora" else None},
                )
                del model
                if dev.type == "cuda":
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
