"""Shared helpers: devices, seeding, metrics and the results log."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

RESULTS_DIR = Path(os.environ.get("EUROFM_RESULTS", "results"))
RUNS_CSV = RESULTS_DIR / "runs.csv"
PREDS_DIR = RESULTS_DIR / "preds"

FIELDS = [
    "method",  # scratch | linear_probe | zero_shot | lora | full_ft | vlm_zero_shot | vlm_few_shot | vlm_lora
    "model",
    "k",  # labelled images per class ("0" for zero-shot, "all" for the whole pool)
    "seed",
    "accuracy",
    "macro_f1",
    "n_test",
    "total_params",
    "trainable_params",
    "train_seconds",
    "ms_per_image",  # inference latency on this machine, batch inference
    "extra",  # JSON dict with method-specific details
    "device",
    "timestamp",
]


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_key(method: str, model: str, k: str, seed: int) -> str:
    return f"{method}__{model}__k{k}__s{seed}"


def completed_runs() -> set[str]:
    """Keys of runs already in results/runs.csv, so interrupted sessions can resume."""
    if not RUNS_CSV.exists():
        return set()
    with RUNS_CSV.open() as f:
        return {run_key(r["method"], r["model"], r["k"], int(r["seed"])) for r in csv.DictReader(f)}


def log_run(
    *,
    method: str,
    model: str,
    k: str,
    seed: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_params: int = 0,
    trainable_params: int = 0,
    train_seconds: float = 0.0,
    ms_per_image: float = 0.0,
    extra: dict | None = None,
) -> dict:
    """Append one row to results/runs.csv and save the raw predictions for confusion matrices.

    Predictions of -1 (e.g. an unparseable VLM answer) count as wrong.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_true)
    row = {
        "method": method,
        "model": model,
        "k": k,
        "seed": seed,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 5),
        "n_test": len(y_true),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "train_seconds": round(float(train_seconds), 2),
        "ms_per_image": round(float(ms_per_image), 3),
        "extra": json.dumps(extra or {}, sort_keys=True),
        "device": device_name(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # predictions first: a run only counts as completed once its CSV row exists
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PREDS_DIR / f"{run_key(method, model, k, seed)}.npz", y_true=y_true, y_pred=y_pred)
    new = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[{method} | {model} | k={k} | seed={seed}] acc={row['accuracy']:.4f} macro-F1={row['macro_f1']:.4f}")
    return row


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.t0
