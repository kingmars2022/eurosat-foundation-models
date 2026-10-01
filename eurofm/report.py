"""Aggregate results/runs.csv into tables and figures.

    python -m eurofm.report

Writes
    results/summary.md                       accuracy table (mean ± std over seeds) + cost table
    results/figures/learning_curves{,_dark}.png  headline: accuracy vs labels per class
    results/figures/linear_probes{,_dark}.png    every frozen encoder under the same linear probe
    results/figures/confusion_<run>{,_dark}.png  confusion matrix of the strongest 50-shot model
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from .common import PREDS_DIR, RESULTS_DIR, RUNS_CSV, run_key  # noqa: E402
from .data import CLASS_NAMES  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"

MODEL_NAMES = {
    "resnet18_scratch": "ResNet-18 (random init)",
    "resnet50_in1k": "ResNet-50 ImageNet",
    "dinov3_vitb16_web": "DINOv3 ViT-B web",
    "dinov3_vitl16_web": "DINOv3 ViT-L web",
    "dinov3_vitl16_sat": "DINOv3 ViT-L satellite",
    "clip_vitb16": "CLIP ViT-B/16",
    "siglip2_vitb16": "SigLIP 2 ViT-B/16",
    "remoteclip_vitb32": "RemoteCLIP ViT-B/32",
    "qwen3.5-0.8b": "Qwen3.5-0.8B",
    "qwen3.5-2b": "Qwen3.5-2B",
    "qwen3.5-4b": "Qwen3.5-4B",
    "qwen3.5-9b": "Qwen3.5-9B",
    "gemma3-4b": "Gemma 3 4B",
}
METHOD_NAMES = {
    "scratch": "trained from scratch",
    "linear_probe": "linear probe",
    "zero_shot": "zero-shot (prompts)",
    "lora": "LoRA fine-tune",
    "full_ft": "full fine-tune",
    "full": "full fine-tune",
    "vlm_zero_shot": "VLM zero-shot",
    "vlm_few_shot": "VLM in-context few-shot",
    "vlm_lora": "VLM LoRA fine-tune",
}
METHOD_ORDER = ["scratch", "zero_shot", "linear_probe", "lora", "full", "vlm_zero_shot", "vlm_few_shot", "vlm_lora"]

# Validated categorical palette (fixed order; colour follows the entity, never its rank).
SLOTS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"],
}
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  seq=["#fcfcfb", "#86b6ef", "#2a78d6", "#0d366b"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 seq=["#1a1a19", "#184f95", "#3987e5", "#cde2fb"]),
}
# Headline figure: (method, model-prefix) -> palette slot
HEADLINE = [
    (("linear_probe", "dinov3_vitl16_sat"), 0),
    (("lora", "dinov3_vitl16_sat"), 1),
    (("linear_probe", "dinov3_vitl16_web"), 2),
    (("linear_probe", "resnet50_in1k"), 3),
    (("vlm_lora", None), 4),  # first VLM fine-tuned
    (("scratch", "resnet18_scratch"), 5),
]
PROBE_ORDER = ["dinov3_vitl16_sat", "dinov3_vitl16_web", "dinov3_vitb16_web", "siglip2_vitb16", "remoteclip_vitb32",
               "clip_vitb16", "resnet50_in1k"]


def label(method: str, model: str) -> str:
    return f"{MODEL_NAMES.get(model, model)} · {METHOD_NAMES.get(method, method)}"


def load_runs() -> pd.DataFrame:
    df = pd.read_csv(RUNS_CSV, dtype={"k": str})
    df = df.drop_duplicates(subset=["method", "model", "k", "seed"], keep="last")
    df["k_num"] = df["k"].map(lambda k: np.inf if k == "all" else float(k))
    df["parse_rate"] = df["extra"].map(lambda e: json.loads(e).get("parse_rate", np.nan))
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["method", "model", "k", "k_num"])
    agg = g.agg(
        acc=("accuracy", "mean"), acc_std=("accuracy", "std"), f1=("macro_f1", "mean"), seeds=("seed", "nunique"),
        total_params=("total_params", "max"), trainable_params=("trainable_params", "max"),
        train_s=("train_seconds", "mean"), ms=("ms_per_image", "median"), parse_rate=("parse_rate", "mean"),
    ).reset_index()
    agg["acc_std"] = agg["acc_std"].fillna(0.0)
    agg["order"] = agg["method"].map(lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    return agg.sort_values(["order", "model", "k_num"])


# ----------------------------------------------------------------------------- tables
def fmt_params(n: float) -> str:
    if not n or np.isnan(n):
        return "0"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(int(n))


def write_summary(agg: pd.DataFrame, df: pd.DataFrame) -> None:
    ks = sorted(agg["k_num"].unique())
    kcols = ["all" if k == np.inf else str(int(k)) for k in ks]
    lines = ["# Results", "", f"Test set: {int(df['n_test'].max())} images (fixed, stratified). "
             "Accuracy in %, mean ± std over seeds. Columns = labelled images per class (0 = zero-shot).", ""]
    lines.append("| Model | Method | " + " | ".join(kcols) + " |")
    lines.append("|---|---|" + "---:|" * len(kcols))
    for (method, model), grp in agg.groupby(["method", "model"], sort=False):
        cells = []
        for k in ks:
            r = grp[grp["k_num"] == k]
            if r.empty:
                cells.append("")
            else:
                r = r.iloc[0]
                cells.append(f"{100 * r.acc:.1f} ± {100 * r.acc_std:.1f}" if r.seeds > 1 else f"{100 * r.acc:.1f}")
        lines.append(f"| {MODEL_NAMES.get(model, model)} | {METHOD_NAMES.get(method, method)} | " + " | ".join(cells) + " |")

    lines += ["", "## Cost", "", "Largest budget per row. Latency is batch inference on the device used (see runs.csv).", "",
              "| Model | Method | Total params | Trainable params | Train time (s) | Inference (ms/img) | VLM parse rate |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for (method, model), grp in agg.groupby(["method", "model"], sort=False):
        r = grp.sort_values("k_num").iloc[-1]
        pr = "" if np.isnan(r.parse_rate) else f"{100 * r.parse_rate:.1f}%"
        lines.append(f"| {MODEL_NAMES.get(model, model)} | {METHOD_NAMES.get(method, method)} | {fmt_params(r.total_params)} | "
                     f"{fmt_params(r.trainable_params)} | {r.train_s:.0f} | {r.ms:.2f} | {pr} |")
    lines += ["", f"Device(s): {', '.join(sorted(df['device'].dropna().unique()))}", ""]
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines))
    print(f"wrote {RESULTS_DIR / 'summary.md'}")


# ----------------------------------------------------------------------------- figures
def style_axes(ax, t: dict) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=9)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def curve_axes(title: str, subtitle: str, t: dict, ks: list[float], legend_below: bool = False):
    fig, ax = plt.subplots(figsize=(8.5, 6.2 if legend_below else 5.2), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    style_axes(ax, t)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(int(k)) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("Labelled training images per class", color=t["ink2"], fontsize=10)
    ax.set_ylabel("Test accuracy (%)", color=t["ink2"], fontsize=10)
    top = 0.955 if legend_below else 0.95
    fig.text(0.06, top, title, color=t["ink"], fontsize=13, fontweight="bold", ha="left")
    fig.text(0.06, top - (0.038 if legend_below else 0.045), subtitle, color=t["ink2"], fontsize=9.5, ha="left")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.87 if legend_below else 0.85, bottom=0.25 if legend_below else 0.12)
    return fig, ax


def plot_series(ax, grp: pd.DataFrame, color: str, name: str, t: dict) -> None:
    grp = grp[np.isfinite(grp["k_num"]) & (grp["k_num"] > 0)].sort_values("k_num")
    if grp.empty:
        return
    x, y, s = grp["k_num"].values, 100 * grp["acc"].values, 100 * grp["acc_std"].values
    ax.fill_between(x, y - s, y + s, color=color, alpha=0.14, linewidth=0)
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6,
            markeredgecolor=t["surface"], markeredgewidth=1.5, label=name)


def reference_line(ax, value: float, text: str, t: dict, above: bool = True) -> None:
    ax.axhline(value, color=t["muted"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(text, xy=(1.0, value), xycoords=("axes fraction", "data"), xytext=(-4, 4 if above else -4),
                textcoords="offset points", ha="right", va="bottom" if above else "top", fontsize=8.5, color=t["ink2"],
                bbox=dict(boxstyle="square,pad=0.15", facecolor=t["surface"], edgecolor="none"))


def legend(ax, t: dict, below: bool = False) -> None:
    if below:  # under the plot, so it can never cover a curve or a reference-line label
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5, frameon=False,
                  labelcolor=t["ink2"])
        return
    leg = ax.legend(loc="lower right", fontsize=8.5, frameon=True, labelcolor=t["ink2"])
    leg.get_frame().set_facecolor(t["surface"])
    leg.get_frame().set_edgecolor(t["grid"])


def fig_learning_curves(agg: pd.DataFrame, mode: str) -> None:
    t, slots = THEME[mode], SLOTS[mode]
    ks = sorted(k for k in agg["k_num"].unique() if np.isfinite(k) and k > 0)
    if not ks:
        return
    fig, ax = curve_axes("How many labels does each approach need?",
                         "EuroSAT land-cover classification · shaded band = ±1 std over seeds · dashed = zero labels", t, ks,
                         legend_below=True)
    for (method, model), slot in HEADLINE:
        sel = agg[agg["method"] == method]
        if model is not None:
            sel = sel[sel["model"] == model]
        elif not sel.empty:
            sel = sel[sel["model"] == sel["model"].iloc[0]]
        if not sel.empty:
            plot_series(ax, sel, slots[slot], label(method, sel["model"].iloc[0]), t)
    refs = []
    for method in ("zero_shot", "vlm_zero_shot"):
        zs = agg[(agg["method"] == method) & (agg["k"] == "0")]
        if not zs.empty:
            best = zs.loc[zs["acc"].idxmax()]
            refs.append((100 * best.acc, f"best {METHOD_NAMES[method]}: {MODEL_NAMES.get(best.model, best.model)} "
                                         f"({100 * best.acc:.1f}%)"))
    # higher line labelled above, lower one below, so close values never collide
    for i, (value, text) in enumerate(sorted(refs, reverse=True)):
        reference_line(ax, value, text, t, above=i == 0)
    legend(ax, t, below=True)
    save(fig, "learning_curves", mode)


def fig_linear_probes(agg: pd.DataFrame, mode: str) -> None:
    t, slots = THEME[mode], SLOTS[mode]
    lp = agg[agg["method"] == "linear_probe"]
    ks = sorted(k for k in lp["k_num"].unique() if np.isfinite(k) and k > 0)
    if not ks:
        return
    fig, ax = curve_axes("Which frozen encoder gives the best features?",
                         "Same logistic-regression probe on every encoder · shaded band = ±1 std over seeds", t, ks)
    for slot, model in enumerate(PROBE_ORDER):
        sel = lp[lp["model"] == model]
        if not sel.empty:
            plot_series(ax, sel, slots[slot], MODEL_NAMES.get(model, model), t)
    legend(ax, t)
    save(fig, "linear_probes", mode)


def fig_confusion(agg: pd.DataFrame, mode: str, k: str = "50") -> None:
    t = THEME[mode]
    cand = agg[agg["k"] == k]
    if cand.empty:
        return
    best = cand.loc[cand["acc"].idxmax()]
    path = PREDS_DIR / f"{run_key(best.method, best.model, k, 0)}.npz"
    if not path.exists():
        return
    z = np.load(path)
    valid = z["y_pred"] >= 0
    cm = confusion_matrix(z["y_true"][valid], z["y_pred"][valid], labels=range(len(CLASS_NAMES)), normalize="true")
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    # sequential blue: near-zero recedes into the surface, the diagonal stands out
    cmap = LinearSegmentedColormap.from_list(f"seq_{mode}", t["seq"])
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=40, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            v = cm[i, j]
            if round(100 * v) >= 1:
                strong = v > 0.5
                color = ("#ffffff" if strong else t["ink"]) if mode == "light" else ("#0b0b0b" if strong else t["ink"])
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=7.5, color=color)
    ax.set_xlabel("Predicted", color=t["ink2"], fontsize=10)
    ax.set_ylabel("True", color=t["ink2"], fontsize=10)
    fig.suptitle(f"Where does {label(best.method, best.model)} go wrong?", color=t["ink"], fontsize=12,
                 fontweight="bold", x=0.02, ha="left")
    ax.set_title(f"{k} labels per class, seed 0 · row-normalised (%)", color=t["ink2"], fontsize=9, loc="left")
    fig.tight_layout()
    save(fig, f"confusion_{best.method}_{best.model}_k{k}", mode)


def save(fig, name: str, mode: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}{'' if mode == 'light' else '_dark'}.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    df = load_runs()
    agg = aggregate(df)
    write_summary(agg, df)
    for mode in ("light", "dark"):
        fig_learning_curves(agg, mode)
        fig_linear_probes(agg, mode)
        fig_confusion(agg, mode)


if __name__ == "__main__":
    main()
