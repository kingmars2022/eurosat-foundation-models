"""Linear probe: logistic regression on frozen, cached encoder features.

    python -m eurofm.linear_probe --encoders dinov3_vitl16_sat resnet50_in1k --budgets 1,5,10,50,100,500 --seeds 0 1 2

With ``--export-head`` it also fits one probe on the whole labelled pool and saves it to
``results/heads/<encoder>.npz`` for the Gradio demo.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import data
from .common import RESULTS_DIR, completed_runs, log_run, run_key
from .encoders import ENCODERS
from .extract_features import load_features

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def fit_probe(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[LogisticRegression, float]:
    """Pick C by 5-fold CV when there are at least 5 examples per class; otherwise use C=10."""
    base = LogisticRegression(max_iter=3000, C=10.0)
    if np.bincount(y).min() >= 5:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        search = GridSearchCV(base, {"C": C_GRID}, cv=cv, n_jobs=-1)
        search.fit(x, y)
        return search.best_estimator_, float(search.best_params_["C"])
    base.fit(x, y)
    return base, 10.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    ap.add_argument("--budgets", default="1,5,10,50,100,500", help="labelled images per class; 'all' = whole pool")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data", default=str(data.DEFAULT_DATA))
    ap.add_argument("--export-head", action="store_true", help="also fit on the full pool and save the head")
    args = ap.parse_args()

    ds = data.load(args.data)
    done = completed_runs()
    y_test = ds.labels[ds.test_idx]
    for key in args.encoders:
        feats, meta = load_features(key)
        feats = l2_normalize(feats)
        for k in data.parse_budgets(args.budgets):
            for seed in args.seeds:
                kname = data.budget_name(k)
                if run_key("linear_probe", key, kname, seed) in done:
                    continue
                train_idx = data.sample_kshot(ds, k, seed)
                t0 = time.perf_counter()
                clf, c = fit_probe(feats[train_idx], ds.labels[train_idx], seed)
                train_s = time.perf_counter() - t0
                pred = clf.predict(feats[ds.test_idx])
                n_head = clf.coef_.size + clf.intercept_.size
                log_run(
                    method="linear_probe",
                    model=key,
                    k=kname,
                    seed=seed,
                    y_true=y_test,
                    y_pred=pred,
                    total_params=meta["total_params"] + n_head,
                    trainable_params=n_head,
                    train_seconds=train_s,
                    ms_per_image=meta["ms_per_image"],
                    extra={"C": c, "n_train": int(len(train_idx)), "pretrained": meta["pretrained"]},
                )

        if args.export_head:
            clf, c = fit_probe(feats[ds.pool_idx], ds.labels[ds.pool_idx], 0)
            out = RESULTS_DIR / "heads" / f"{key}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, coef=clf.coef_, intercept=clf.intercept_, class_names=np.asarray(ds.class_names), C=c)
            acc = (clf.predict(feats[ds.test_idx]) == y_test).mean()
            print(f"exported {out} (full pool, C={c}, test acc={acc:.4f})")


if __name__ == "__main__":
    main()
