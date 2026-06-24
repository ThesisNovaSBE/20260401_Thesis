"""Train the Stage 1 model selected in config.yaml and save the artifact.

- Patient-level held-out test split (touched only by evaluate.py).
- Uses tuned hyperparameters from models/<model>_best_params.json if present,
  otherwise sensible defaults.
- Selects the decision threshold for the high-recall mandate using out-of-fold
  predictions on the TRAINING set (no leakage from the test set).
- Saves model + threshold + split + metadata to models/ (gitignored).

Usage:
    python -m src.model.train                 # uses config.yaml run.mode
    python -m src.model.train --mode full
    python -m src.model.train --model logistic_regression
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.config import load_config, get_model_dir
from src.data.features import load_feature_matrix, split_xy
from src.model.models import build_estimator, fit_estimator, default_params
from src.model.cv import oof_predict
from src.model.splits import grouped_train_test_split
from src.model.metrics import select_threshold_for_recall, auprc, auroc


def _load_params(name: str, model_dir) -> dict:
    path = model_dir / f"{name}_best_params.json"
    if path.exists():
        print(f"[train] Using tuned params from {path.name}")
        return json.loads(path.read_text())
    print("[train] No tuned params found — using defaults.")
    return default_params(name)


def train(cfg: dict) -> None:
    mode = cfg["run"]["mode"]
    seed = cfg["run"]["random_state"]
    name = cfg["stage1"]["model"]
    n_splits = cfg["run"][mode]["cv_folds"]
    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_feature_matrix(cfg, mode)
    X, y, groups, _subgroups, feat_cols = split_xy(matrix)
    print(f"[train] mode={mode} model={name} rows={len(y):,} features={len(feat_cols)} "
          f"pos_rate={y.mean():.1%}")

    train_idx, test_idx = grouped_train_test_split(
        y, groups, test_size=cfg["stage1"]["test_size"], seed=seed)
    Xtr, ytr, gtr = X.iloc[train_idx].reset_index(drop=True), y[train_idx], groups[train_idx]

    params = _load_params(name, model_dir)

    # ── Threshold selection via out-of-fold predictions on TRAIN ──
    print("[train] Computing out-of-fold predictions for threshold selection...")
    oof = oof_predict(name, Xtr, ytr, gtr, cfg, seed, n_splits, params)
    target_recall = cfg["stage1"]["target_recall"]
    threshold = select_threshold_for_recall(ytr, oof, target_recall)
    print(f"[train] OOF AUPRC={auprc(ytr, oof):.4f} AUROC={auroc(ytr, oof):.4f} "
          f"| threshold@recall>={target_recall}: {threshold:.4f}")

    # ── Fit final model on the full training set ──
    est = build_estimator(name, params, ytr, cfg, seed)
    if name == "xgboost":
        itr, iva = next(GroupShuffleSplit(
            n_splits=1, test_size=0.15, random_state=seed).split(Xtr, ytr, gtr))
        fit_estimator(est, name, Xtr.iloc[itr], ytr[itr], Xtr.iloc[iva], ytr[iva])
    else:
        fit_estimator(est, name, Xtr, ytr)

    artifact = {
        "estimator": est, "model_name": name, "threshold": float(threshold),
        "feature_cols": feat_cols, "params": params, "seed": seed, "mode": mode,
        "train_idx": train_idx, "test_idx": test_idx,
        "target_recall": target_recall,
    }
    out_path = model_dir / f"stage1_{name}.joblib"
    joblib.dump(artifact, out_path)
    print(f"[train] Saved model -> {out_path}")
    print("[train] Done. Run `python -m src.model.evaluate` to score the held-out test set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 1 model")
    parser.add_argument("--mode", choices=["quick", "full"], default=None)
    parser.add_argument("--model", choices=["logistic_regression", "xgboost",
                                            "histgradientboosting"], default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.mode:
        cfg["run"]["mode"] = args.mode
    if args.model:
        cfg["stage1"]["model"] = args.model
    train(cfg)
