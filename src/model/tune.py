"""Optuna hyperparameter search for the Stage 1 model (optimises CV AUPRC).

- Search spaces come from src/model/models.py (ranges per docs/MODELING_PLAN.md).
- Grouped, stratified CV on the TRAINING portion only (test set never touched).
- MedianPruner stops unpromising trials early.
- Writes best params to models/<model>_best_params.json for train.py to pick up.

Usage:
    python -m src.model.tune                  # uses config.yaml run.mode
    python -m src.model.tune --mode full
    python -m src.model.tune --model histgradientboosting
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna

from src.config import load_config, get_model_dir
from src.data.features import load_feature_matrix, split_xy
from src.model.cv import fit_predict_fold
from src.model.models import suggest_params
from src.model.splits import grouped_train_test_split, make_cv
from src.model.metrics import auprc


def run_study(cfg: dict) -> dict:
    mode = cfg["run"]["mode"]
    seed = cfg["run"]["random_state"]
    name = cfg["stage1"]["model"]
    n_splits = cfg["run"][mode]["cv_folds"]
    n_trials = cfg["run"][mode]["optuna_trials"]

    matrix = load_feature_matrix(cfg, mode)
    X, y, groups, _subgroups, feat_cols = split_xy(matrix)

    train_idx, _test_idx = grouped_train_test_split(
        y, groups, test_size=cfg["stage1"]["test_size"], seed=seed)
    Xtr = X.iloc[train_idx].reset_index(drop=True)
    ytr, gtr = y[train_idx], groups[train_idx]

    print(f"[tune] mode={mode} model={name} trials={n_trials} cv_folds={n_splits} "
          f"train_rows={len(ytr):,}")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(name, trial)
        cv = make_cv(n_splits, seed)
        fold_scores = []
        for i, (tr, va) in enumerate(cv.split(Xtr, ytr, gtr)):
            proba = fit_predict_fold(name, Xtr, ytr, gtr, tr, va, params, cfg, seed)
            fold_scores.append(auprc(ytr[va], proba))
            trial.report(float(np.mean(fold_scores)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"[tune] Best CV AUPRC: {study.best_value:.4f}")
    print(f"[tune] Best params: {study.best_params}")

    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / f"{name}_best_params.json"
    out_path.write_text(json.dumps(study.best_params, indent=2))
    print(f"[tune] Saved best params -> {out_path}")
    return study.best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna search for Stage 1 model")
    parser.add_argument("--mode", choices=["quick", "full"], default=None)
    parser.add_argument("--model", choices=["logistic_regression", "xgboost",
                                            "histgradientboosting"], default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.mode:
        cfg["run"]["mode"] = args.mode
    if args.model:
        cfg["stage1"]["model"] = args.model
    run_study(cfg)
