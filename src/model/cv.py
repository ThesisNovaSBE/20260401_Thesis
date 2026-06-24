"""Cross-validation helpers shared by tune.py and train.py.

All splitting is grouped by patient (subject_id). For XGBoost we carve a small
inner grouped validation set out of each fold's training data so early stopping
never peeks at the fold's scoring data.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.model.models import build_estimator, fit_estimator, default_params
from src.model.splits import make_cv


def fit_predict_fold(name, X, y, groups, tr, va, params, cfg, seed) -> np.ndarray:
    """Fit on fold-train, return predicted probabilities for fold-validation."""
    Xtr, ytr = X.iloc[tr], y[tr]
    Xva = X.iloc[va]
    est = build_estimator(name, params, ytr, cfg, seed)

    if name == "xgboost":
        gtr = groups[tr]
        inner_tr, inner_va = next(GroupShuffleSplit(
            n_splits=1, test_size=0.15, random_state=seed).split(Xtr, ytr, gtr))
        fit_estimator(est, name, Xtr.iloc[inner_tr], ytr[inner_tr],
                      Xtr.iloc[inner_va], ytr[inner_va])
    else:
        fit_estimator(est, name, Xtr, ytr)

    return est.predict_proba(Xva)[:, 1]


def oof_predict(name, X, y, groups, cfg, seed, n_splits, params=None) -> np.ndarray:
    """Out-of-fold predicted probabilities over a grouped, stratified CV."""
    params = params or default_params(name)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    cv = make_cv(n_splits, seed)
    oof = np.full(len(y), np.nan)
    for tr, va in cv.split(X, y, groups):
        oof[va] = fit_predict_fold(name, X, y, groups, tr, va, params, cfg, seed)
    return oof
