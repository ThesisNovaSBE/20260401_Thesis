"""Evaluate a trained Stage 1 model on the held-out test set.

Reports (saved to models/<model>_metrics.json and printed):
- AUPRC (primary), AUROC (secondary), Brier calibration score
- Operating point at the configured recall target (recall/precision/specificity/F2 + confusion matrix)
- Precision/specificity trade-off at recall 0.80 / 0.85 / 0.90
- Per-subgroup AUROC (sex, age band) as a fairness check

Usage:
    python -m src.model.evaluate
    python -m src.model.evaluate --model logistic_regression
"""

from __future__ import annotations

import argparse
import json

import joblib

from src.config import load_config, get_model_dir
from src.data.features import load_feature_matrix, split_xy
from src.model.metrics import full_report


def _print_report(report: dict, name: str) -> None:
    print("\n" + "=" * 64)
    print(f"  Stage 1 evaluation — {name}  (held-out test set)")
    print("=" * 64)
    print(f"  AUPRC (primary): {report['auprc']:.4f}")
    print(f"  AUROC          : {report['auroc']:.4f}")
    print(f"  Brier (calib.) : {report['brier']:.4f}")
    print(f"  Base rate      : {report['base_rate']:.1%}")

    op = report["operating_point"]
    print(f"\n  Operating point (threshold={op['threshold']:.4f}):")
    print(f"    recall={op['recall']:.3f}  precision={op['precision']:.3f}  "
          f"specificity={op['specificity']:.3f}  F2={op['f2']:.3f}")
    print(f"    confusion: TP={op['tp']} FP={op['fp']} TN={op['tn']} FN={op['fn']}")

    print("\n  Recall/precision trade-off:")
    for row in report["recall_tradeoff"]:
        print(f"    recall>={row['recall_target']:.2f} -> recall={row['recall']:.3f} "
              f"precision={row['precision']:.3f} specificity={row['specificity']:.3f} "
              f"(thr={row['threshold']:.4f})")

    print("\n  Subgroup AUROC (fairness check):")
    for col, levels in report["subgroup_auroc"].items():
        print(f"    [{col}]")
        for level, stats in levels.items():
            a = "n/a " if stats["auroc"] is None else f"{stats['auroc']:.3f}"
            print(f"      {level:<8} AUROC={a}  n={stats['n']}  pos_rate={stats['pos_rate']:.1%}")
    print("=" * 64 + "\n")


def evaluate(cfg: dict) -> dict:
    name = cfg["stage1"]["model"]
    model_dir = get_model_dir()
    artifact = joblib.load(model_dir / f"stage1_{name}.joblib")

    matrix = load_feature_matrix(cfg, artifact["mode"])
    X, y, _groups, subgroups, _feat = split_xy(matrix)

    test_idx = artifact["test_idx"]
    Xte = X.iloc[test_idx][artifact["feature_cols"]]
    yte = y[test_idx]
    sub_te = subgroups.iloc[test_idx].reset_index(drop=True)

    score = artifact["estimator"].predict_proba(Xte)[:, 1]
    report = full_report(yte, score, artifact["threshold"],
                         cfg["stage1"]["recall_report_points"], subgroups=sub_te)

    _print_report(report, name)

    out_path = model_dir / f"{name}_metrics.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[evaluate] Saved metrics -> {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 model")
    parser.add_argument("--model", choices=["logistic_regression", "xgboost",
                                            "histgradientboosting"], default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.model:
        cfg["stage1"]["model"] = args.model
    evaluate(cfg)
