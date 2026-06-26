"""One-command Stage 2 runner: fine-tune Clinical-Longformer + score flagged patients.

Runs the full Stage 2 pipeline in sequence:
  1. Fine-tune Clinical-Longformer on discharge notes (train split)
  2. Score Stage 1 flagged patients on the test split
  3. Print a summary of how many flags were confirmed / pruned

Requires:
  - Stage 1 artifact:  models/stage1_xgboost.joblib
                       (run setup_demo.py then python -m src.model.train first)
  - MIMIC-IV-Note:     MIMIC_IV_NOTE_DIR set in .env
                       (Stage 2 cannot run on synthetic data — no synthetic notes exist)
  - HuggingFace model: yikuan8/Clinical-Longformer
                       (downloads automatically on first run, ~500 MB)

Usage:
    python setup_stage2.py
    python setup_stage2.py --mode full
"""

import argparse
import sys
from pathlib import Path

from src.config import load_config, has_real_data


def check_prerequisites(cfg: dict) -> None:
    from src.config import get_model_dir
    model_dir = get_model_dir()
    stage1_name = cfg["stage1"]["model"]
    artifact = model_dir / f"stage1_{stage1_name}.joblib"
    if not artifact.exists():
        print(f"[setup_stage2] ERROR: Stage 1 artifact not found at {artifact}")
        print("  Run setup_demo.py then python -m src.model.train first.")
        sys.exit(1)

    note_dir = cfg["data"].get("mimic_iv_note_dir", "")
    if not note_dir or not Path(note_dir).exists():
        print("[setup_stage2] ERROR: MIMIC-IV-Note path not configured or not found.")
        print("  Set MIMIC_IV_NOTE_DIR in your .env file.")
        print("  (Stage 2 requires real discharge notes — synthetic data has none.)")
        sys.exit(1)

    print(f"[setup_stage2] Stage 1 artifact: {artifact}")
    print(f"[setup_stage2] MIMIC-IV-Note:    {note_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Stage 2 end-to-end")
    parser.add_argument("--mode", choices=["quick", "full"], default=None,
                        help="Override run.mode from config.yaml")
    args = parser.parse_args()

    cfg = load_config()
    if args.mode:
        cfg["run"]["mode"] = args.mode

    print("=" * 64)
    print("  Readmission Prediction — Stage 2 (Clinical-Longformer)")
    print("=" * 64)
    print(f"  Mode:  {cfg['run']['mode']}")
    print(f"  Model: {cfg['stage2']['model_name']}")
    print(f"  Max sequence length: {cfg['stage2']['max_seq_length']} tokens")
    print()

    check_prerequisites(cfg)

    # ── Step 1: Fine-tune ──
    print("\n[1/2] Fine-tuning Clinical-Longformer on discharge notes...")
    print("      (model downloads automatically from HuggingFace on first run)\n")
    from src.stage2.train import train_stage2
    train_stage2(cfg)

    # ── Step 2: Predict ──
    print("\n[2/2] Scoring Stage 1 flagged patients with fine-tuned model...\n")
    from src.stage2.predict import predict_stage2
    results = predict_stage2(cfg)

    # ── Summary ──
    total = len(results)
    confirmed = int(results["stage2_confirmed"].sum())
    pruned = total - confirmed
    true_pos = int(
        (results["stage2_confirmed"] == 1) & (results["readmission_30d"] == 1)
    ).sum() if "readmission_30d" in results.columns else "n/a"

    print("\n" + "=" * 64)
    print("  Stage 2 complete")
    print("=" * 64)
    print(f"  Stage 1 flagged:      {total:,}")
    print(f"  Stage 2 confirmed:    {confirmed:,}  ({confirmed / max(total, 1):.1%} retained)")
    print(f"  Stage 2 pruned:       {pruned:,}  ({pruned / max(total, 1):.1%} removed)")
    print(f"  Results saved to:     models/stage2_results.csv")
    print()
    print("  Next: python setup_stage3.py  (requires Ollama + phi4-mini)")
    print("=" * 64)


if __name__ == "__main__":
    main()
