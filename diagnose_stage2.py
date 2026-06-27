"""
End-to-End Pipeline Diagnostic
================================
Runs the COMPLETE readmission prediction pipeline from scratch — Stage 1 through Stage 3.
Use this script to verify that your environment, data paths, and all dependencies are
correctly configured before submitting or sharing the project.

What it does
------------
  Stage 0  — Environment + path validation (packages, MIMIC paths, Ollama)
  Stage 1  — Delete stale cache → rebuild feature matrix → train XGBoost
  Stage 2  — Load discharge notes → fine-tune Clinical-Longformer (smoke test,
             capped at 4 gradient steps) → score Stage 1 flagged patients
  Stage 3  — Generate one sample explanation with phi4-mini via Ollama
             (skipped automatically if Ollama is not running)

Stage 2 smoke test vs full training
------------------------------------
  This script caps Stage 2 training at 4 gradient steps so it runs in ~5 minutes
  and confirms no crash. The model saved here is NOT usable for real predictions.
  For full Stage 2 fine-tuning (3 epochs, several hours on CPU), run:
      python setup_stage2.py

Output
------
  All output is also saved to diagnose_stage2.log (via tee in the run command below).

Run
---
    cd /Users/lenny/20260401_Thesis
    python diagnose_stage2.py 2>&1 | tee diagnose_stage2.log
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# ─── Helpers ──────────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
GREEN = "\033[32m"
RED   = "\033[31m"
CYAN  = "\033[36m"
RESET = "\033[0m"

def banner(title: str) -> None:
    print(f"\n{BOLD}{'═'*64}{RESET}", flush=True)
    print(f"{BOLD}{CYAN}  {title}{RESET}", flush=True)
    print(f"{BOLD}{'═'*64}{RESET}", flush=True)

def step(label: str) -> None:
    print(f"\n  {BOLD}▸ {label}{RESET}", flush=True)

def ok(detail: str = "") -> None:
    msg = f"  {GREEN}✓ PASS{RESET}" + (f" — {detail}" if detail else "")
    print(msg, flush=True)

def warn(detail: str) -> None:
    print(f"  ⚠  WARN — {detail}", flush=True)

def fail(detail) -> None:
    print(f"  {RED}✗ FAIL — {detail}{RESET}", flush=True)
    print(f"\n  Stopping. Fix the issue above and re-run.", flush=True)
    sys.exit(1)

def section_result(label: str, value) -> None:
    print(f"    {label:<30} {value}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — Environment + paths
# ══════════════════════════════════════════════════════════════════════════════

banner("STAGE 0 — Environment & Path Validation")

# ── Package versions ──────────────────────────────────────────────────────────
step("Python + package versions")
try:
    import importlib.metadata as meta
    section_result("Python", sys.version.split()[0])
    required = {
        "torch":          "2.0",
        "transformers":   "4.30",
        "accelerate":     "0.27",
        "joblib":         "1.2",
        "xgboost":        "1.7",
        "scikit-learn":   "1.2",
        "numpy":          "1.24",
        "pandas":         "1.5",
        "ollama":         "0.3",
    }
    missing = []
    for pkg, min_ver in required.items():
        try:
            v = meta.version(pkg)
            section_result(pkg, v)
        except meta.PackageNotFoundError:
            section_result(pkg, "NOT INSTALLED ← required")
            missing.append(pkg)
    if missing:
        fail(f"Missing packages: {missing}. Run: pip install {' '.join(missing)}")
    ok()
except Exception as e:
    fail(e)


# ── Config + data paths ───────────────────────────────────────────────────────
step("Config + data path check")
try:
    from src.config import load_config, has_real_data, get_model_dir, get_data_dir
    cfg = load_config()

    mimic_dir  = cfg["data"].get("mimic_iv_dir", "")
    note_dir   = cfg["data"].get("mimic_iv_note_dir", "")
    model_name_s2 = cfg["stage2"]["model_name"]
    mode       = cfg["run"]["mode"]
    model_dir  = get_model_dir()

    section_result("run.mode",            mode)
    section_result("stage2.model",        model_name_s2)
    section_result("mimic_iv_dir",        mimic_dir or "(not set — synthetic fallback)")
    section_result("mimic_iv_note_dir",   note_dir or "(not set — Stage 2 will fail)")
    section_result("mimic_iv_dir exists", Path(mimic_dir).exists() if mimic_dir else False)
    section_result("note_dir exists",     Path(note_dir).exists() if note_dir else False)

    if not has_real_data(cfg):
        warn("MIMIC_IV_DIR not set or missing — Stage 1 will use SYNTHETIC data. "
             "Stage 2 will fail (no real notes for synthetic hadm_ids).")
    if not note_dir or not Path(note_dir).exists():
        warn("MIMIC_IV_NOTE_DIR not set or missing — Stage 2 cannot run. "
             "Set it in .env to enable Stage 2.")
    ok()
except Exception as e:
    fail(e)


# ── Ollama check (non-fatal — Stage 3 will be skipped if unavailable) ─────────
step("Ollama + phi4-mini availability (optional, needed for Stage 3)")
ollama_ok = False
try:
    import ollama as _ollama
    models_response = _ollama.list()
    # Handle both dict-style and object-style responses across ollama SDK versions
    if hasattr(models_response, "models"):
        model_names = [m.model for m in models_response.models]
    else:
        model_names = [m.get("name", "") for m in models_response.get("models", [])]

    ollama_model = cfg["stage3"]["ollama_model"]
    if any(ollama_model in n for n in model_names):
        section_result("Ollama running",  True)
        section_result(f"{ollama_model} available", True)
        ollama_ok = True
        ok(f"{ollama_model} ready")
    else:
        section_result("Ollama running",  True)
        section_result(f"{ollama_model} available", False)
        warn(f"{ollama_model} not pulled. Run: ollama pull {ollama_model}. Stage 3 will be skipped.")
except Exception:
    warn("Ollama not running or not installed. Stage 3 will be skipped.")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Feature matrix + XGBoost training
# ══════════════════════════════════════════════════════════════════════════════

banner("STAGE 1 — Feature Engineering + XGBoost Training")

# ── Delete stale cache (forces rebuild from real data) ────────────────────────
step("Clearing stale feature cache (forces rebuild from configured data source)")
try:
    cache_path = get_data_dir().parent / cfg["output"]["processed_dir"] / "features.csv"
    if cache_path.exists():
        cache_path.unlink()
        section_result("Deleted cache", str(cache_path))
    else:
        section_result("Cache", "not present (clean start)")
    ok()
except Exception as e:
    fail(e)


# ── Build feature matrix ──────────────────────────────────────────────────────
step("Building feature matrix (reads MIMIC-IV or generates synthetic data)")
try:
    from src.data.features import load_feature_matrix, split_xy
    matrix = load_feature_matrix(cfg, mode)
    X, y, groups, subgroups, feat_cols = split_xy(matrix)

    section_result("Matrix shape",     matrix.shape)
    section_result("Feature columns",  len(feat_cols))
    section_result("Readmission rate", f"{y.mean():.1%}")
    section_result("Unique patients",  matrix["subject_id"].nunique())

    # Warn if hadm_ids look synthetic (round multiples of 100000)
    sample_ids = matrix["hadm_id"].head(5).tolist()
    if all(int(h) % 100000 == 0 for h in sample_ids):
        warn("hadm_ids look synthetic (e.g. 200000). Stage 2 will fail — "
             "set MIMIC_IV_DIR in .env.")
    ok()
except Exception as e:
    fail(e)


# ── Train Stage 1 model ───────────────────────────────────────────────────────
step("Training Stage 1 XGBoost model (cross-validation + threshold selection)")
try:
    from src.model.train import train as train_stage1
    train_stage1(cfg)
    ok()
except Exception as e:
    fail(e)


# ── Load + verify artifact ────────────────────────────────────────────────────
step("Verifying Stage 1 artifact")
try:
    import joblib
    stage1_name = cfg["stage1"]["model"]
    artifact_path = model_dir / f"stage1_{stage1_name}.joblib"
    artifact = joblib.load(artifact_path)

    section_result("Artifact path",   str(artifact_path))
    section_result("Train rows",      len(artifact["train_idx"]))
    section_result("Test rows",       len(artifact["test_idx"]))
    section_result("Threshold",       f"{artifact['threshold']:.4f}")
    section_result("Features",        len(artifact["feature_cols"]))
    ok()
except Exception as e:
    fail(e)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Clinical-Longformer fine-tuning + inference
# ══════════════════════════════════════════════════════════════════════════════

banner("STAGE 2 — Clinical-Longformer Fine-Tuning (smoke test, 4 steps)")
print(
    "  NOTE: Training is capped at 4 gradient steps to verify no crash.\n"
    "  The saved model is NOT suitable for real inference.\n"
    "  For full training (3 epochs, several hours on CPU), run:\n"
    "      python setup_stage2.py",
    flush=True
)

note_dir_ok = bool(note_dir) and Path(note_dir).exists()
if not note_dir_ok:
    warn("Skipping Stage 2 — MIMIC_IV_NOTE_DIR not set. Set it in .env and re-run.")
else:

    # ── Import torch and set env guards ──────────────────────────────────────
    step("Import torch + set device guards")
    try:
        import torch
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        cuda = torch.cuda.is_available()
        mps  = torch.backends.mps.is_available()
        section_result("torch version",    torch.__version__)
        section_result("cuda available",   cuda)
        section_result("mps available",    mps)
        section_result("device (forced)",  "cpu")
        ok()
    except Exception as e:
        fail(e)


    # ── Load discharge notes ──────────────────────────────────────────────────
    step("Loading discharge notes from MIMIC-IV-Note")
    try:
        from src.stage2.dataset import load_notes, build_notes_dataframe, ClinicalNotesDataset
        from src.schemas import TARGET_COL

        train_hadm_ids = set(matrix.iloc[artifact["train_idx"]]["hadm_id"].astype(int))
        test_hadm_ids  = set(matrix.iloc[artifact["test_idx"]]["hadm_id"].astype(int))
        all_hadm_ids   = train_hadm_ids | test_hadm_ids

        notes    = load_notes(cfg, hadm_ids=all_hadm_ids)
        labels   = matrix[["hadm_id", "subject_id", TARGET_COL]]
        notes_df = build_notes_dataframe(notes, labels)

        train_df = notes_df[notes_df["hadm_id"].isin(train_hadm_ids)].reset_index(drop=True)
        test_df  = notes_df[notes_df["hadm_id"].isin(test_hadm_ids)].reset_index(drop=True)

        section_result("Notes loaded",      len(notes))
        section_result("Train notes",       len(train_df))
        section_result("Test notes",        len(test_df))
        section_result("Coverage (train)",  f"{len(train_df)/max(len(train_hadm_ids),1):.1%} of Stage 1 train admissions have a note")
        section_result("Coverage (test)",   f"{len(test_df)/max(len(test_hadm_ids),1):.1%} of Stage 1 test admissions have a note")

        if len(train_df) == 0:
            fail("No training notes found — hadm_ids from Stage 1 have no matching notes. "
                 "Check that MIMIC_IV_DIR and MIMIC_IV_NOTE_DIR are from compatible datasets.")
        ok()
    except SystemExit:
        raise
    except Exception as e:
        fail(e)


    # ── Tokenizer ─────────────────────────────────────────────────────────────
    step(f"Loading tokenizer: {model_name_s2}")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_s2)
        section_result("Vocab size", tokenizer.vocab_size)
        ok("(downloads ~500 MB on first run — cached afterwards)")
    except Exception as e:
        fail(e)


    # ── Build datasets + verify lazy tokenization ─────────────────────────────
    step("Building ClinicalNotesDataset (lazy tokenization)")
    try:
        max_length = cfg["stage2"]["max_seq_length"]
        train_ds = ClinicalNotesDataset(
            train_df["text"].tolist(), train_df[TARGET_COL].tolist(), tokenizer, max_length)
        test_ds = ClinicalNotesDataset(
            test_df["text"].tolist(), test_df[TARGET_COL].tolist(), tokenizer, max_length)

        sample = train_ds[0]
        section_result("Train dataset size", len(train_ds))
        section_result("Test dataset size",  len(test_ds))
        section_result("Token shape",        tuple(sample["input_ids"].shape))
        ok()
    except Exception as e:
        fail(e)


    # ── Load model ────────────────────────────────────────────────────────────
    step(f"Loading model: {model_name_s2}")
    try:
        from transformers import LongformerForSequenceClassification
        model_s2 = LongformerForSequenceClassification.from_pretrained(
            model_name_s2, num_labels=2, ignore_mismatched_sizes=True)
        n_params = sum(p.numel() for p in model_s2.parameters()) / 1e6
        section_result("Parameters", f"{n_params:.0f}M")
        ok()
    except Exception as e:
        fail(e)


    # ── Smoke-test fine-tuning (4 steps) ──────────────────────────────────────
    step("Fine-tuning smoke test (4 gradient steps — verifies training loop only)")
    try:
        from transformers import (
            Trainer, TrainingArguments, EarlyStoppingCallback
        )
        from sklearn.metrics import average_precision_score, roc_auc_score

        smoke_dir = tempfile.mkdtemp(prefix="stage2_smoke_")
        class_weights = torch.tensor(
            [1.0, float((train_df[TARGET_COL] == 0).sum()) / max(int((train_df[TARGET_COL] == 1).sum()), 1)],
            dtype=torch.float32,
        )

        def _compute_metrics(eval_pred):
            logits, labels = eval_pred
            probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()
            auprc_val = float(average_precision_score(labels, probs))
            try:
                auroc_val = float(roc_auc_score(labels, probs))
            except ValueError:
                auroc_val = 0.0
            return {"auprc": auprc_val, "auroc": auroc_val}

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
                loss = loss_fn(outputs.logits, labels)
                return (loss, outputs) if return_outputs else loss

        smoke_args = TrainingArguments(
            output_dir=smoke_dir,
            max_steps=4,                      # hard cap — just verifies no crash
            per_device_train_batch_size=cfg["stage2"]["batch_size"],
            per_device_eval_batch_size=cfg["stage2"]["batch_size"] * 2,
            gradient_accumulation_steps=2,    # reduced for smoke test
            learning_rate=cfg["stage2"]["learning_rate"],
            eval_strategy="no",               # skip eval in smoke test
            save_strategy="no",
            fp16=False,
            use_cpu=True,
            seed=cfg["run"]["random_state"],
            report_to="none",
            logging_steps=1,
            dataloader_num_workers=0,
        )
        smoke_trainer = WeightedTrainer(
            model=model_s2,
            args=smoke_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=_compute_metrics,
        )
        smoke_trainer.train()
        section_result("Training steps completed", 4)
        ok("Training loop works. Run `python setup_stage2.py` for full fine-tuning.")
    except Exception as e:
        fail(e)


    # ── Stage 2 predict (quick sanity check using stage1 artifact) ────────────
    step("Stage 2 inference sanity check (scores a sample of flagged patients)")
    print(
        "  NOTE: Using the smoke-test model (4 steps) — scores are random.\n"
        "  This step only verifies the inference pipeline runs end-to-end.",
        flush=True
    )
    try:
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        stage1_scores = artifact["estimator"].predict_proba(
            X.iloc[artifact["test_idx"]][artifact["feature_cols"]])[:, 1]
        stage1_threshold = artifact["threshold"]
        from src.schemas import TARGET_COL as TC

        test_meta = matrix.iloc[artifact["test_idx"]][["hadm_id", "subject_id", TC]].reset_index(drop=True)
        test_meta = test_meta.copy()
        test_meta["stage1_score"] = stage1_scores
        flagged = test_meta[test_meta["stage1_score"] >= stage1_threshold].reset_index(drop=True)

        # Only score admissions that have notes
        flagged_with_notes = flagged[flagged["hadm_id"].isin(notes_df["hadm_id"])].reset_index(drop=True)
        sample = flagged_with_notes.head(min(20, len(flagged_with_notes)))  # score max 20 for speed

        if len(sample) == 0:
            warn("No flagged admissions have matching notes — inference skipped. "
                 "Check dataset alignment.")
        else:
            sample_notes = notes_df[notes_df["hadm_id"].isin(sample["hadm_id"])].reset_index(drop=True)
            infer_ds = ClinicalNotesDataset(
                sample_notes["text"].tolist(),
                sample_notes[TC].tolist(),
                tokenizer, max_length,
            )
            loader = DataLoader(infer_ds, batch_size=2, shuffle=False)
            model_s2.eval()
            probs = []
            with torch.no_grad():
                for batch in loader:
                    inputs = {k: v for k, v in batch.items() if k != "labels"}
                    logits = model_s2(**inputs).logits
                    probs.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())

            section_result("Stage 1 flagged",   len(flagged))
            section_result("With notes",         len(flagged_with_notes))
            section_result("Scored (sample)",    len(sample))
            section_result("Stage 2 scores",     [round(p, 3) for p in probs[:5]] + (["..."] if len(probs) > 5 else []))
            ok("Inference pipeline runs end-to-end.")
    except Exception as e:
        fail(e)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Explanation generation via Ollama + phi4-mini
# ══════════════════════════════════════════════════════════════════════════════

banner("STAGE 3 — Explanation Generation (phi4-mini via Ollama)")

if not ollama_ok:
    print("  Skipped — Ollama not running or phi4-mini not pulled.", flush=True)
    print(f"  To enable: start Ollama, then run: ollama pull {cfg['stage3']['ollama_model']}", flush=True)
else:
    step("Generating one sample explanation")
    try:
        from src.stage3.explain import generate_explanation

        # Build a minimal patient record from the feature matrix for the prompt
        sample_patient = matrix.iloc[0].to_dict()
        sample_patient["stage1_score"]   = 0.75
        sample_patient["stage2_score"]   = 0.65
        sample_patient["stage2_confirmed"] = 1

        cfg_with_stage3 = {**cfg, "stage3": {**cfg["stage3"], "enabled": True}}
        explanation = generate_explanation(sample_patient, cfg_with_stage3)

        section_result("Patient hadm_id",  sample_patient.get("hadm_id", "n/a"))
        section_result("Stage 1 score",    sample_patient["stage1_score"])
        section_result("Stage 2 score",    sample_patient["stage2_score"])
        print(f"\n  --- Explanation ---", flush=True)
        print(f"  {explanation}", flush=True)
        print(f"  -------------------", flush=True)
        ok()
    except Exception as e:
        fail(e)


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

banner("DIAGNOSTIC COMPLETE")
print(f"""
  {GREEN}All pipeline stages ran without a crash.{RESET}

  What was verified:
    Stage 1  ✓  Feature matrix built from {'real MIMIC-IV' if has_real_data(cfg) else 'SYNTHETIC'} data
    Stage 1  ✓  XGBoost trained + artifact saved
    Stage 2  {'✓  Notes loaded + training loop verified (smoke test)' if note_dir_ok else '⚠  Skipped (MIMIC_IV_NOTE_DIR not set)'}
    Stage 3  {'✓  phi4-mini explanation generated' if ollama_ok else '⚠  Skipped (Ollama not running)'}

  Next steps:
    • Full Stage 2 fine-tuning:   python setup_stage2.py
    • Full Stage 3 explanations:  python setup_stage3.py
    • Evaluate Stage 1:           python -m src.model.evaluate
""", flush=True)
