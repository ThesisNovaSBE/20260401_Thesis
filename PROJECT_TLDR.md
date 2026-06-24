This is the single source of truth for project context. LLMs and teammates should read this file first before doing any work. Then read the most recent file in `sessions/` to see current state.

# LLM-Augmented Hospital Readmission Prediction — A Two-Stage "Triage-and-Verify" Approach

**Program:** M.Sc. Business Analytics, Nova SBE
**Advisor:** Prof. Yufei Shen
**Team size:** 4

## Goal

Predict unplanned hospital readmission as accurately as possible using a two-stage pipeline:

1. A classical ML model flags high-risk patients from **structured** MIMIC-IV data (tuned for high recall).
2. A clinical language model reads the **clinical notes** (MIMIC-IV-Note) of flagged patients and prunes false positives.

## Pipeline Stages

### Stage 1 — Classical ML (Structured Data)

- Models: Logistic Regression, XGBoost, HistGradientBoosting
- Input: Structured EHR features from MIMIC-IV
- Objective: **High recall** — cast a wide net, accept more false positives

### Stage 2 — Clinical Encoder (Notes)

- Model: Fine-tuned Bio_ClinicalBERT
- Input: Clinical notes (MIMIC-IV-Note) for patients flagged in Stage 1
- Objective: Confirm or reject each flag, reducing false positives
- Runs locally, small and fast

### Stage 3 — Explanation (Optional, Later)

- Model: Local quantized generative model via Ollama (Gemma 3 or Mistral)
- Input: Confirmed high-risk cases from Stage 2
- Objective: Plain-language explanation of why a patient is flagged

## Data

- **MIMIC-IV** (structured tables) + **MIMIC-IV-Note** (clinical notes)
- Source: PhysioNet — credentialed access required

## Team Roles

- 1 person: literature review
- 2 people: core coding (data pipeline + models)
- 1 person (joining later): evaluation, explanation layer, integration

## Design Principles

- **Simplicity** — keep the architecture straightforward
- **Small/local models** — no large cloud APIs in the pipeline
- **Build it ourselves** — understand every component
