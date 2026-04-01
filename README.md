# Nova SBE Work Project — LaTeX Template
## LLM vs. Structured ML for 30-Day Hospital Readmission Prediction Using MIMIC-IV
### Master of Science in Business Analytics

**Supervisor:** Assistant Professor Yufei Shen

---

## Team & Section Ownership

| Name | Student ID | Sections | Target pages |
|------|-----------|----------|-------------|
| Benjamin Iby | 70314 | Introduction, Results, Discussion, Conclusion + formatting & presentation | ~30 pp |
| Lennart Stenzel | 70485 | Data & Methodology: Modelling (3.4–3.6) | ~19 pp |
| Martin Schmitz | 75242 | Data & Methodology: Data (3.1–3.3) | ~19 pp |
| Thomas Teichmann | 74817 | Literature Review (Section 2) | ~22 pp |

**Total body target: ~90 pages** (10 pp buffer for front matter lists)

---

## Page Budget

| Section | Owner | Target | Counts toward limit? |
|---------|-------|--------|----------------------|
| Cover page | — | p. 0 | No |
| Abstract | — | p. 1 | No |
| Table of Contents | — | ~1 pp | **Yes** |
| List of Figures | — | ~1 pp | **Yes** |
| List of Tables | — | ~1 pp | **Yes** |
| List of Abbreviations | — | ~1 pp | **Yes** |
| 1. Introduction | Benjamin | ~2 pp | **Yes** |
| 2. Literature Review | Thomas | ~22 pp | **Yes** |
| 3. Data & Methodology | Martin + Lennart | ~38 pp | **Yes** |
| 4. Results | Benjamin | ~18 pp | **Yes** |
| 5. Discussion | Benjamin | ~8 pp | **Yes** |
| 6. Conclusion | Benjamin | ~2 pp | **Yes** |
| References | — | p. 26+ | No |
| Appendices | — | unlimited | No |

---

## Files

| File | Purpose |
|------|---------|
| `nova_sbe_thesis.tex` | Main template — edit this |
| `references.bib` | Bibliography entries |
| `nova_logo.jpg` | Nova SBE logo (already included) |

---

## How to Compile

**Recommended (full bibliography + acronyms + cross-references):**
```bash
pdflatex nova_sbe_thesis
makeglossaries nova_sbe_thesis
biber nova_sbe_thesis
pdflatex nova_sbe_thesis
pdflatex nova_sbe_thesis
```
> `makeglossaries` is required to build the List of Abbreviations. Run it after the first `pdflatex` pass.

**Or use latexmk (handles all passes automatically):**
```bash
latexmk -pdf nova_sbe_thesis.tex
```

**Overleaf:**
1. Upload all files (`nova_sbe_thesis.tex`, `references.bib`, `nova_logo.jpg`)
2. Set compiler → `pdfLaTeX`
3. Set bibliography tool → `biber`
4. Overleaf handles glossaries automatically — no extra steps needed

---

## How to Use Abbreviations

Abbreviations are defined near the top of `nova_sbe_thesis.tex`. To add a new one:
```latex
\newacronym{key}{ABBR}{Full Form}
```

To use in text:
```latex
\gls{key}      % First use: "Full Form (ABBR)"; subsequent: "ABBR"
\glspl{key}    % Plural form
\Gls{key}      % Capitalised first letter
```

The List of Abbreviations is generated automatically from all `\gls{}` calls in the document. Only abbreviations actually used in the text will appear.

---

## Citation Commands

| Command | Output |
|---------|--------|
| `\parencite{key}` | (Author Year) |
| `\parencite[p.~13]{key}` | (Author Year, p. 13) |
| `\textcite{key}` | Author (Year) |

---

## Current References (`references.bib`)

| Cite key | Reference |
|----------|-----------|
| `johnson2023mimic` | Johnson et al. (2023) — MIMIC-IV dataset |
| `johnson2023note` | Johnson et al. (2023) — MIMIC-IV-Note |
| `alsentzer2019` | Alsentzer et al. (2019) — ClinicalBERT |
| `huang2020clinicalbert` | Huang et al. (2020) — ClinicalBERT & readmission |
| `gupta2022` | Gupta et al. (2022) — MIMIC-IV processing pipeline |

---

## Thesis Structure

```
Abstract                                         (p. 1, not counted)
Table of Contents
List of Figures
List of Tables
List of Abbreviations

1. Introduction                      ~2 pp    Benjamin Iby
2. Literature Review                ~22 pp    Thomas Teichmann
   2.1  Hospital Readmission Prediction
        2.1.1  Clinical and Economic Context
        2.1.2  Structured Machine Learning Approaches
        2.1.3  The Underutilisation of Clinical Notes
   2.2  Clinical NLP and Large Language Models
        2.2.1  Evolution of NLP in Healthcare
        2.2.2  BERT and ClinicalBERT
        2.2.3  Modern Large Language Models
        2.2.4  LLMs for Clinical Prediction Tasks
   2.3  The MIMIC Dataset Family
   2.4  Research Gap and Contribution
3. Data and Methodology             ~38 pp    Martin Schmitz + Lennart Stenzel
   3.1  Research Objectives and Questions       Martin
   3.2  Dataset: MIMIC-IV and MIMIC-IV-Note     Martin
        3.2.1  Cohort Definition and Label Engineering
        3.2.2  Exploratory Data Analysis
   3.3  Structured ML Baseline                  Martin
   3.4  LLM-Based Note Modelling                Lennart
        3.4.1  Clinical Note Preprocessing
        3.4.2  Representation Strategy
   3.5  Combined Model                          Lennart
   3.6  Evaluation Framework                    Lennart
4. Results                          ~18 pp    Benjamin Iby
   4.1  Structured ML Baseline Performance
   4.2  LLM-Based Model Performance
   4.3  Combined Model Performance
   4.4  Comparative Analysis
5. Discussion                        ~8 pp    Benjamin Iby
   5.1  Do LLMs Close the Gap?
   5.2  Clinical and Practical Implications
   5.3  Limitations
   5.4  Future Work
6. Conclusion                        ~2 pp    Benjamin Iby

References                                       (not counted)
Appendix A: Data Processing Pipeline             (not counted)
Appendix B: Model Hyperparameters                (not counted)
Appendix C: Additional Results and Robustness Checks
```
