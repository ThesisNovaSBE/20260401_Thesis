# Nova SBE Work Project — LaTeX Template
## LLM vs. Structured ML for 30-Day Hospital Readmission Prediction Using MIMIC-IV
### Master of Science in Business Analytics

**Team:** Benjamin Iby (70314) · Lennart Stenzel (70485) · Martin Schmitz (75242) · Thomas Teichmann (74817)
**Supervisor:** Assistant Professor Yufei Shen

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
> The extra `makeglossaries` pass is required to build the List of Abbreviations.

**Or use latexmk (auto-runs all passes including glossaries):**
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

Abbreviations are defined at the top of `nova_sbe_thesis.tex` using `\newacronym`. To add a new one:
```latex
\newacronym{key}{ABBR}{Full Form}
```

To use in text:
```latex
\gls{key}      % First use: "Full Form (ABBR)"; subsequent uses: "ABBR"
\glspl{key}    % Plural form
\Gls{key}      % Capitalised first letter
```

The List of Abbreviations is generated automatically from all `\gls{}` calls used in the document.

---

## Citation Commands

| Command | Output |
|---------|--------|
| `\parencite{key}` | (Author Year) |
| `\parencite[p.~13]{key}` | (Author Year, p. 13) |
| `\textcite{key}` | Author (Year) |

---

## Current References (references.bib)

| Cite key | Reference |
|----------|-----------|
| `johnson2023mimic` | Johnson et al. (2023) — MIMIC-IV dataset |
| `johnson2023note` | Johnson et al. (2023) — MIMIC-IV-Note |
| `alsentzer2019` | Alsentzer et al. (2019) — ClinicalBERT |
| `huang2020clinicalbert` | Huang et al. (2020) — ClinicalBERT & readmission |
| `gupta2022` | Gupta et al. (2022) — MIMIC-IV processing pipeline |

---

## Page Budget

| Section | Pages | Counts toward 25-page limit? |
|---------|-------|-------------------------------|
| Cover page | p. 0 | No |
| Abstract page | p. 1 | No |
| Table of Contents | After p. 1 | **Yes** |
| List of Figures | — | **Yes** |
| List of Tables | — | **Yes** |
| List of Abbreviations | — | **Yes** |
| Work Project body | pp. 2–25 | **Yes** |
| References | p. 26+ | No |
| Appendices | After references | No |

> **Note:** The front matter lists (TOC, figures, tables, abbreviations) count toward the 25-page limit. Keep the body text within budget accordingly.

---

## Thesis Structure

```
1. Introduction                        (~1–2 pages)
2. Literature Review                   (~15–20 pages)
   2.1 Hospital Readmission Prediction
       2.1.1 Clinical and Economic Context
       2.1.2 Structured ML Approaches
       2.1.3 Underutilisation of Clinical Notes
   2.2 Clinical NLP and Large Language Models
       2.2.1 Evolution of NLP in Healthcare
       2.2.2 BERT and ClinicalBERT
       2.2.3 Modern Large Language Models
       2.2.4 LLMs for Clinical Prediction Tasks
   2.3 The MIMIC Dataset Family
   2.4 Research Gap and Contribution
3. Data and Methodology                (~15–20 pages)
   3.1 Research Objectives and Questions
   3.2 Dataset: MIMIC-IV and MIMIC-IV-Note
       3.2.1 Cohort Definition and Label Engineering
       3.2.2 Exploratory Data Analysis
   3.3 Structured ML Baseline
   3.4 LLM-Based Note Modelling
       3.4.1 Clinical Note Preprocessing
       3.4.2 Representation Strategy
   3.5 Combined Model
   3.6 Evaluation Framework
4. Results                             (~10–15 pages)
   4.1 Structured ML Baseline Performance
   4.2 LLM-Based Model Performance
   4.3 Combined Model Performance
   4.4 Comparative Analysis
5. Discussion                          (~5–8 pages)
   5.1 Do LLMs Close the Gap?
   5.2 Clinical and Practical Implications
   5.3 Limitations
   5.4 Future Work
6. Conclusion                          (~1–2 pages)
References
Appendix A: Data Processing Pipeline
Appendix B: Model Hyperparameters
Appendix C: Additional Results and Robustness Checks
```
