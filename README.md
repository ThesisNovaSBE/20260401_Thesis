# Nova SBE Work Project — LaTeX Template

## Files
| File | Purpose |
|------|---------|
| `nova_sbe_thesis.tex` | Main template — edit this |
| `references.bib` | Bibliography entries |
| `nova_sbe_logo.png` | Place your Nova SBE logo here |

## How to Compile

**Recommended (full bibliography + cross-references):**
```bash
pdflatex nova_sbe_thesis
biber nova_sbe_thesis
pdflatex nova_sbe_thesis
pdflatex nova_sbe_thesis
```

**Or use latexmk (auto-runs all passes):**
```bash
latexmk -pdf nova_sbe_thesis.tex
```

**Overleaf:** Upload all files, set compiler to `pdfLaTeX`, and bibliography tool to `biber`.

## Customise
- Fill in `\projecttitle`, `\programname`, `\defensedate` near the top of the .tex file
- Drop your Nova SBE logo as `nova_sbe_logo.png` in the same folder
- Add references to `references.bib` following the AER examples provided

## Citation Commands
| Command | Output |
|---------|--------|
| `\parencite{key}` | (Author Year) |
| `\parencite{key,13}` | (Author Year, 13) |
| `\textcite{key}` | Author (Year) |

## Page Budget Reminder
| Section | Pages | Counts toward limit? |
|---------|-------|----------------------|
| Cover page | p. 0 | No |
| Abstract page | p. 1 | No |
| Table of Contents (optional) | After p. 1 | **Yes** |
| Work Project | pp. 2–25 | **Yes** |
| References | p. 26+ | No |
| Appendices | After references | No |
