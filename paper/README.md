# LaTeX manuscript (English and Korean)

- `main_en.tex`: English manuscript. `main_ko.tex`: Korean manuscript (same content).
- Text source: `docs/paper_draft_en.md` and `docs/paper_draft_ko.md` (converted 2026-10-20). Edit the md drafts first, then carry the change into both `.tex` files.
- `refs.bib` is shared. Bibliography style `unsrtnat` numbers references in order of first citation, so the numbers differ from the md drafts.
- `figures/` holds copies of `../figures/*.pdf` (drawn by `scripts/2026-10-20/make_figures.py`; Korean figures are the `*_ko.pdf` files). After redrawing, copy them again: `cp ../figures/*.pdf figures/`.
- Both files need **XeLaTeX** (`latexmkrc` selects it). Local build: `latexmk main_en.tex` and `latexmk main_ko.tex`.

## Overleaf
1. Zip this folder (`main_en.tex`, `main_ko.tex`, `refs.bib`, `latexmkrc`, `figures/`) and use New Project > Upload Project.
2. Menu > Compiler: XeLaTeX. Menu > Main document: `main_en.tex` or `main_ko.tex`.
3. The Korean file uses `kotex` with the Noto Serif CJK KR font when it is installed (Overleaf has it); otherwise kotex's default Hangul font is used.

## Numbering differences from the md drafts
- LaTeX numbers figures 1–3 (the md drafts call them Fig. 2–4; no Fig. 1 exists yet).
- Tables follow their position: md Table 6 (prior-style baselines) is Table 5 here and md Table 5 (early termination) is Table 6.
