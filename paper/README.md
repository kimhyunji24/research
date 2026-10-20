# LaTeX manuscript (English and Korean)

- `main_en.tex`: English manuscript. `main_ko.tex`: Korean manuscript (same content).
- Text source: `docs/paper_draft_en.md` and `docs/paper_draft_ko.md` (converted 2026-10-20). Edit the md drafts first, then carry the change into both `.tex` files.
- `refs.bib` is shared. Bibliography style `unsrtnat` numbers references in order of first citation, so the numbers differ from the md drafts.
- `figures/` holds copies of `../figures/*.pdf` (drawn by `scripts/2026-10-20/make_figures.py`; Korean figures are the `*_ko.pdf` files). After redrawing, copy them again: `cp ../figures/*.pdf figures/`.
- Both files need **XeLaTeX** (`latexmkrc` selects it). Local build: `latexmk main_en.tex` and `latexmk main_ko.tex`.

## Full and conference versions
- `main_en.tex` / `main_ko.tex` are the **full version** (arXiv, journal).
- `main_en_short.tex` is the **conference version** (single column, 9 pages of main text; letter paper, 5.5 x 9 in text, 10 pt Times-like font). It condenses the full version and adds no numbers or claims of its own; its tables and figures are `\input` from `floats_en/`, which `scripts/2026-10-21/extract_floats.py en` generates from `main_en.tex`. After editing the full version, rerun that script, then `python3 scripts/2026-10-21/check_short_numbers.py en` (every number in the short text must exist in the full version; also prints the page where references start).
- Edit order: md draft -> full `.tex` -> extract floats -> short `.tex`.

## Overleaf
1. Zip this folder (`main_en.tex`, `main_ko.tex`, `main_en_short.tex`, `refs.bib`, `latexmkrc`, `figures/`, `floats_en/`, `floats_ko/`) and use New Project > Upload Project.
2. Menu > Compiler: XeLaTeX. Menu > Main document: `main_en.tex`, `main_ko.tex` or `main_en_short.tex`.
3. The Korean file uses `kotex` with the Noto Serif CJK KR font when it is installed (Overleaf has it); otherwise kotex's default Hangul font is used.

## Numbering differences from the md drafts
- LaTeX numbers figures 1–3 (the md drafts call them Fig. 2–4; no Fig. 1 exists yet).
- Tables follow their position: md Table 6 (prior-style baselines) is Table 5 here and md Table 5 (early termination) is Table 6.
