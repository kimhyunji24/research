"""Compare every decimal number in the LaTeX manuscripts with the md drafts they were converted from.

    python3 scripts/2026-10-20/check_tex_numbers.py
Comments are removed from both sides; a number counted a different number of times is reported.
"""
import collections
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NUM = re.compile(r"\d+\.\d+")


def md_numbers(path):
    t = open(path, encoding="utf-8").read()
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = t.split("## References")[0].split("## 참고문헌")[0]
    t = re.sub(r"!\[Fig\. \d\]\([^)]*\)", "", t)  # image links
    # section cross-references ("Section 4.2", "4.2절", headings) become \ref in LaTeX
    t = re.sub(r"(Sections?|Section) \d\.\d|\d\.\d절|^#+ \d\.\d", "", t, flags=re.M)
    return collections.Counter(NUM.findall(t))


def tex_numbers(path):
    t = "".join(re.sub(r"(?<!\\)%.*", "", l) for l in open(path, encoding="utf-8"))  # comments, not \%
    t = t.split("\\begin{document}")[1]
    t = re.sub(r"\\includegraphics(\[[^]]*\])?\{[^}]*\}", "", t)
    t = re.sub(r"\\begin\{tabularx\}\{\\linewidth\}\{[^\n]*", "", t)  # column specs (p{2.2cm} etc.)
    t = re.sub(r"width=0\.\d+", "", t)
    return collections.Counter(NUM.findall(t))


bad = 0
for md, tex in (("docs/paper_draft_en.md", "paper/main_en.tex"), ("docs/paper_draft_ko.md", "paper/main_ko.tex")):
    a, b = md_numbers(os.path.join(ROOT, md)), tex_numbers(os.path.join(ROOT, tex))
    diff = {k: (a[k], b[k]) for k in set(a) | set(b) if a[k] != b[k]}
    print(f"{tex}: {sum(a.values())} numbers in md, {sum(b.values())} in tex; differing values: {len(diff)}")
    for k, (x, y) in sorted(diff.items()):
        print(f"   {k}: md {x} vs tex {y}")
    bad += len(diff)
print("RESULT:", "all numbers match" if not bad else f"{bad} differing values")
