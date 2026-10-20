"""Check that the conference (short) version adds no numbers: every decimal and percentage in its text must appear in the full version.

    python3 scripts/2026-10-21/check_short_numbers.py en|ko
Also reports the page on which the main text ends (the "Code and data availability" heading) in the compiled PDF.
"""
import io
import os
import re
import subprocess
import sys

lang = sys.argv[1]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(ROOT, "paper")
NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*%?")


def text(path):
    t = "".join(re.sub(r"(?<!\\)%.*", "", l) for l in io.open(path, encoding="utf-8"))
    t = t.split("\\begin{document}")[1]
    t = re.sub(r"\\input\{[^}]*\}", " ", t)  # floats are shared files, identical in both versions
    t = re.sub(r"\\(cite|ref|label|includegraphics|setcounter|renewcommand)\{[^}]*\}(\{[^}]*\})?", " ", t)
    t = re.sub(r"\\(begin|end)\{[^}]*\}(\{[^}]*\})?(\{[^}]*\})?", " ", t)
    return t


full = text(os.path.join(P, f"main_{lang}.tex"))
short = text(os.path.join(P, f"main_{lang}_short.tex"))
fullset = set(m.group(0).lstrip("+-") for m in NUM.finditer(full))
missing = sorted(set(m.group(0).lstrip("+-") for m in NUM.finditer(short)) - fullset)
print(f"numbers in short version not found in the full version: {missing if missing else 'none'}")

pdf = os.path.join(P, f"main_{lang}_short.pdf")
if os.path.exists(pdf):
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
    for p in range(1, n + 1):
        pg = subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), pdf, "-"], capture_output=True, text=True).stdout
        if re.search(r"^(References|참고문헌)$", pg, re.M):
            print(f"PDF pages: {n}; references start on page {p}")
            break
