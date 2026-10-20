"""Check that every value drawn in Figs. 2-4 matches the manuscript tables/text and fits the axes.

Reads the same CSVs as make_figures.py; prints a comparison and reports any mismatch.
    /root/figenv/bin/python scripts/2026-10-20/check_figures.py
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
bad = []


def read(n):
    return pd.read_csv(os.path.join(RES, n))


def check(label, got, want, tol=0.0005):
    ok = abs(got - want) <= tol
    print(f"{'OK ' if ok else 'BAD'} {label}: csv {got:+.4f} vs draft {want:+.4f}")
    if not ok:
        bad.append(label)


def in_range(label, lo, hi, axlo, axhi):
    ok = axlo <= lo and hi <= axhi
    print(f"{'OK ' if ok else 'BAD'} {label}: data {lo:.3f}..{hi:.3f} within axis {axlo}..{axhi}")
    if not ok:
        bad.append(label)


# Fig. 2 (lines, steps 0-3) vs Table 1
print("== Fig. 2 vs Table 1")
base = read("baselines_main_3b.csv")
prim = read("probe_main_3b_primary.csv")
t1 = {"baseline1": [0.500] * 4, "baseline2": [0.603] * 4, "baseline3": [0.659, 0.633, 0.671, 0.700],
      "baseline4": [0.673, 0.700, 0.834, 0.814], "baseline5": [0.701, 0.732, 0.823, 0.818]}
for m, vals in t1.items():
    for s, v in enumerate(vals):
        r = base[(base.method == m) & (base.step == s)].iloc[0]
        check(f"{m} step {s}", r.auroc, v)
        assert r.n == 250 and r.n_fail == 194, (m, s)
        if s <= 2:  # the ladder in baselines_main_3b equals the one stored with the pre-registered run
            p = prim[(prim.method == m) & (prim.step == s)].iloc[0]
            assert abs(p.auroc - r.auroc) < 1e-12 and abs(p.ci_lo - r.ci_lo) < 1e-12, (m, s)
for s, v in enumerate([0.520, 0.599, 0.665]):
    r = prim[(prim.method == "linear_pcafull") & (prim.step == s)].iloc[0]
    check(f"probe step {s}", r.auroc, v)
assert sorted(prim[prim.method == "linear_pcafull"].step) == [0, 1, 2]
print("ladder identical in both files for steps 0-2; probe has steps 0-2 only; n = 250, 194 failures")
b5 = base[base.method == "baseline5"]
pr = prim[prim.method == "linear_pcafull"]
in_range("Fig. 2 values and shaded CIs", min(b5.ci_lo.min(), pr.ci_lo.min(), 0.5),
         max(b5.ci_hi.max(), pr.ci_hi.max(), base.auroc.max()), 0.40, 0.92)

# Fig. 3 (forest, both levels) vs Tables 1, 2, 3, 4
print("== Fig. 3 vs Tables 1-4")
rows = {
    "pre-registered 3B 50%": ("probe_main_3b_primary_diff.csv",
                              [(-0.181, -0.276, -0.069), (-0.133, -0.230, -0.031), (-0.159, -0.240, -0.073)]),
    "confirmatory 3B 75%": ("confirm_main_3b_diff.csv",
                            [(0.039, -0.030, 0.100), (0.035, -0.030, 0.096), (-0.029, -0.087, 0.029)]),
    "Qwen3-4B 75%": ("task_qwen3_4b_diff_p.csv",
                     [(-0.049, -0.103, 0.005), (0.001, -0.076, 0.071), (-0.037, -0.096, 0.033)]),
    "ReAct 75%": ("task_react_qwen25_3b_diff_p.csv",
                  [(-0.002, -0.057, 0.045), (-0.036, -0.096, 0.021), (-0.128, -0.204, -0.058)]),
}
los, his = [], []
for lab, (f, vals) in rows.items():
    dd = read(f)
    for s, (est, lo, hi) in enumerate(vals):
        r = dd[(dd.method == "linear_pcafull") & (dd.vs == "baseline5") & (dd.step == s)].iloc[0]
        check(f"{lab} step {s} est", r.auroc_diff, est)
        check(f"{lab} step {s} lo", r.ci_lo, lo)
        check(f"{lab} step {s} hi", r.ci_hi, hi)
        los.append(r.ci_lo); his.append(r.ci_hi)
t3 = {"baseline4": [(0.003, -0.058, 0.063), (-0.026, -0.078, 0.019), (0.054, -0.017, 0.136)],
      "baseline5": [(-0.027, -0.088, 0.025), (-0.040, -0.088, 0.002), (0.007, -0.038, 0.051)]}
wg = read("within_game_diff.csv")
for vs, vals in t3.items():
    for k, (est, lo, hi) in enumerate(vals):
        r = wg[(wg.a == "probe") & (wg.b == vs) & (wg.step == k + 1)].iloc[0]
        check(f"within-game probe - {vs} step {k + 1} est", r["diff"], est)
        check(f"within-game probe - {vs} step {k + 1} lo", r.ci_lo, lo)
        check(f"within-game probe - {vs} step {k + 1} hi", r.ci_hi, hi)
        los.append(r.ci_lo); his.append(r.ci_hi)
sup = sum(lo > 0 for lo in los)
print(f"superior (lower bound > 0): {sup} of {len(los)} comparisons")
if sup:
    bad.append("a comparison is superior; text says none")
in_range("Fig. 3 CIs", min(los), max(his), -0.35, 0.16)

# Fig. 4 (heatmap, linear, 4 readers x 4 depths x 3 steps) vs Appendix Table A6 and Section 4.1
print("== Fig. 4 vs Table A6 and Section 4.1")
a6 = {"0.5B": [0.618, 0.615, 0.660, 0.642], "1.5B": [0.589, 0.631, 0.717, 0.675],
      "3B": [0.595, 0.594, 0.727, 0.682], "7B-4bit": [0.608, 0.643, 0.704, 0.717]}
cells, dag, pos = [], 0, []
for k, means in a6.items():
    d = read(f"explore_{k}.csv")
    dd = read(f"explore_{k}_diff.csv")
    for c, m in enumerate(means):
        check(f"{k} L{c} mean AUROC (A6)", d[d.method == f"linear_L{c}_pcafull"].auroc.mean(), m)
        for s in range(3):
            r = dd[(dd.method == f"linear_L{c}_pcafull") & (dd.vs == "baseline5") & (dd.step == s)].iloc[0]
            cells.append(r.auroc_diff)
            dag += (r.ci_hi < 0) or (r.ci_lo > 0)
            if r.auroc_diff > 0:
                pos.append((k, c, s, round(r.auroc_diff, 4), round(r.ci_lo, 3), round(r.ci_hi, 3)))
print(f"linear cells: {len(cells)}; CI excludes zero (marked †): {dag}")
print("positive linear cells (reader, depth index, step, diff, lo, hi):", pos)
print(f"max linear cell {max(cells):+.4f} (text: no combination above +0.004)")
if max(cells) > 0.004 + 0.0005:
    bad.append("linear cell above +0.004")
in_range("Fig. 4 colour range", min(cells), max(cells), -0.25, 0.25)
# Dagger count by sign
neg_ex = 0
for k in a6:
    dd = read(f"explore_{k}_diff.csv")
    dd = dd[dd.method.str.startswith("linear_") & (dd.vs == "baseline5")]
    neg_ex += int((dd.ci_hi < 0).sum())
    if (dd.ci_lo > 0).any():
        bad.append(f"{k}: a linear cell has CI above zero")
print(f"cells with CI entirely below zero: {neg_ex} of 48; cells with CI entirely above zero: 0")

print("\nRESULT:", "all checks passed" if not bad else f"{len(bad)} problems: {bad}")
