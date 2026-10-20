"""Figures 2-4 of the manuscript, drawn only from existing results/*.csv (no new analysis).

Run with a separate plotting environment so the frozen .venv is not changed:
    /root/figenv/bin/python scripts/2026-10-20/make_figures.py [--lang en|ko] [--overwrite]
Writes figures/fig{2,3,4}_*.pdf and .png (Korean: *_ko.pdf/png, needs the NanumGothic font);
stops if any output of that language already exists.
Values are checked against the manuscript by scripts/2026-10-20/check_figures.py.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")

# Reference palette (dataviz skill, light mode).
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAYS = ["#c3c2b7", "#a8a7a0", "#898781", "#6b6a66"]  # rungs 1-4 (context, de-emphasized)
NEUTRAL = "#f0efec"
# Diverging: red pole = probe below rung 5, blue pole = probe above, gray midpoint at 0.
DIVERGE = LinearSegmentedColormap.from_list(
    "probe_minus_rung5", ["#a52b2b", "#e34948", "#f4c4c0", NEUTRAL, "#b7d3f6", "#2a78d6", "#184f95"])

STEMS = {2: "fig2_baseline_ladder", 3: "fig3_probe_minus_baseline", 4: "fig4_reader_depth"}
OLD_STEMS = ["fig3_probe_minus_rung5", "fig4_reader_size"]  # earlier forms, removed on redraw

# All figure text; Korean terms follow docs/paper_draft_ko.md (단, 프로브, 사전 등록, 확증, 탐색, reader).
TEXT = {
    "en": {
        "font": ["DejaVu Sans"], "suffix": "",
        "rung": {1: "Rung 1: step index", 2: "Rung 2: + task type", 3: "Rung 3: + repetition, action prob.",
                 4: "Rung 4: + progress", 5: "Rung 5: + TF-IDF of prompt"},
        "probe2": "Probe (3B, 50%, pre-registered);\nno step 3: not a pre-registered round",
        "step": "Step {}", "row_step": "step {}",
        "auroc": "Failure-prediction AUROC",
        "shade2": "Shaded: 95% CI (scene-level cluster bootstrap) for rung 5 and the probe",
        "task": ["Qwen2.5-3B, v2, 50% (pre-registered; 250 ep., 194 fail.)",
                 "Qwen2.5-3B, v2, 75% (confirmatory hold-out; 250, 205)",
                 "Qwen3-4B, v2, 75% (pre-specified; 250, 156)",
                 "Qwen2.5-3B, ReAct prompt, 75% (pre-recorded; 250, 161)"],
        "wg": {"baseline4": "Probe − rung 4", "baseline5": "Probe − rung 5"},
        "wg_n": "{} (pre-registered; 39 games, 470 pairs/step)",
        "pre": "Pre-registered / pre-specified", "conf": "Confirmatory hold-out",
        "sec": ["Task level (between games): probe − rung 5", "Attempt level (within games, Qwen2.5-3B, 75%)"],
        "margin": "−0.05 margin\n(confirmatory only;\nset after exploration)",
        "diff3": "Probe − baseline (AUROC difference, 95% CI)",
        "r7b": "7B (4-bit)", "depth": "Reader depth (share of layers)", "reader": "Reader (Qwen2.5-Instruct)",
        "cbar": "Probe − rung 5 (AUROC)",
        "explore": "Exploratory (chosen after the pre-registered result; same 250 episodes)",
    },
    "ko": {
        "font": ["NanumGothic", "DejaVu Sans"], "suffix": "_ko",
        "rung": {1: "1단: step 번호", 2: "2단: + 과제 유형", 3: "3단: + 반복, 행동 확률",
                 4: "4단: + 진행 정도", 5: "5단: + 프롬프트 TF-IDF"},
        "probe2": "프로브(3B, 50%, 사전 등록);\nstep 3 없음: 사전 등록 라운드가 아님",
        "step": "step {}", "row_step": "step {}",
        "auroc": "실패 예측 AUROC",
        "shade2": "음영: 5단과 프로브의 95% CI(scene 단위 군집 bootstrap)",
        "task": ["Qwen2.5-3B, v2, 50% (사전 등록; 250 episode, 실패 194)",
                 "Qwen2.5-3B, v2, 75% (확증 hold-out; 250, 205)",
                 "Qwen3-4B, v2, 75% (사전 지정; 250, 156)",
                 "Qwen2.5-3B, ReAct 프롬프트, 75% (사전 기록; 250, 161)"],
        "wg": {"baseline4": "프로브 − 4단", "baseline5": "프로브 − 5단"},
        "wg_n": "{} (사전 등록; 게임 39개, step당 470쌍)",
        "pre": "사전 등록 / 사전 지정", "conf": "확증 hold-out",
        "sec": ["과제 수준(게임 간): 프로브 − 5단", "시도 수준(게임 내, Qwen2.5-3B, 75%)"],
        "margin": "−0.05 한계\n(확증에만 적용;\n탐색 뒤에 정함)",
        "diff3": "프로브 − baseline (AUROC 차이, 95% CI)",
        "r7b": "7B (4비트)", "depth": "reader 깊이(전체 층 중 비율)", "reader": "reader (Qwen2.5-Instruct)",
        "cbar": "프로브 − 5단 (AUROC)",
        "explore": "탐색(사전 등록 결과를 본 뒤 고름; 같은 250개 episode)",
    },
}
T = TEXT["en"]


def set_style(lang):
    global T
    T = TEXT[lang]
    plt.rcParams.update({
        "font.family": T["font"], "font.size": 8.5, "axes.unicode_minus": True,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def read(name):
    return pd.read_csv(os.path.join(RES, name))


def save(fig, stem):
    stem = stem + T["suffix"]
    for ext in ("pdf", "png"):
        # no creation date in the PDF, so a redraw from the same CSVs gives the same file
        meta = {"CreationDate": None} if ext == "pdf" else {}
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), bbox_inches="tight", facecolor="white", metadata=meta)
    plt.close(fig)
    print("wrote", stem)


def spread(ys, gap):
    """Nudge label y-positions apart (keeps order) so direct labels do not collide."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def fig2():
    """Task level, first 250 episodes: ladder over steps 0-3, rung 5 and the probe emphasized."""
    base = read("baselines_main_3b.csv")
    probe = read("probe_main_3b_primary.csv")
    probe = probe[probe.method == "linear_pcafull"].sort_values("step")
    names = T["rung"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ends = []
    for i in range(1, 6):
        s = base[base.method == f"baseline{i}"].sort_values("step")
        assert list(s.step) == [0, 1, 2, 3], i
        if i < 5:
            ax.plot(s.step, s.auroc, color=GRAYS[i - 1], lw=1.4, marker="o", ms=3.5, zorder=2)
            ends.append((s.auroc.iloc[-1], names[i], INK2))
        else:
            ax.fill_between(s.step, s.ci_lo, s.ci_hi, color=BLUE, alpha=0.13, lw=0, zorder=1)
            ax.plot(s.step, s.auroc, color=BLUE, lw=2.2, marker="o", ms=5, zorder=4)
            ends.append((s.auroc.iloc[-1], names[i], INK))
    ax.fill_between(probe.step, probe.ci_lo, probe.ci_hi, color=ORANGE, alpha=0.13, lw=0, zorder=1)
    ax.plot(probe.step, probe.auroc, color=ORANGE, lw=2.2, marker="D", ms=5, zorder=5)
    ax.text(1.1, 0.455, T["probe2"], fontsize=7, color=INK, va="center", ha="left")
    ys = spread([e[0] for e in ends], 0.03)
    for (y0, lab, col), y in zip(ends, ys):
        ax.text(3.08, y, lab, fontsize=7, color=col, va="center", ha="left")
    ax.set_xticks([0, 1, 2, 3], [T["step"].format(k) for k in range(4)])
    ax.set_xlim(-0.15, 3.05)
    ax.set_ylim(0.40, 0.92)
    ax.set_ylabel(T["auroc"])
    ax.tick_params(axis="x", length=0)
    ax.set_title(T["shade2"], fontsize=7.5, color=MUTED, loc="left")
    save(fig, STEMS[2])


def fig3():
    """Every pre-registered / pre-specified / confirmatory probe-vs-baseline comparison, both levels."""
    task = [  # label, status, file
        (T["task"][0], "pre", "probe_main_3b_primary_diff.csv"),
        (T["task"][1], "conf", "confirm_main_3b_diff.csv"),
        (T["task"][2], "pre", "task_qwen3_4b_diff_p.csv"),
        (T["task"][3], "pre", "task_react_qwen25_3b_diff_p.csv"),
    ]
    rows = []  # (y-label, est, lo, hi, status, group)
    for lab, st, f in task:
        d = read(f)
        d = d[(d.method == "linear_pcafull") & (d.vs == "baseline5")].sort_values("step")
        assert list(d.step) == [0, 1, 2], f
        for _, r in d.iterrows():
            rows.append((T["row_step"].format(int(r.step)), r.auroc_diff, r.ci_lo, r.ci_hi, st, lab))
    wg = read("within_game_diff.csv")
    for vs in ("baseline4", "baseline5"):
        d = wg[(wg.a == "probe") & (wg.b == vs)].sort_values("step")
        assert list(d.step) == [1, 2, 3], vs
        for _, r in d.iterrows():
            rows.append((T["row_step"].format(int(r.step)), r["diff"], r.ci_lo, r.ci_hi, "pre",
                         T["wg_n"].format(T["wg"][vs])))
    style = {"pre": (BLUE, "o", T["pre"]), "conf": (AQUA, "D", T["conf"])}

    # y positions: groups separated by a header line and a gap
    ypos, headers, y = [], [], 0.0
    sections = [(T["sec"][0], 0, 12), (T["sec"][1], 12, 18)]
    for title, a, b in sections:
        y -= 0.6
        headers.append((y, title, True))
        prev = None
        for k in range(a, b):
            if rows[k][5] != prev:
                y -= 1.0
                headers.append((y, rows[k][5], False))
                prev = rows[k][5]
            y -= 1.0
            ypos.append(y)
        y -= 0.4
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.axvline(0, color=INK2, lw=0.9, zorder=1)
    seen = set()
    for (lab, est, lo, hi, st, grp), yy in zip(rows, ypos):
        col, mk, name = style[st]
        ax.errorbar(est, yy, xerr=[[est - lo], [hi - est]], fmt=mk, ms=5.5, color=col, mfc=col, mec=col,
                    elinewidth=1.6, capsize=0, zorder=3, label=name if st not in seen else None)
        seen.add(st)
        ax.text(-0.335, yy, lab, fontsize=7, color=INK2, va="center", ha="left")
    conf_y = [yy for r, yy in zip(rows, ypos) if r[4] == "conf"]
    ax.plot([-0.05, -0.05], [min(conf_y) - 0.45, max(conf_y) + 0.45], color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.text(-0.058, float(np.mean(conf_y)), T["margin"], fontsize=6.3, color=MUTED, ha="right", va="center")
    for yy, text, major in headers:
        ax.text(-0.345, yy, text, fontsize=8 if major else 7.3, color=INK if major else INK2,
                fontweight="bold" if major else "normal", va="center", ha="left")
    ax.set_xlim(-0.35, 0.16)
    ax.set_xticks([-0.2, -0.1, 0, 0.1])
    ax.set_ylim(y - 0.2, 0.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(T["diff3"])
    ax.legend(loc="upper right", fontsize=7.3, labelcolor=INK2, handletextpad=0.3)
    save(fig, STEMS[3])


def fig4():
    """Exploratory: every reader x depth (linear probe), probe − rung 5, no post hoc selection."""
    readers = [("0.5B", "0.5B"), ("1.5B", "1.5B"), ("3B", "3B"), ("7B-4bit", T["r7b"])]
    layers = [("L0", "25%"), ("L1", "50%"), ("L2", "75%"), ("L3", "100%")]
    grids = []
    for j in range(3):
        g = np.zeros((4, 4)); ex = np.zeros((4, 4), bool)
        for i, (k, _) in enumerate(readers):
            d = read(f"explore_{k}_diff.csv")
            for c, (lay, _) in enumerate(layers):
                r = d[(d.method == f"linear_{lay}_pcafull") & (d.vs == "baseline5") & (d.step == j)]
                assert len(r) == 1, (k, lay, j)
                r = r.iloc[0]
                g[i, c] = r.auroc_diff
                ex[i, c] = (r.ci_hi < 0) or (r.ci_lo > 0)
        grids.append((g, ex))
    lim = 0.25
    allv = np.concatenate([g.ravel() for g, _ in grids])
    assert allv.min() >= -lim and allv.max() <= lim, (allv.min(), allv.max())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for j, (ax, (g, ex)) in enumerate(zip(axes, grids)):
        im = ax.pcolormesh(np.arange(-0.5, 4), np.arange(-0.5, 4), g, cmap=DIVERGE, norm=norm,
                           edgecolors="white", linewidth=1.5)
        for i in range(4):
            for c in range(4):
                v = g[i, c]
                txt = (f"{v:+.3f}" if abs(v) < 0.01 else f"{v:+.2f}").replace("-", "−") + ("†" if ex[i, c] else "")
                ax.text(c, i, txt, ha="center", va="center", fontsize=6.8,
                        color="white" if abs(v) > 0.14 else INK)
        ax.set_xticks(range(4), [l for _, l in layers])
        ax.set_yticks(range(4), [l for _, l in readers])
        ax.tick_params(length=0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlim(-0.5, 3.5); ax.set_ylim(3.5, -0.5)
        ax.set_title(T["step"].format(j), fontsize=8.5, color=INK, loc="left")
    axes[1].set_xlabel(T["depth"])
    axes[0].set_ylabel(T["reader"])
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(T["cbar"], color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK2, length=0)
    fig.suptitle(T["explore"], x=0.125, ha="left", fontsize=7.5, color=MUTED, y=1.02)
    save(fig, STEMS[4])


if __name__ == "__main__":
    lang = sys.argv[sys.argv.index("--lang") + 1] if "--lang" in sys.argv else "en"
    set_style(lang)
    os.makedirs(OUT, exist_ok=True)
    existing = [s + T["suffix"] for s in STEMS.values() for e in ("pdf", "png")
                if os.path.exists(os.path.join(OUT, f"{s}{T['suffix']}.{e}"))]
    if existing and "--overwrite" not in sys.argv:
        sys.exit(f"outputs exist: {existing}; pass --overwrite to redraw")
    if lang == "en":
        for s in OLD_STEMS:
            for e in ("pdf", "png"):
                p = os.path.join(OUT, f"{s}.{e}")
                if os.path.exists(p):
                    os.remove(p)
                    print("removed old", p)
    fig2()
    fig3()
    fig4()
