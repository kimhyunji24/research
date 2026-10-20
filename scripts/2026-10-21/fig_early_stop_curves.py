"""B3 (docs/review_b_plan_2026-10-21.md): savings vs. lost successes for each early-stopping rule, drawn only from
results/early_stop_3b_curves.csv and results/early_stop_3b_fold.csv (no new computation).

    /root/figenv/bin/python scripts/2026-10-21/fig_early_stop_curves.py [--lang en|ko]
Writes figures/figA_early_stop_curves{,_ko}.pdf/png; stops if they exist.
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STYLE = {"cap": ("#a8a7a0", "--"), "loop": ("#6b6a66", ":"), "baseline4t": ("#1baf7a", "-"),
         "baseline5t": ("#eb6834", "-"), "probe": ("#2a78d6", "-")}
TEXT = {
    "en": dict(font=["DejaVu Sans"], suffix="", x="Lost successes (share of successful games stopped)",
               y="Saving (share of steps not run)", zoom="Zoom: loss 0-10%", full="All thresholds",
               fold="threshold chosen on other folds (target loss 5%)",
               names={"cap": "Step cap", "loop": "Loop detector", "baseline4t": "Rung 4 score",
                      "baseline5t": "Rung 5 score", "probe": "Probe (75% depth)"}),
    "ko": dict(font=["NanumGothic", "DejaVu Sans"], suffix="_ko", x="성공 손실(멈춘 성공 게임 비율)",
               y="절감률(실행하지 않은 step 비율)", zoom="확대: 손실 0-10%", full="모든 문턱",
               fold="다른 fold에서 고른 문턱(목표 손실 5%)",
               names={"cap": "step 상한", "loop": "루프 감지", "baseline4t": "4단 점수",
                      "baseline5t": "5단 점수", "probe": "프로브(75% 깊이)"}),
}
ap = argparse.ArgumentParser(); ap.add_argument("--lang", default="en", choices=["en", "ko"]); a = ap.parse_args()
T = TEXT[a.lang]
stem = os.path.join(ROOT, "figures", f"figA_early_stop_curves{T['suffix']}")
for ext in ("pdf", "png"):
    if os.path.exists(f"{stem}.{ext}"): raise SystemExit(f"output exists: {stem}.{ext}")
cur = pd.read_csv(os.path.join(ROOT, "results", "early_stop_3b_curves.csv"))
fold = pd.read_csv(os.path.join(ROOT, "results", "early_stop_3b_fold.csv"))
plt.rcParams.update({"font.family": T["font"], "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
for ax, (xmax, title) in zip(axes, ((1.0, T["full"]), (0.10, T["zoom"]))):
    for m, (col, ls) in STYLE.items():
        c = cur[cur.method == m].sort_values(["loss", "saving"])
        ax.step(c.loss, c.saving, where="post", color=col, ls=ls, lw=1.4, label=T["names"][m])
        f = fold[fold.method == m]
        ax.scatter(f.test_loss, f.test_saving, color=col, s=22, zorder=3, edgecolor="white", linewidth=0.6)
    ax.set_xlim(-0.003 if xmax < 1 else -0.02, xmax); ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left"); ax.set_xlabel(T["x"])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
axes[0].set_ylabel(T["y"])
axes[0].scatter([], [], color=MUTED, s=22, label=T["fold"])
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.3, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{stem}.{ext}", bbox_inches="tight", facecolor="white",
                metadata={"CreationDate": None} if ext == "pdf" else None)
print("saved", stem)
