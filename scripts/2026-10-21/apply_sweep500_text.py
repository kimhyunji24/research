"""Add the 500-episode training-size sweep (docs/sweep500_plan_2026-10-21.md, results/sweep500_*.csv) to the manuscripts:
4.8 training-size paragraph, Tables 8 and 9, discussion (three explanations, complementary reading) and the short versions.
Every anchor must occur exactly once; files are written only after all anchors matched.

    python3 scripts/2026-10-21/apply_sweep500_text.py
"""
import io, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGED = {}


def edit(path, pairs):
    p = os.path.join(ROOT, path)
    t = io.open(p, encoding="utf-8").read()
    for old, new in pairs:
        n = t.count(old)
        assert n == 1, f"{path}: anchor found {n} times: {old[:80]!r}"
        t = t.replace(old, new)
    STAGED[p] = t
    print(f"{path}: {len(pairs)} edits")


def num(s):
    s = re.sub(r"(?<![\w.$])([−+])(\d+\.\d+)", lambda m: f"${'-' if m.group(1) == '−' else '+'}{m.group(2)}$", s)
    s = s.replace(" − ", " $-$ ")
    return re.sub(r"(\d)–(\d)", r"\1--\2", s)


def en_tex(s):
    s = s.replace("%", r"\%").replace("Section 4.8", r"Section~\ref{sec:extra}").replace("Table 9", r"Table~\ref{tab:extra}")
    return num(s)


def ko_tex(s):
    s = s.replace("%", r"\%").replace(" ~ ", r" \textasciitilde{} ").replace("4.8절", r"\ref{sec:extra}절").replace("표 9", r"표~\ref{tab:extra}")
    return num(s)


# ---------------------------------------------------------------- shared tables
SUM_OLD = ("| Training-set size | Does the gap to rung 5 close with more training data? | 75% probe − rung 5 from −0.002 (about 50 training episodes) to −0.025 (about 200); all CIs include zero | No; point estimates widen |")
SUM_NEW = ("| Training-set size | Does the gap to rung 5 close with more training data? | 75% probe − rung 5, first 250: −0.002 (about 50 training episodes) to −0.025 (about 200); all 500 with fixed evaluation folds: +0.016 (about 50) to +0.006 (about 400), change −0.010 (−0.038, 0.022); all CIs include zero | No |")


def sum_tex(s):  # row of the summary table as SUM_TEX in apply_review4_text.py built it
    cells = [c.strip() for c in s.strip("|").split("|")]
    return " & ".join(en_tex(c) for c in cells) + r" \\"


ROWS_MD_OLD = "| Training data 100% | 50% probe vs. rung 5 | 0.594 | 0.752 | −0.158 (−0.216, −0.090) |\n"
ROWS_MD_NEW = ROWS_MD_OLD + ("| All 500 episodes, training about 50 | 75% probe vs. rung 5 | 0.651 | 0.635 | +0.016 (−0.018, 0.049) |\n"
                             "| All 500 episodes, training about 400 | 75% probe vs. rung 5 | 0.707 | 0.701 | +0.006 (−0.024, 0.037) |\n")
ROWS_TEX_OLD = "Training data 100\\% & 50\\% probe vs.\\ rung 5 & 0.594 & 0.752 & $-0.158$ ($-0.216$, $-0.090$) \\\\\n"
ROWS_TEX_NEW = ROWS_TEX_OLD + ("All 500 episodes, training about 50 & 75\\% probe vs.\\ rung 5 & 0.651 & 0.635 & $+0.016$ ($-0.018$, 0.049) \\\\\n"
                               "All 500 episodes, training about 400 & 75\\% probe vs.\\ rung 5 & 0.707 & 0.701 & $+0.006$ ($-0.024$, 0.037) \\\\\n")
CAP_MD_OLD = "| Table 9 (exploratory). Additional checks, first 250 episodes, mean over steps 0–2"
CAP_MD_NEW = "| Table 9 (exploratory). Additional checks, first 250 episodes unless stated, mean over steps 0–2"
CAP_TEX_OLD = r"\caption{Exploratory: additional checks on the first 250 episodes (mean over steps 0--2; AUROC unless stated).}"
CAP_TEX_NEW = r"\caption{Exploratory: additional checks on the first 250 episodes unless stated (mean over steps 0--2; AUROC unless stated).}"

# ---------------------------------------------------------------- English
EN = [
    ("(exploratory; plan fixed before the run, each analysis run once; no hold-out data).",
     "(exploratory; plan fixed before the run, each analysis run once; no hold-out data except in the 500-episode training-size sweep below)."),
    ("The 50% probe stayed below rung 5 at every size (−0.104 to −0.158; Table 9).",
     "The 50% probe stayed below rung 5 at every size (−0.104 to −0.158; Table 9). "
     "To compare sizes on the same evaluation rows, we repeated the sweep on all 500 episodes with the evaluation folds fixed and 12.5 to 100% of each training fold (about 50 to 400 episodes; exploratory, planned after the results above and using the hold-out). "
     "The probe minus rung 5 was +0.016, +0.005, +0.011, +0.009 and +0.006, all intervals including zero, and its change from about 50 to about 400 training episodes was −0.010 (−0.038 to +0.022). "
     "Within this range the number of training episodes therefore did not change the gap. Restricted to the last 250 episodes, the full-size run reproduced the confirmatory values exactly; restricted to the first 250, it gave −0.003 (−0.025 when training on the first 250 alone). "
     "The 50% probe stayed below rung 5 at every size here as well (−0.042 to −0.057)."),
    ("We see three explanations, none tested directly.", "We see three explanations; only the third was examined directly, and only exploratorily."),
    ("and between about 50 and 200 training episodes the gap to rung 5 did not narrow (Section 4.8).",
     "and with fixed evaluation folds the gap to rung 5 did not change between about 50 and 400 training episodes (Section 4.8)."),
    ("between about 50 and 200 training episodes the gap did not narrow (its point estimates widened; Section 4.8), and the hold-out, trained on about 400, cannot be compared directly because its evaluation set differs.",
     "with the evaluation rows fixed, the gap did not change between about 50 and 400 training episodes (Section 4.8), so within this range more training data did not close or reverse it."),
]
edit("docs/paper_draft_en.md", EN + [(SUM_OLD, SUM_NEW), (ROWS_MD_OLD, ROWS_MD_NEW), (CAP_MD_OLD, CAP_MD_NEW)])
edit("paper/main_en.tex", [(en_tex(o), en_tex(n)) for o, n in EN] + [(sum_tex(SUM_OLD), sum_tex(SUM_NEW)), (ROWS_TEX_OLD, ROWS_TEX_NEW), (CAP_TEX_OLD, CAP_TEX_NEW)])

# ---------------------------------------------------------------- Korean
KO = [
    ("(탐색; 계획은 실행 전 고정, 분석마다 1회 실행; 홀드아웃 데이터는 쓰지 않음).",
     "(탐색; 계획은 실행 전 고정, 분석마다 1회 실행; 아래 500개 sweep 외에는 홀드아웃 데이터를 쓰지 않음)."),
    ("50% 프로브는 모든 크기에서 5단보다 낮았다(−0.104 ~ −0.158; 표 9).",
     "50% 프로브는 모든 크기에서 5단보다 낮았다(−0.104 ~ −0.158; 표 9). "
     "같은 평가 행에서 크기를 비교하려고, 500개 episode 전체에서 평가 fold를 고정하고 각 학습 fold의 12.5–100%(약 50–400개 episode)로 sweep을 다시 했다(탐색; 위 결과를 본 뒤 계획했고 홀드아웃을 씀). "
     "프로브 − 5단은 +0.016, +0.005, +0.011, +0.009, +0.006으로 신뢰구간이 모두 0을 포함했고, 약 50개에서 약 400개로 갈 때의 변화는 −0.010(−0.038 ~ +0.022)이었다. "
     "따라서 이 범위에서 학습 episode 수는 격차를 바꾸지 않았다. 전체 크기로 학습한 결과를 뒤 250개로 제한하면 확증 값이 그대로 재현되었고, 앞 250개로 제한하면 −0.003이었다(앞 250개만으로 학습하면 −0.025). "
     "50% 프로브는 여기서도 모든 크기에서 5단보다 낮았다(−0.042 ~ −0.057)."),
    ("세 가지 설명이 가능하며 어느 것도 직접 검정하지 않았다.", "세 가지 설명이 가능하며, 셋째만 탐색적으로 직접 살펴보았다."),
    ("학습 episode 약 50–200개 범위에서 5단과의 격차는 줄지 않았다(4.8절).",
     "평가 fold를 고정하면 학습 episode 약 50–400개 범위에서 5단과의 격차가 변하지 않았다(4.8절)."),
    ("학습 episode 약 50–200개 범위에서는 격차가 줄지 않았고(점추정은 오히려 벌어짐; 4.8절), 약 400개로 학습한 홀드아웃은 평가 집합이 달라 직접 비교할 수 없다.",
     "평가 행을 고정하면 학습 episode 약 50–400개 범위에서 격차가 변하지 않았으므로(4.8절), 이 범위에서는 학습 데이터를 늘려도 격차가 메워지거나 뒤집히지 않았다."),
]
edit("docs/paper_draft_ko.md", KO + [(SUM_OLD, SUM_NEW), (ROWS_MD_OLD, ROWS_MD_NEW), (CAP_MD_OLD, CAP_MD_NEW)])
edit("paper/main_ko.tex", [(ko_tex(o), ko_tex(n)) for o, n in KO] + [(sum_tex(SUM_OLD), sum_tex(SUM_NEW)), (ROWS_TEX_OLD, ROWS_TEX_NEW), (CAP_TEX_OLD, CAP_TEX_NEW)])

# ---------------------------------------------------------------- short versions
edit("paper/main_en_short.tex", [
    (r"the probe minus rung 5 widened rather than narrowed as the training data grew from 25\% to 100\% of each training fold ($-0.002$ to $-0.025$; all intervals include zero);",
     r"the probe minus rung 5 did not narrow as the training data grew, either on the first 250 episodes ($-0.002$ to $-0.025$ for about 50 to 200 training episodes) or on all 500 with fixed evaluation folds ($+0.016$ to $+0.006$ for about 50 to 400; change $-0.010$, $-0.038$ to $+0.022$);"),
    ("We see three explanations, none tested directly.", "We see three explanations; only the third was examined directly, and only exploratorily."),
    ("and five stronger probes, did not exceed the baseline either.",
     "and five stronger probes, did not exceed the baseline either, and the gap did not change between about 50 and 400 training episodes."),
    ("between about 50 and 200 training episodes the gap did not narrow, and the hold-out (about 400 training episodes) is not directly comparable.",
     "with the evaluation rows fixed, the gap did not change between about 50 and 400 training episodes."),
])
edit("paper/main_ko_short.tex", [
    (r"학습 데이터를 각 학습 fold의 25\%에서 100\%로 늘려도 프로브 $-$ 5단은 줄지 않고 점추정이 오히려 벌어졌으며($-0.002$ \textasciitilde{} $-0.025$, 신뢰구간 모두 0 포함)",
     r"학습 데이터를 늘려도 프로브 $-$ 5단은 줄지 않았고(앞 250개에서 학습 약 50--200개일 때 $-0.002$ \textasciitilde{} $-0.025$, 500개 전체에서 평가 fold를 고정하고 약 50--400개일 때 $+0.016$ \textasciitilde{} $+0.006$, 변화 $-0.010$, $-0.038$ \textasciitilde{} $+0.022$)"),
    ("세 가지 설명이 가능하며 어느 것도 직접 검정하지 않았다.", "세 가지 설명이 가능하며, 셋째만 탐색적으로 직접 살펴보았다."),
    ("더 강한 프로브 다섯 개도 baseline을 넘지 못했다.", "더 강한 프로브 다섯 개도 baseline을 넘지 못했고, 학습 episode 약 50--400개 범위에서 격차도 변하지 않았다."),
    ("학습 episode 약 50--200개 범위에서는 격차가 줄지 않았고, 약 400개로 학습한 홀드아웃은 직접 비교할 수 없다.",
     "평가 행을 고정하면 학습 episode 약 50--400개 범위에서 격차가 변하지 않았다."),
])

for p, t in STAGED.items():
    io.open(p, "w", encoding="utf-8", newline="\n").write(t)
print("written", len(STAGED), "files")
