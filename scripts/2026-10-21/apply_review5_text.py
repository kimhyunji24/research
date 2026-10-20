"""Round-5 review, text fixes (no new analysis): weaker wording of the training-size result ("no detectable change",
CI in the sentence, the same-row first-250 change, curves not levelled off), abstract numbers with the selection caveat,
two Table A8 rows for the 2026-10-21 analyses, probe coverage not distinguishable from 95% or 80%, Table A6 -> A9 in
Section 4.1, the short-version wording, Table 8 "Not resolved", and a limitation line on larger training sets.
Every anchor must occur exactly once; files are written only after all anchors matched.

    python3 scripts/2026-10-21/apply_review5_text.py
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
    s = s.replace("%", r"\%").replace("Section 4.8", r"Section~\ref{sec:extra}").replace("Table A9", r"Table~\ref{tab:a9}").replace("Table A6", r"Table~\ref{tab:a6}")
    return num(s)


def ko_tex(s):
    s = s.replace("%", r"\%").replace(" ~ ", r" \textasciitilde{} ").replace("4.8절", r"\ref{sec:extra}절")
    s = s.replace("표 A9", r"표~\ref{tab:a9}").replace("표 A6", r"표~\ref{tab:a6}")
    return num(s)


# ---------------------------------------------------------------- shared (English tables in both languages)
SUM_MD = [("all CIs include zero | No |", "all CIs include zero | Not resolved (CI of the change −0.038 to +0.022) |")]
SUM_TEX = [(r"all CIs include zero & No \\", r"all CIs include zero & Not resolved (CI of the change $-0.038$ to $+0.022$) \\")]
A8_ROWS = [
    ("Additional checks",  # no section reference: Table A8 is also printed in the short versions, which have no 4.8
     "Six exploratory checks (training-set size, low false-positive region, split unit, interval coverage, action-token position, text encoder) planned after the pre-registered, confirmatory and exploratory results had been read",
     "2026-10-21, plan fixed before each check was run", "Exploratory only; no verdict changed"),
    ("Training-size sweep",
     "Sweep on all 500 episodes reuses the confirmatory hold-out for training and evaluation; planned after the first-250 learning curve and the hold-out results had been read",
     "2026-10-21, plan fixed before the run", "Exploratory only; the confirmatory verdict is unchanged"),
]
A8_LAST_MD = "| Second prompt (ReAct) | Methods of the secondary success-rate comparison (Wilson CI, game-level bootstrap, exact McNemar) were not in the plan | 2026-10-20, fixed in the script before running | Secondary result only |\n"
A8_MD = A8_LAST_MD + "".join("| " + " | ".join(r) + " |\n" for r in A8_ROWS)
A8_LAST_TEX = "Second prompt (ReAct) & Methods of the secondary success-rate comparison (Wilson CI, game-level bootstrap, exact McNemar) were not in the plan & 2026-10-20, fixed in the script before running & Secondary result only \\\\\n"
A8_TEX = A8_LAST_TEX + "".join("\\addlinespace\n" + " & ".join(en_tex(c) for c in r) + " \\\\\n" for r in A8_ROWS)

# ---------------------------------------------------------------- English
EN = [
    ("At the best exploratory depth the differences on the same games were −0.034 to −0.011.",
     "At the best exploratory depth the differences on the same games were −0.034 to −0.011, and +0.006 (−0.024 to +0.037) on all 500 games with about 400 training episodes; because this depth was chosen on the first 250 games, these values favor the probe, and the only value free of that selection is the held-out half (+0.015)."),
    ("for the one configuration tested on held-out games (75% depth) the intervals exclude gains above about 0.10 AUROC,",
     "for the one configuration tested on held-out games (75% depth) the single hold-out's intervals exclude gains above about 0.10 AUROC,"),
    ("all cells in Appendix Table A6)", "all cells in Appendix Table A9)"),
    ("The probe minus rung 5 was +0.016, +0.005, +0.011, +0.009 and +0.006, all intervals including zero, and its change from about 50 to about 400 training episodes was −0.010 (−0.038 to +0.022). "
     "Within this range the number of training episodes therefore did not change the gap. Restricted to the last 250 episodes, the full-size run reproduced the confirmatory values exactly; restricted to the first 250, it gave −0.003 (−0.025 when training on the first 250 alone).",
     "The probe minus rung 5 was +0.016, +0.005, +0.011, +0.009 and +0.006, all intervals including zero. Its change from about 50 to about 400 training episodes was −0.010 with a 95% CI of −0.038 to +0.022, so we detected no change, but a narrowing of up to about 0.02 is not ruled out. "
     "On the same evaluation rows (the first 250 episodes), training on about 400 episodes of all 500 instead of about 200 of the first 250 moved the difference from −0.025 to −0.003, a change of +0.022 toward narrowing; both values lie well within their intervals. "
     "Restricted to the last 250 episodes, the full-size run reproduced the confirmatory values exactly. Both methods were still improving at about 400 training episodes (probe 0.651 to 0.707, rung 5 0.635 to 0.701), so the curves had not levelled off, and we do not extrapolate beyond this range."),
    ("For the probe the coverage is consistent with the nominal level (its Wilson interval includes 95%); for rung 5 it is below.",
     "For the probe the Wilson interval includes both 95% and 80%, so its coverage cannot be told apart from either the nominal level or the attempt-level coverage; for rung 5 it is below nominal."),
    ("and with fixed evaluation folds the gap to rung 5 did not change between about 50 and 400 training episodes (Section 4.8).",
     "and with fixed evaluation folds we detected no change in the gap to rung 5 between about 50 and 400 training episodes (change −0.010, 95% CI −0.038 to +0.022; Section 4.8), although both methods were still improving."),
    ("with the evaluation rows fixed, the gap did not change between about 50 and 400 training episodes (Section 4.8), so within this range more training data did not close or reverse it.",
     "with the evaluation rows fixed, we detected no change in the gap between about 50 and 400 training episodes (Section 4.8), but the interval allows a narrowing of up to about 0.02, and on the first 250 episodes the gap moved from −0.025 to −0.003 as training grew from about 200 to 400. Whether larger training sets would close or reverse it is untested."),
    ("(85–90%; below it for rung 5, consistent with it for the probe; Section 4.8).",
     "(85–90%; below nominal for rung 5, and for the probe not distinguishable from either 95% or 80%; Section 4.8)."),
    ("which need several continuations branched from the same prefix, that is, new collection.",
     "which need several continuations branched from the same prefix, that is, new collection. Training sets beyond about 400 episodes were not tried: both learning curves were still rising there, and more episodes need new collection."),
]
edit("docs/paper_draft_en.md", EN + SUM_MD + [(A8_LAST_MD, A8_MD)])
EN_TEX = [(en_tex(o), en_tex(n)) for o, n in EN]
EN_TEX[0] = (EN_TEX[0][0].replace("$-0.011$.", "$-0.011.$"), EN_TEX[0][1])  # the tex abstract had the period inside the math
edit("paper/main_en.tex", EN_TEX + SUM_TEX + [(A8_LAST_TEX, A8_TEX)])

# ---------------------------------------------------------------- Korean
KO = [
    ("탐색에서 가장 좋은 깊이의 차이는 같은 게임에서 −0.034 ~ −0.011이었다.",
     "탐색에서 가장 좋은 깊이의 차이는 같은 게임에서 −0.034 ~ −0.011이었고, 학습 episode 약 400개로 500개 게임 전체에서 보면 +0.006(−0.024 ~ +0.037)이었다. 이 깊이는 앞 250개 게임에서 골랐으므로 이 값들은 프로브에 유리하며, 그 선택에서 자유로운 값은 홀드아웃 절반의 +0.015뿐이다."),
    ("홀드아웃에서 시험한 한 설정(75% 깊이)의 신뢰구간은", "홀드아웃에서 시험한 한 설정(75% 깊이)의 단일 홀드아웃 신뢰구간은"),
    ("전체는 부록 표 A6)", "전체는 부록 표 A9)"),
    ("프로브 − 5단은 +0.016, +0.005, +0.011, +0.009, +0.006으로 신뢰구간이 모두 0을 포함했고, 약 50개에서 약 400개로 갈 때의 변화는 −0.010(−0.038 ~ +0.022)이었다. "
     "따라서 이 범위에서 학습 episode 수는 격차를 바꾸지 않았다. 전체 크기로 학습한 결과를 뒤 250개로 제한하면 확증 값이 그대로 재현되었고, 앞 250개로 제한하면 −0.003이었다(앞 250개만으로 학습하면 −0.025).",
     "프로브 − 5단은 +0.016, +0.005, +0.011, +0.009, +0.006으로 신뢰구간이 모두 0을 포함했다. 약 50개에서 약 400개로 갈 때의 변화는 −0.010이고 95% 신뢰구간이 −0.038 ~ +0.022라, 변화는 탐지되지 않았지만 약 0.02까지의 좁아짐은 배제하지 못한다. "
     "같은 평가 행(앞 250개)에서 학습을 약 200개(앞 250개)에서 약 400개(500개 전체)로 늘리면 차이가 −0.025에서 −0.003으로, 좁아지는 방향으로 +0.022 움직였다. 두 값 모두 구간 안의 변동이다. "
     "전체 크기로 학습한 결과를 뒤 250개로 제한하면 확증 값이 그대로 재현되었다. 약 400개에서도 두 방법 모두 계속 올라(프로브 0.651에서 0.707, 5단 0.635에서 0.701) 곡선이 평평해지지 않았으므로, 이 범위 너머로 외삽하지 않는다."),
    ("프로브의 포함률은 명목 수준과 맞지만(Wilson 구간이 95%를 포함) 5단은 명목보다 낮다.",
     "프로브의 Wilson 구간은 95%와 80%를 모두 포함해 명목 수준과도 시도 수준의 포함률과도 구별되지 않고, 5단은 명목보다 낮다."),
    ("평가 fold를 고정하면 학습 episode 약 50–400개 범위에서 5단과의 격차가 변하지 않았다(4.8절).",
     "평가 fold를 고정하면 학습 episode 약 50–400개 범위에서 5단과의 격차 변화가 탐지되지 않았다(변화 −0.010, 95% 신뢰구간 −0.038 ~ +0.022; 4.8절). 다만 두 방법 모두 아직 오르고 있었다."),
    ("평가 행을 고정하면 학습 episode 약 50–400개 범위에서 격차가 변하지 않았으므로(4.8절), 이 범위에서는 학습 데이터를 늘려도 격차가 메워지거나 뒤집히지 않았다.",
     "평가 행을 고정하면 학습 episode 약 50–400개 범위에서 격차 변화가 탐지되지 않았지만(4.8절), 구간은 약 0.02까지의 좁아짐을 허용하고, 앞 250개에서는 학습을 약 200개에서 400개로 늘리자 격차가 −0.025에서 −0.003으로 움직였다. 더 큰 학습 집합에서 격차가 메워지거나 뒤집히는지는 시험하지 않았다."),
    ("(85–90%; 5단은 명목보다 낮고 프로브는 명목과 맞음; 4.8절).", "(85–90%; 5단은 명목보다 낮고 프로브는 95%와도 80%와도 구별되지 않음; 4.8절)."),
    ("즉 새 수집이 필요해 추정하지 않았다.",
     "즉 새 수집이 필요해 추정하지 않았다. 약 400개를 넘는 학습 집합도 시도하지 않았다. 그 지점에서 두 학습 곡선이 아직 오르고 있었고, episode를 늘리려면 새 수집이 필요하다."),
]
edit("docs/paper_draft_ko.md", KO + SUM_MD + [(A8_LAST_MD, A8_MD)])
edit("paper/main_ko.tex", [(ko_tex(o), ko_tex(n)) for o, n in KO] + SUM_TEX + [(A8_LAST_TEX, A8_TEX)])

# ---------------------------------------------------------------- short versions
edit("paper/main_en_short.tex", [
    (r"for the one configuration tested on held-out games (75\% depth) the intervals exclude",
     r"for the one configuration tested on held-out games (75\% depth) the single hold-out's intervals exclude"),
    ("Exploratory checks on the first 250 episodes (each run once):", "Exploratory checks (first 250 episodes, plus one sweep on all 500; each run once):"),
    ("the probe minus rung 5 did not narrow as the training data grew,", "no narrowing of the probe minus rung 5 was detected as the training data grew,"),
    (r"; change $-0.010$, $-0.038$ to $+0.022$);",
     r"; change $-0.010$, 95\% CI $-0.038$ to $+0.022$, so a narrowing of up to about 0.02 is not ruled out; both methods were still improving);"),
    (r"(the probe's 90\% is consistent with nominal)", r"(the probe's 90\% cannot be told apart from 95\% or 80\%)"),
    ("and the gap did not change between about 50 and 400 training episodes.", "and no change in the gap was detected between about 50 and 400 training episodes."),
    ("with the evaluation rows fixed, the gap did not change between about 50 and 400 training episodes.",
     "with the evaluation rows fixed, no change in the gap was detected between about 50 and 400 training episodes, although the interval allows a narrowing of up to about 0.02 and both curves were still rising."),
    ("because they need new collection or a margin that was not registered.",
     "because they need new collection or a margin that was not registered. Training sets beyond about 400 episodes were not tried, although both learning curves were still rising."),
])
edit("paper/main_ko_short.tex", [
    (r"홀드아웃에서 시험한 한 설정(75\% 깊이)의 신뢰구간은", r"홀드아웃에서 시험한 한 설정(75\% 깊이)의 단일 홀드아웃 신뢰구간은"),
    ("앞 250개 episode의 탐색 점검(분석마다 1회)에서,", "탐색 점검(앞 250개 episode와 500개 전체 sweep 하나, 분석마다 1회)에서,"),
    (r"학습 데이터를 늘려도 프로브 $-$ 5단은 줄지 않았고(", r"학습 데이터를 늘려도 프로브 $-$ 5단이 줄어드는 것은 탐지되지 않았고("),
    (r"변화 $-0.010$, $-0.038$ \textasciitilde{} $+0.022$)",
     r"변화 $-0.010$, 95\% 신뢰구간 $-0.038$ \textasciitilde{} $+0.022$로 약 0.02까지의 좁아짐은 배제하지 못하며 두 방법 모두 아직 오르는 중)"),
    (r"(프로브의 90\%는 명목과 맞음)", r"(프로브의 90\%는 95\%와도 80\%와도 구별되지 않음)"),
    ("학습 episode 약 50--400개 범위에서 격차도 변하지 않았다.", "학습 episode 약 50--400개 범위에서 격차 변화도 탐지되지 않았다."),
    ("평가 행을 고정하면 학습 episode 약 50--400개 범위에서 격차가 변하지 않았다.",
     "평가 행을 고정하면 학습 episode 약 50--400개 범위에서 격차 변화가 탐지되지 않았지만, 구간은 약 0.02까지의 좁아짐을 허용하고 두 곡선 모두 아직 오르고 있었다."),
    ("새 수집이나 등록하지 않은 마진이 필요해 시도하지 않았다.",
     "새 수집이나 등록하지 않은 마진이 필요해 시도하지 않았다. 두 학습 곡선이 아직 오르고 있었지만 약 400개를 넘는 학습 집합은 시도하지 않았다."),
])

for p, t in STAGED.items():
    io.open(p, "w", encoding="utf-8", newline="\n").write(t)
print("written", len(STAGED), "files")
