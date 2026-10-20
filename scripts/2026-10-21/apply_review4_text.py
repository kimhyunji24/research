"""Round-4 review, text fixes A (no new analysis): learning-curve wording, training-episode counts, coverage by method with
a rough widening sensitivity, removal of "in response to review", abstract and introduction caveats, a summary table
before the numeric 4.8 table (now Table 9, with 50% rows), Table A9 with all 96 exploratory cells and CIs, post hoc
depth note, and a "not done" limitation. Applied to the md drafts, the full tex and the short tex (English, Korean).
Every anchor must occur exactly once; files are written only after all anchors matched.

    python3 scripts/2026-10-21/apply_review4_text.py
"""
import csv, io, os, re

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
    s = s.replace("%", r"\%").replace(" [3]", r"~\cite{ruan2026doomed}").replace("z ≈ 1.555", r"$z \approx 1.555$")
    s = s.replace("permutation p-values", "permutation $p$-values")
    for a, b in (("Table A9", r"Table~\ref{tab:a9}"), ("Table A6", r"Table~\ref{tab:a6}"), ("Table 9", r"Table~\ref{tab:extra}"),
                 ("Table 8", r"Table~\ref{tab:extrasum}"), ("Section 4.8", r"Section~\ref{sec:extra}")):
        s = s.replace(a, b)
    return num(s)


def ko_tex(s):
    s = s.replace("%", r"\%").replace(" [3]", r"~\cite{ruan2026doomed}").replace("z ≈ 1.555", r"$z \approx 1.555$")
    s = s.replace(" ~ ", r" \textasciitilde{} ").replace("순열 p를", "순열 $p$를")
    for a, b in (("표 A9", r"표~\ref{tab:a9}"), ("표 A6", r"표~\ref{tab:a6}"), ("표 9", r"표~\ref{tab:extra}"),
                 ("4.8절", r"\ref{sec:extra}절")):
        s = s.replace(a, b)
    return num(s)


# ---------------------------------------------------------------- tables
SUM = [  # check, question, result, changed?
    ("Training-set size", "Does the gap to rung 5 close with more training data?",
     "75% probe − rung 5 from −0.002 (about 50 training episodes) to −0.025 (about 200); all CIs include zero", "No; point estimates widen"),
    ("Low false-positive region", "Is the probe better where stopping matters?",
     "Detection at 10% false positives, probe − rung 5: −0.139 (−0.273, 0.075)", "No"),
    ("Split unit", "Does splitting by game inflate the real methods?",
     "Game − scene split: −0.006 (probe), −0.027 (rung 5)", "No"),
    ("Interval coverage", "Are task-level intervals too narrow?",
     "Shuffled-label coverage 90.0% (probe), 85.0% (rung 5), 88.3% (difference)", "No; pre-registered step 1 only narrowly below zero after widening"),
    ("Reading position", "Does the last action token change the result?",
     "75% action token − rung 5: −0.019 (−0.085, 0.046)", "No"),
    ("Text encoder", "Is there a stronger text baseline?",
     "bge embedding + rung 4 − rung 5: −0.115 (−0.187, −0.052)", "Not answered; this encoder was weaker than TF-IDF"),
]
SUM_MD = ("| Table 8 (exploratory). Additional checks: question, result and whether the conclusion changes | Question | Result (mean over steps 0–2, 95% CI) | Conclusion changed? |\n"
          "|---|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in SUM))
SUM_TEX = "\n".join([r"\begin{table}[t]", r"\centering\small",
                     r"\caption{Exploratory: additional checks, the question each answers, and whether the conclusion changes.}\label{tab:extrasum}",
                     r"\begin{tabularx}{\linewidth}{@{}lLLL@{}}", r"\toprule",
                     r"Check & Question & Result (mean over steps 0--2, 95\% CI) & Conclusion changed? \\", r"\midrule"]
                    + [" & ".join(en_tex(x) for x in r) + r" \\" for r in SUM] + [r"\bottomrule", r"\end{tabularx}", r"\end{table}"])

P50 = [("25%", "0.580", "0.684", "−0.104", "−0.152", "−0.055"), ("50%", "0.582", "0.714", "−0.132", "−0.183", "−0.076"),
       ("75%", "0.600", "0.733", "−0.133", "−0.185", "−0.075"), ("100%", "0.594", "0.752", "−0.158", "−0.216", "−0.090")]
P50_MD = "".join(f"| Training data {f} | 50% probe vs. rung 5 | {a} | {b} | {d} ({lo}, {hi}) |\n" for f, a, b, d, lo, hi in P50)
P50_TEX = "".join(f"Training data {f.replace('%', chr(92) + '%')} & 50\\% probe vs.\\ rung 5 & {a} & {b} & ${d.replace('−', '-')}$ (${lo.replace('−', '-')}$, ${hi.replace('−', '-')}$) \\\\\n"
                  for f, a, b, d, lo, hi in P50)
ROW100_MD = "| Training data 100% | 75% probe vs. rung 5 | 0.727 | 0.752 | −0.025 (−0.077, 0.029) |\n"
ROW100_TEX = "Training data 100\\% & 75\\% probe vs.\\ rung 5 & 0.727 & 0.752 & $-0.025$ ($-0.077$, 0.029) \\\\\n"


def fmt(v, sign):
    r = round(v, 3)
    s = f"{abs(r):.3f}"
    if r < 0: return "−" + s
    return ("+" + s) if sign and r > 0 else s


READERS = [("0.5B", "0.5B reader"), ("1.5B", "1.5B reader"), ("3B", "3B reader (= actor)"), ("7B-4bit", "7B reader (4-bit)")]
A9 = []
for key, name in READERS:
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", f"explore_{key}_diff.csv"))))
    assert len(rows) == 24
    for kind, kname in (("linear", "Linear"), ("mlp", "MLP")):
        for L, depth in enumerate(("25%", "50%", "75%", "100%")):
            cells = []
            for step in "012":
                r = [x for x in rows if x["method"] == f"{kind}_L{L}_pcafull" and x["step"] == step]
                assert len(r) == 1
                r = r[0]
                cells.append(f"{fmt(float(r['auroc_diff']), True)} ({fmt(float(r['ci_lo']), False)}, {fmt(float(r['ci_hi']), False)})")
            A9.append((name, kname, depth, cells))
assert len(A9) == 32
pre = [c for n, k, d, c in A9 if n.startswith("3B") and k == "Linear" and d == "50%"][0]
assert pre == ["−0.181 (−0.276, −0.069)", "−0.133 (−0.230, −0.031)", "−0.159 (−0.240, −0.073)"], pre


def a9_md(title, foot):
    return (f"| {title} | Probe | Depth | step 0 | step 1 | step 2 |\n|---|---|---|---|---|---|\n"
            + "".join(f"| {n} | {k} | {d} | {c[0]} | {c[1]} | {c[2]} |\n" for n, k, d, c in A9) + "\n" + foot + "\n")


def a9_tex(foot):
    rows = [f"{n} & {k} & {d.replace('%', chr(92) + '%')} & " + " & ".join(num(x) for x in c) + r" \\" for n, k, d, c in A9]
    return "\n".join([r"\begin{table}[h]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
                      r"\caption{Exploratory: probe $-$ rung 5 AUROC difference (95\% CI) for all 96 cells (4 readers $\times$ 2 probe types $\times$ 4 depths $\times$ 3 steps), first 250 episodes.}\label{tab:a9}",
                      r"\begin{tabular}{@{}lllccc@{}}", r"\toprule", r"Reader & Probe & Depth & Step 0 & Step 1 & Step 2 \\", r"\midrule"]
                     + rows + [r"\bottomrule", r"\end{tabular}", r"\par\smallskip\raggedright\footnotesize", foot,
                               "% Source: results/explore_{0.5B,1.5B,3B,7B-4bit}_diff.csv", r"\end{table}"])


A9_TITLE = "Table A9 (exploratory). Probe − rung 5 AUROC difference (95% CI) for all 96 cells, first 250 episodes"
A9_FOOT_EN = "Scene-level cluster bootstrap, 1,000 resamples; *n* = 250 (194 failures) per step. The 7B reader used 4-bit weights. The 3B reader at 50% depth (linear) is the pre-registered cell. <!-- Source: `results/explore_{0.5B,1.5B,3B,7B-4bit}_diff.csv` -->"
A9_FOOT_EN_TEX = r"Scene-level cluster bootstrap, 1,000 resamples; $n = 250$ (194 failures) per step. The 7B reader used 4-bit weights. The 3B reader at 50\% depth (linear) is the pre-registered cell."
A9_FOOT_KO = "장면 단위 cluster bootstrap 1,000회. step당 *n* = 250(실패 194). 7B reader는 4비트 가중치를 썼다. 3B reader의 50% 깊이 선형 칸이 사전 등록 조건이다. <!-- 출처: `results/explore_{0.5B,1.5B,3B,7B-4bit}_diff.csv` -->"
A9_FOOT_KO_TEX = r"장면 단위 cluster bootstrap 1,000회. step당 $n = 250$(실패 194). 7B reader는 4비트 가중치를 썼다. 3B reader의 50\% 깊이 선형 칸이 사전 등록 조건이다."

# ---------------------------------------------------------------- English (md text; tex = en_tex unless given)
EN = [
    ("(all 95% confidence intervals below zero),",
     "(all 95% confidence intervals below zero; the second action only narrowly after a rough correction for interval coverage),"),
    ("The game explained 69.5% of the outcome variance (95% confidence interval 61.3–76.5%).",
     "In this sample, whose games were stratified by greedy outcome, the game explained 69.5% of the outcome variance (95% confidence interval 61.3–76.5%)."),
    ("and the held-out intervals exclude gains above about 0.10 AUROC,",
     "and for the one configuration tested on held-out games (75% depth) the intervals exclude gains above about 0.10 AUROC,"),
    ("Splitting by game inflated AUROC to 0.97 for a synthetic feature that contained only scene information.",
     "Splitting by game inflated AUROC to 0.97 for a synthetic feature that contained only scene information; with the real probe and rung 5 the change was small (−0.006 and −0.027)."),
    ("With about 250 training episodes, adding 2,048 dimensions", "With about 200 training episodes per fold, adding 2,048 dimensions"),
    ("In response to review, we ran six further checks", "We ran six further checks"),
    ("The probe minus rung 5 was −0.002, −0.011, −0.021 and −0.025, all intervals including zero. The gap did not shrink as the training data grew, which suggests that within this range the probe was not limited by the number of training episodes; we did not test larger training sets. The 50% probe stayed below rung 5 at every size (−0.104 to −0.158).",
     "The probe minus rung 5 was −0.002, −0.011, −0.021 and −0.025, all intervals including zero: the point estimates widened rather than narrowed, because rung 5 gained more from the added data (0.684 to 0.752) than the probe (0.681 to 0.727). The confirmatory hold-out, the only result trained on about 400 episodes (all 500 episodes, five folds), went the other way (mean +0.015), but its evaluation set differs (rung 5 at step 0: 0.607 vs. 0.701), so the two cannot be compared. The 50% probe stayed below rung 5 at every size (−0.104 to −0.158; Table 9)."),
    ("rather than 95%. Task-level intervals may therefore also be somewhat too narrow; the 60 runs share the same episodes, so these coverage intervals are themselves approximate.",
     "rather than 95%. For the probe the coverage is consistent with the nominal level (its Wilson interval includes 95%); for rung 5 it is below. The 60 runs share the same episodes, so these coverage intervals are themselves approximate, and shuffled-label coverage is only a proxy for coverage under the real labels. As a rough sensitivity check, widening the upper half of the pre-registered intervals by a factor of 1.26 (88% coverage, z ≈ 1.555) left all three upper bounds below zero (−0.040, −0.004 and −0.051), step 1 only narrowly."),
    ("Third, with about 250 training episodes per round and 2,048 dimensions,",
     "Third, with about 200 training episodes per fold (about 400 for the hold-out) and 2,048 dimensions,"),
    ("and the gap to rung 5 did not shrink as the training data grew from about 50 to 200 episodes (Section 4.8).",
     "and between about 50 and 200 training episodes the gap to rung 5 did not narrow (Section 4.8)."),
    ("trained on about 250 episodes per round without hand-made features,", "trained on about 200 episodes per fold without hand-made features,"),
    ("and on the hold-out it was ahead at steps 0 and 1, though not significantly.",
     "and on the hold-out (about 400 training episodes) it was ahead at steps 0 and 1, though not significantly. The 75% depth was the best of the exploratory depths, chosen after seeing them; the pre-registered 50% cell was 0.13–0.18 below rung 5."),
    ("the gap did not shrink between about 50 and 200 training episodes (Section 4.8), and larger training sets were not tested.",
     "between about 50 and 200 training episodes the gap did not narrow (its point estimates widened; Section 4.8), and the hold-out, trained on about 400, cannot be compared directly because its evaluation set differs."),
    ("and task-level intervals were closer to but still below nominal (85–90%; Section 4.8).",
     "and task-level intervals were closer to nominal (85–90%; below it for rung 5, consistent with it for the probe; Section 4.8)."),
    ("was checked only in an exploratory analysis of the first 250 episodes.",
     "was checked only in an exploratory analysis of the first 250 episodes. Probes at 25% depth, nearer the input, were weaker (0.595 for the 3B reader; Table A6); a fine-tuned text model was not tried."),
    ("so they, like the attempt-level intervals, may be too narrow, and the variance",
     "so some of them, like the attempt-level intervals, may be too narrow (with a rough correction the pre-registered difference at step 1 is only just below zero), and the variance"),
    ("MLP cells and CIs: Appendix Table A6.", "Linear and MLP cells with CIs: Appendix Table A9."),
    ("Full results with CIs and permutation p-values are in the supplementary material.", "Differences with 95% CIs for all 96 cells are in Table A9."),
]
EN_NOTDONE = ("We did not add a second environment, because it needs new collection outside the registered plans; we did not run a formal equivalence analysis, because no equivalence margin was registered (the 0.05 non-inferiority margin was set after exploration); "
              "and we did not estimate Monte Carlo value targets (the success probability from each state), which need several continuations branched from the same prefix, that is, new collection.")
EN_INTRO_OLD_MD = "(Table 8). With the full training folds, all four methods reproduced their earlier values exactly."
EN_INTRO_NEW_MD = "(Tables 8 and 9). With the full training folds, all four methods (the 75% and 50% probes, rungs 4 and 5) reproduced their earlier values exactly.\n\n" + SUM_MD
EN_INTRO_OLD_TEX = r"(Table~\ref{tab:extra}). With the full training folds, all four methods reproduced their earlier values exactly."
EN_INTRO_NEW_TEX = (r"(Tables~\ref{tab:extrasum} and~\ref{tab:extra}). With the full training folds, all four methods (the 75\% and 50\% probes, rungs 4 and 5) reproduced their earlier values exactly."
                    + "\n\n" + SUM_TEX)

edit("docs/paper_draft_en.md", EN + [
    (EN_INTRO_OLD_MD, EN_INTRO_NEW_MD),
    ("| Table 8 (exploratory). Additional checks, first 250 episodes", "| Table 9 (exploratory). Additional checks, first 250 episodes"),
    (ROW100_MD, ROW100_MD + P50_MD),
    ("- **Policy shift.**", "- **Not done.** " + EN_NOTDONE + "\n- **Policy shift.**"),
    ("\n## References", "\n" + a9_md(A9_TITLE, A9_FOOT_EN) + "\n## References"),
])
edit("paper/main_en.tex", [(en_tex(o), en_tex(n)) for o, n in EN] + [
    (EN_INTRO_OLD_TEX, EN_INTRO_NEW_TEX),
    (ROW100_TEX, ROW100_TEX + P50_TEX),
    (r"\item \textbf{Policy shift.}", r"\item \textbf{Not done.} " + en_tex(EN_NOTDONE) + "\n" + r"\item \textbf{Policy shift.}"),
    ("\n\\end{document}", "\n" + a9_tex(A9_FOOT_EN_TEX) + "\n\n\\end{document}"),
])

# ---------------------------------------------------------------- Korean
KO = [
    ("(95% 신뢰구간 모두 0 아래), 새 250개 게임에서는",
     "(95% 신뢰구간 모두 0 아래, 구간 포함률을 대략 보정하면 두 번째 action은 간신히 0 아래), 새 250개 게임에서는"),
    ("게임은 결과 분산의 69.5%를 설명했다(95% 신뢰구간 61.3–76.5%).",
     "탐욕 결과로 층화해 고른 이 표본에서 게임은 결과 분산의 69.5%를 설명했다(95% 신뢰구간 61.3–76.5%)."),
    ("홀드아웃 신뢰구간은 약 0.10 AUROC를 넘는 이득을 배제하지만",
     "홀드아웃에서 시험한 한 설정(75% 깊이)의 신뢰구간은 약 0.10 AUROC를 넘는 이득을 배제하지만"),
    ("게임 단위로 나누면 장면 정보만 담은 합성 특징에서도 AUROC가 0.97까지 부풀었다.",
     "게임 단위로 나누면 장면 정보만 담은 합성 특징에서도 AUROC가 0.97까지 부풀었다. 실제 프로브와 5단에서는 변화가 작았다(−0.006, −0.027)."),
    ("학습 episode가 약 250개라 5단에 2,048차원을", "fold당 학습 episode가 약 200개라 5단에 2,048차원을"),
    ("리뷰에 따라 앞 250개 episode의", "앞 250개 episode의"),
    ("프로브 − 5단은 −0.002, −0.011, −0.021, −0.025였고 신뢰구간은 모두 0을 포함했다. 학습 데이터가 늘어도 격차가 줄지 않았으므로, 이 범위에서는 프로브가 학습 episode 수에 제한되지 않았음을 시사한다. 더 큰 학습 집합은 시험하지 않았다. 50% 프로브는 모든 크기에서 5단보다 낮았다(−0.104 ~ −0.158).",
     "프로브 − 5단은 −0.002, −0.011, −0.021, −0.025였고 신뢰구간은 모두 0을 포함했다. 점추정은 줄지 않고 오히려 벌어졌는데, 5단이 데이터에서 얻은 이득(0.684에서 0.752)이 프로브(0.681에서 0.727)보다 컸기 때문이다. 약 400개 episode로 학습한 유일한 결과인 확증 홀드아웃(500개 전체, 5-fold)은 반대 방향(평균 +0.015)이었지만 평가 집합이 달라(step 0의 5단 0.607 대 0.701) 둘을 비교할 수 없다. 50% 프로브는 모든 크기에서 5단보다 낮았다(−0.104 ~ −0.158; 표 9)."),
    ("로 95%보다 낮았다. 따라서 과제 수준 구간도 다소 좁을 수 있다. 60회는 같은 episode를 공유하므로 이 포함률의 구간도 근사이다.",
     "로 95%보다 낮았다. 프로브의 포함률은 명목 수준과 맞지만(Wilson 구간이 95%를 포함) 5단은 명목보다 낮다. 60회는 같은 episode를 공유하므로 이 포함률의 구간도 근사이고, 라벨을 섞은 데이터의 포함률은 실제 라벨에서의 포함률을 대신하는 지표일 뿐이다. 대략의 민감도로 사전 등록 비교의 구간 위쪽 절반을 1.26배(포함률 88%, z ≈ 1.555) 넓혀도 세 상한은 모두 0 아래였고(−0.040, −0.004, −0.051), step 1은 간신히 0 아래였다."),
    ("셋째, 라운드당 학습 episode가 약 250개이고 차원이 2,048이라", "셋째, fold당 학습 episode가 약 200개(홀드아웃은 약 400개)이고 차원이 2,048이라"),
    ("학습 데이터를 약 50개에서 200개 episode로 늘려도 5단과의 격차는 줄지 않았다(4.8절).",
     "학습 episode 약 50–200개 범위에서 5단과의 격차는 줄지 않았다(4.8절)."),
    ("손으로 만든 특징 없이 라운드당 약 250개 episode로 학습한 프로브는", "손으로 만든 특징 없이 fold당 약 200개 episode로 학습한 프로브는"),
    ("홀드아웃의 step 0과 1에서는 유의하지 않지만 앞섰다.",
     "홀드아웃(학습 episode 약 400개)의 step 0과 1에서는 유의하지 않지만 앞섰다. 75% 깊이는 탐색한 깊이들을 본 뒤 고른 최선값이며, 사전 등록한 50% 칸은 5단보다 0.13–0.18 낮았다."),
    ("학습 episode 약 50–200개 범위에서는 격차가 줄지 않았고(4.8절), 더 큰 학습 집합은 시험하지 않았다.",
     "학습 episode 약 50–200개 범위에서는 격차가 줄지 않았고(점추정은 오히려 벌어짐; 4.8절), 약 400개로 학습한 홀드아웃은 평가 집합이 달라 직접 비교할 수 없다."),
    ("과제 수준 구간은 이보다 낫지만 여전히 명목보다 낮았다(85–90%; 4.8절).",
     "과제 수준 구간은 명목에 더 가까웠다(85–90%; 5단은 명목보다 낮고 프로브는 명목과 맞음; 4.8절)."),
    ("앞 250개 episode의 탐색 점검으로만 확인했다.",
     "앞 250개 episode의 탐색 점검으로만 확인했다. 입력에 가까운 25% 깊이의 프로브는 더 약했고(3B reader 0.595; 표 A6), 미세조정한 텍스트 모델은 시도하지 않았다."),
    ("시도 수준 구간처럼 좁을 수 있고,", "일부는 시도 수준 구간처럼 좁을 수 있고(대략 보정하면 사전 등록 차이의 step 1이 간신히 0 아래),"),
    ("MLP 칸과 CI: 부록 표 A6.", "선형·MLP 칸과 CI: 부록 표 A9."),
    ("CI와 순열 p를 포함한 전체 결과는 보조 자료에 있다.", "96칸 모두의 차이와 95% CI는 표 A9에 있다."),
]
KO_NOTDONE = ("두 번째 환경은 등록한 계획 밖의 새 수집이 필요해 넣지 않았다. 동등성 마진을 등록하지 않았으므로(0.05 비열등 마진은 탐색 뒤에 정함) 정식 동등성 분석은 하지 않았다. "
              "각 상태에서의 성공 확률(Monte Carlo value 타깃)은 같은 prefix에서 갈라진 여러 continuation, 즉 새 수집이 필요해 추정하지 않았다.")
KO_INTRO_OLD_MD = "(표 8). 학습 fold 전체로 학습하면 네 방법 모두 앞선 값을 그대로 재현했다."
KO_INTRO_NEW_MD = "(표 8, 9). 학습 fold 전체로 학습하면 네 방법(75%·50% 프로브, 4단, 5단) 모두 앞선 값을 그대로 재현했다.\n\n" + SUM_MD
KO_INTRO_OLD_TEX = r"(표~\ref{tab:extra}). 학습 fold 전체로 학습하면 네 방법 모두 앞선 값을 그대로 재현했다."
KO_INTRO_NEW_TEX = (r"(표~\ref{tab:extrasum}, \ref{tab:extra}). 학습 fold 전체로 학습하면 네 방법(75\%·50\% 프로브, 4단, 5단) 모두 앞선 값을 그대로 재현했다."
                    + "\n\n" + SUM_TEX)
edit("docs/paper_draft_ko.md", KO + [
    (KO_INTRO_OLD_MD, KO_INTRO_NEW_MD),
    ("| Table 8 (exploratory). Additional checks, first 250 episodes", "| Table 9 (exploratory). Additional checks, first 250 episodes"),
    (ROW100_MD, ROW100_MD + P50_MD),
    ("- **정책 차이.**", "- **하지 않은 것.** " + KO_NOTDONE + "\n- **정책 차이.**"),
    ("\n## 참고문헌", "\n" + a9_md(A9_TITLE, A9_FOOT_KO) + "\n## 참고문헌"),
])
edit("paper/main_ko.tex", [(ko_tex(o), ko_tex(n)) for o, n in KO] + [
    (KO_INTRO_OLD_TEX, KO_INTRO_NEW_TEX),
    (ROW100_TEX, ROW100_TEX + P50_TEX),
    (r"\item \textbf{정책 차이.}", r"\item \textbf{하지 않은 것.} " + ko_tex(KO_NOTDONE) + "\n" + r"\item \textbf{정책 차이.}"),
    ("\n\\end{document}", "\n" + a9_tex(A9_FOOT_KO_TEX) + "\n\n\\end{document}"),
])

# ---------------------------------------------------------------- short versions (tex text directly)
edit("paper/main_en_short.tex", [
    (r"(all 95\% confidence intervals below zero), and on 250 held-out games",
     r"(all 95\% confidence intervals below zero; the second action only narrowly after a rough correction for interval coverage), and on 250 held-out games"),
    (r"the held-out intervals exclude gains above about 0.10 AUROC,",
     r"for the one configuration tested on held-out games (75\% depth) the intervals exclude gains above about 0.10 AUROC,"),
    (r"in our sampled data the game explains 69.5\%", r"in our sampled data, whose games were stratified by greedy outcome, the game explains 69.5\%"),
    ("for a synthetic feature that encodes only the scene;",
     r"for a synthetic feature that encodes only the scene (with the real probe and rung 5 the change was small, $-0.006$ and $-0.027$);"),
    (r"the probe minus rung 5 did not shrink as the training data grew from 25\% to 100\% of each training fold ($-0.002$ to $-0.025$);",
     r"the probe minus rung 5 widened rather than narrowed as the training data grew from 25\% to 100\% of each training fold ($-0.002$ to $-0.025$; all intervals include zero);"),
    (r"On shuffled labels, task-level intervals covered the null value in 85--90\% of runs rather than 95\%.",
     r"On shuffled labels, task-level intervals covered the null value in 85--90\% of runs rather than 95\% (the probe's 90\% is consistent with nominal); widening the pre-registered intervals 1.26-fold kept all upper bounds below zero, step 1 only narrowly ($-0.004$)."),
    ("Third, with about 250 training episodes per round and 2,048 dimensions,", "Third, with about 200 training episodes per fold and 2,048 dimensions,"),
    ("trained on about 250 episodes per round, the probe came within", "trained on about 200 episodes per fold, the probe came within"),
    ("the gap did not shrink between about 50 and 200 training episodes, and larger training sets were not tested.",
     r"between about 50 and 200 training episodes the gap did not narrow, and the hold-out (about 400 training episodes) is not directly comparable. The 75\% depth was chosen after seeing the exploratory depths; the pre-registered 50\% cell was 0.13--0.18 below rung 5."),
    (r"(deviations in Appendix Table~\ref{tab:a8}).",
     r"(deviations in Appendix Table~\ref{tab:a8}). A second environment, a formal equivalence analysis and Monte Carlo value targets were not attempted, because they need new collection or a margin that was not registered."),
    ("\\input{floats_en/tab_a6}\n", "\\input{floats_en/tab_a6}\n\\input{floats_en/tab_a9}\n"),
])
edit("paper/main_ko_short.tex", [
    (r"(95\% 신뢰구간 모두 0 아래), 새 250개", r"(95\% 신뢰구간 모두 0 아래, 구간 포함률을 대략 보정하면 두 번째 action은 간신히 0 아래), 새 250개"),
    ("홀드아웃 신뢰구간은 약 0.10 AUROC를", r"홀드아웃에서 시험한 한 설정(75\% 깊이)의 신뢰구간은 약 0.10 AUROC를"),
    ("우리 샘플링 데이터에서는 게임이 결과 분산의", "탐욕 결과로 층화해 고른 우리 샘플링 데이터에서는 게임이 결과 분산의"),
    ("게임 단위로 나누면 장면 정보만 담은 합성 특징에서도 AUROC가 0.97까지 부풀었다.",
     r"게임 단위로 나누면 장면 정보만 담은 합성 특징에서도 AUROC가 0.97까지 부풀었다. 실제 프로브와 5단에서는 변화가 작았다($-0.006$, $-0.027$)."),
    (r"프로브 $-$ 5단은 줄지 않았고($-0.002$ \textasciitilde{} $-0.025$)",
     r"프로브 $-$ 5단은 줄지 않고 점추정이 오히려 벌어졌으며($-0.002$ \textasciitilde{} $-0.025$, 신뢰구간 모두 0 포함)"),
    (r"라벨을 섞은 데이터에서 과제 수준 구간은 귀무값을 95\%가 아니라 85--90\%만 포함했다.",
     r"라벨을 섞은 데이터에서 과제 수준 구간은 귀무값을 95\%가 아니라 85--90\%만 포함했다(프로브의 90\%는 명목과 맞음). 사전 등록 구간을 1.26배 넓혀도 상한은 모두 0 아래였지만 step 1은 간신히 그랬다($-0.004$)."),
    ("셋째, 라운드당 학습 episode가 약 250개이고", "셋째, fold당 학습 episode가 약 200개이고"),
    ("라운드당 약 250개 episode로 학습한 프로브는", "fold당 약 200개 episode로 학습한 프로브는"),
    ("학습 episode 약 50--200개 범위에서는 격차가 줄지 않았고, 더 큰 학습 집합은 시험하지 않았다.",
     r"학습 episode 약 50--200개 범위에서는 격차가 줄지 않았고, 약 400개로 학습한 홀드아웃은 직접 비교할 수 없다. 75\% 깊이는 탐색한 깊이들을 본 뒤 고른 것이며, 사전 등록한 50\% 칸은 5단보다 0.13--0.18 낮았다."),
    (r"(벗어난 점은 부록 표~\ref{tab:a8}).",
     r"(벗어난 점은 부록 표~\ref{tab:a8}). 두 번째 환경, 정식 동등성 분석, Monte Carlo value 타깃은 새 수집이나 등록하지 않은 마진이 필요해 시도하지 않았다."),
    ("\\input{floats_ko/tab_a6}\n", "\\input{floats_ko/tab_a6}\n\\input{floats_ko/tab_a9}\n"),
])

for p, t in STAGED.items():
    io.open(p, "w", encoding="utf-8", newline="\n").write(t)
print("written", len(STAGED), "files")
