"""Add the review-round-3 B results (docs/review_b_plan_2026-10-21.md) to the manuscripts: section 4.8, Table 8 and Fig. 5
in the md drafts and the full tex (English, Korean), one paragraph in the short tex versions, and updated discussion and
limitation sentences. Every anchor must occur exactly once; numbers come from results/review_b*.csv.

    python3 scripts/2026-10-21/apply_review_b_text.py
"""
import io, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGED = {}  # files are written only after every anchor of every file has matched


def edit(path, pairs):
    p = os.path.join(ROOT, path)
    t = io.open(p, encoding="utf-8").read()
    for old, new in pairs:
        n = t.count(old)
        assert n == 1, f"{path}: anchor found {n} times: {old[:70]!r}"
        t = t.replace(old, new)
    STAGED[p] = t
    print(f"{path}: {len(pairs)} edits")


# ---------------------------------------------------------------- Table 8 rows (shared)
ROWS = [  # check, comparison, A, B, diff, lo, hi
    ("Training data 25%", "75% probe vs. rung 5", "0.681", "0.684", "−0.002", "−0.044", "0.036"),
    ("Training data 50%", "75% probe vs. rung 5", "0.703", "0.714", "−0.011", "−0.053", "0.027"),
    ("Training data 75%", "75% probe vs. rung 5", "0.712", "0.733", "−0.021", "−0.062", "0.020"),
    ("Training data 100%", "75% probe vs. rung 5", "0.727", "0.752", "−0.025", "−0.077", "0.029"),
    ("Detection at 5% false positives", "75% probe vs. rung 5", "0.149", "0.215", "−0.065", "−0.261", "0.087"),
    ("Detection at 10% false positives", "75% probe vs. rung 5", "0.268", "0.407", "−0.139", "−0.273", "0.075"),
    ("Partial AUROC (FPR ≤ 0.1)", "75% probe vs. rung 5", "0.552", "0.596", "−0.044", "−0.114", "0.030"),
    ("AUPRC", "75% probe vs. rung 5", "0.881", "0.904", "−0.023", "−0.054", "0.009"),
    ("Splitting by game", "75% probe: game vs. scene split", "0.721", "0.727", "−0.006", "−0.037", "0.025"),
    ("Splitting by game", "Rung 5: game vs. scene split", "0.725", "0.752", "−0.027", "−0.057", "0.001"),
    ("Action token", "75% action token vs. rung 5", "0.734", "0.752", "−0.019", "−0.085", "0.046"),
    ("Action token", "75% action token vs. 75% prompt token", "0.734", "0.727", "+0.006", "−0.050", "0.055"),
    ("Neural text encoder", "Rung 4 + embedding vs. rung 5", "0.637", "0.752", "−0.115", "−0.187", "−0.052"),
]


def md_table(foot):
    out = ["| Table 8 (exploratory). Additional checks, first 250 episodes, mean over steps 0–2 (AUROC unless stated) | A vs. B | A | B | A − B (95% CI) |",
           "|---|---|---|---|---|"]
    for c, v, a, b, d, lo, hi in ROWS:
        out.append(f"| {c} | {v} | {a} | {b} | {d} ({lo}, {hi}) |")
    return "\n".join(out) + "\n\n" + foot


def tm(x):  # number for LaTeX: minus sign in math
    return f"${x.replace('−', '-')}$" if x.startswith(("−", "+")) else x


def tex_table(foot):
    out = [r"\begin{table}[t]", r"\centering\small",
           r"\caption{Exploratory: additional checks on the first 250 episodes (mean over steps 0--2; AUROC unless stated).}\label{tab:extra}",
           r"\begin{tabularx}{\linewidth}{@{}LLccc@{}}", r"\toprule", r"Check & A vs.\ B & A & B & A $-$ B (95\% CI) \\", r"\midrule"]
    for c, v, a, b, d, lo, hi in ROWS:
        c = c.replace("%", r"\%").replace("FPR ≤ 0.1", r"FPR $\le$ 0.1")
        v = v.replace("%", r"\%").replace("vs.", r"vs.\ ")
        v = v.replace(r"vs.\  ", r"vs.\ ")
        out.append(f"{c} & {v} & {a} & {b} & {tm(d)} ({tm(lo)}, {tm(hi)}) \\\\")
    out += [r"\bottomrule", r"\end{tabularx}", r"\par\smallskip\raggedright\footnotesize", foot,
            "% Sources: results/review_b_*.csv, results/review_b6_action*.csv", r"\end{table}"]
    return "\n".join(out)


# ---------------------------------------------------------------- English md / tex 4.8
EN48 = [
    ("In response to review, we ran six further checks on the first 250 episodes at steps 0–2 (exploratory; plan fixed before the run, each analysis run once; no hold-out data). "
     "Intervals are from the scene-level cluster bootstrap (1,000 resamples), and values are means over the three steps unless stated (Table 8). With the full training folds, all four methods reproduced their earlier values exactly."),
    ("**Training-set size.** We trained the 75% probe and rung 5 on 25, 50, 75 and 100% of each outer training fold (about 50 to 200 episodes; five draws below 100%). "
     "The probe minus rung 5 was −0.002, −0.011, −0.021 and −0.025, all intervals including zero. The gap did not shrink as the training data grew, which suggests that within this range the probe was not limited by the number of training episodes; we did not test larger training sets. "
     "The 50% probe stayed below rung 5 at every size (−0.104 to −0.158)."),
    ("**Low false-positive region.** At a 10% false-positive rate the probe detected 0.268 of failures and rung 5 0.407 (difference −0.139, −0.273 to +0.075). "
     "Detection at 5% false positives, partial AUROC up to 10% false positives and AUPRC also gave differences whose intervals included zero, in the same direction as AUROC. "
     "With 56 successes, a 5% false-positive rate corresponds to about three successes, so these estimates are unstable. Steps saved against lost successes for the early-termination rules are plotted in Fig. 5."),
    ("**Splitting by game.** Splitting folds by game instead of by floor plan changed the AUROC by −0.006 (−0.037 to +0.025) for the probe and −0.027 (−0.057 to +0.001) for rung 5, "
     "so the inflation seen with a scene-only synthetic feature did not appear with the real methods. In the greedy data each game has one episode, so this split is also an episode split."),
    ("**Interval coverage at the task level.** On label-shuffled data (20 seeds × 3 steps), task-level intervals contained the null value in 90.0% (54/60; Wilson 95% CI 79.9–95.3%) of runs for the probe, "
     "85.0% (51/60) for rung 5 and 88.3% (53/60) for their difference, rather than 95%. Task-level intervals may therefore also be somewhat too narrow; the 60 runs share the same episodes, so these coverage intervals are themselves approximate."),
    ("**Action-token position.** Reading the hidden state at the last token of the chosen action, teacher-forced through the actor (the position read in [3]), gave an AUROC of 0.746 at 50% depth and 0.734 at 75% depth. "
     "The 75% action-token probe minus rung 5 was −0.019 (−0.085 to +0.046), with no step superior, so the conclusion did not change with the reading position. "
     "The 50% depth, weak at the prompt position (0.594), was about as strong as 75% at the action token, which suggests that the effect of depth depends on the position. The action token comes after the choice and therefore carries later information than the prompt token."),
    ("**Neural text encoder.** Replacing TF-IDF with sentence-encoder embeddings of the same prompt (bge-base-en-v1.5; 510-token chunks, averaged) gave 0.594 alone and 0.637 together with the rung-4 features, below rung 5 by −0.115 (−0.187 to −0.052). "
     "This encoder setup was weaker than TF-IDF, so it does not test whether a stronger text baseline exists, and the probe exceeding it (+0.090, 0.039 to 0.142) is not evidence that the probe beats neural text baselines."),
]
EN_FIG_CAP = ("Exploratory view of the early-termination results (Section 4.6): steps saved against lost successes for each rule, Qwen2.5-3B, 500 games, 100-action horizon. "
              "Left: all thresholds; right: losses up to 10%. Dots: thresholds chosen on other folds for a 5% target loss. The probe is the 75% depth probe with the step index. Curves sweep the threshold on the evaluated data.")
EN_FOOT_MD = "*n* = 250 (194 failures) per step; scene-level cluster bootstrap, 1,000 resamples. Training data as a share of each outer training fold; detection is the true-positive rate at the given false-positive rate. <!-- Sources: `results/review_b_*.csv`, `results/review_b6_action*.csv`, `results/early_stop_3b_curves.csv`, `results/early_stop_3b_fold.csv` -->"
EN_FOOT_TEX = r"$n = 250$ (194 failures) per step; scene-level cluster bootstrap, 1,000 resamples. Training data as a share of each outer training fold; detection is the true-positive rate at the given false-positive rate."


def md_to_tex(s, sec_refs):
    s = s.replace("%", r"\%").replace("**", "")
    for a, b in sec_refs: s = s.replace(a, b)
    s = s.replace("[3]", r"~\cite{ruan2026doomed}").replace(" ~\\cite", "~\\cite")
    s = s.replace("Fig. 5", r"Fig.~\ref{fig:early}").replace("Table 8", r"Table~\ref{tab:extra}")
    s = s.replace(" × ", r" $\times$ ")
    import re
    s = re.sub(r"(?<![\w.])([−+])(\d+\.\d+)", lambda m: f"${'-' if m.group(1) == '−' else '+'}{m.group(2)}$", s)
    s = re.sub(r"(\d)–(\d)", r"\1--\2", s)
    return s


EN_PARAS_TEX = []
for i, p in enumerate(EN48):
    t = md_to_tex(p, [])
    if i:
        head, rest = t.split(". ", 1) if t.startswith(("Training", "Low", "Splitting", "Interval", "Action", "Neural")) else (None, t)
        t = r"\paragraph{" + head + ".} " + rest
    EN_PARAS_TEX.append(t)
EN_FIG_TEX = "\n".join([r"\begin{figure}[tbp]", r"\centering", r"\includegraphics[width=\linewidth]{figA_early_stop_curves.pdf}",
                        r"\caption{" + md_to_tex(EN_FIG_CAP, [("(Section 4.6)", r"(Section~\ref{sec:early})")]) + r"}\label{fig:early}", r"\end{figure}"])

en_md_48 = ("### 4.8 Additional checks (exploratory)\n" + "\n\n".join(EN48) + "\n\n" + md_table(EN_FOOT_MD) +
            "\n\n![Fig. 5](../figures/figA_early_stop_curves.png)\n\n*Fig. 5. " + EN_FIG_CAP + "*\n\n")
en_tex_48s = (r"\subsection{Additional checks}\label{sec:extra}" + "\n" + "\n\n".join(EN_PARAS_TEX) + "\n\n" + tex_table(EN_FOOT_TEX) +
              "\n\n" + EN_FIG_TEX + "\n\n")

# discussion / limitation edits: (md old, md new, tex old, tex new)
EN_DISC = [
    ("although our probe also reads a different position (the last prompt token rather than the last action token).",
     "although our probe reads a different position (the last prompt token rather than the last action token); reading the action token instead gave the same conclusion in an exploratory check (Section 4.8).",
     None, "(Section~\\ref{sec:extra})."),
    ("including a nonlinear probe, did not exceed the baseline either.",
     "including a nonlinear probe, did not exceed the baseline either, and the gap to rung 5 did not shrink as the training data grew from about 50 to 200 episodes (Section 4.8).",
     None, "(Section~\\ref{sec:extra})."),
    ("whether more training data would close or reverse the gap we did not test.",
     "the gap did not shrink between about 50 and 200 training episodes (Section 4.8), and larger training sets were not tested.",
     None, "(Section~\\ref{sec:extra}),"),
    ("; we did not measure the difference with the real probe and baselines.",
     "; with the real probe and rung 5 the change was small (−0.006 and −0.027; Section 4.8).",
     "; we did not measure the difference with the real probe and baselines.",
     "; with the real probe and rung 5 the change was small ($-0.006$ and $-0.027$; Section~\\ref{sec:extra})."),
    ("(coverage 80–94% instead of 95%).",
     "(coverage 80–94% instead of 95%), and task-level intervals were closer to but still below nominal (85–90%; Section 4.8).",
     "(coverage 80--94\\% instead of 95\\%).",
     "(coverage 80--94\\% instead of 95\\%), and task-level intervals were closer to but still below nominal (85--90\\%; Section~\\ref{sec:extra})."),
    ("TF-IDF with logistic regression is a weak text baseline; neural text encoders reading the same prompt were not tested. The probe reads the last prompt token, not the generated action token used in [3].",
     "TF-IDF with logistic regression is a weak text baseline, and the one neural text encoder we tried (bge-base-en-v1.5 embeddings of the same prompt) was weaker still (Section 4.8), so whether a stronger text baseline would widen the gap remains open. The main probe reads the last prompt token, not the action token used in [3]; that reading the action token gives the same conclusion was checked only in an exploratory analysis of the first 250 episodes.",
     "TF-IDF with logistic regression is a weak text baseline; neural text encoders reading the same prompt were not tested. The probe reads the last prompt token, not the generated action token used in~\\cite{ruan2026doomed}.",
     "TF-IDF with logistic regression is a weak text baseline, and the one neural text encoder we tried (bge-base-en-v1.5 embeddings of the same prompt) was weaker still (Section~\\ref{sec:extra}), so whether a stronger text baseline would widen the gap remains open. The main probe reads the last prompt token, not the action token used in~\\cite{ruan2026doomed}; that reading the action token gives the same conclusion was checked only in an exploratory analysis of the first 250 episodes."),
    ("AUROC does not target the low false-positive region that matters for stopping, and we did not report true-positive rates at low false-positive rates. The attainable AUROC from early states was not estimated. Interval coverage was checked only at the attempt level, and the variance",
     "AUROC does not target the low false-positive region that matters for stopping; detection rates at 5% and 10% false positives agreed with AUROC in an exploratory check but rest on 56 successes. The attainable AUROC from early states was not estimated. On shuffled labels, task-level intervals covered the null value in 85–90% of runs rather than 95% (Section 4.8), so they, like the attempt-level intervals, may be too narrow, and the variance",
     "AUROC does not target the low false-positive region that matters for stopping, and we did not report true-positive rates at low false-positive rates. The attainable AUROC from early states was not estimated. Interval coverage was checked only at the attempt level, and the variance",
     "AUROC does not target the low false-positive region that matters for stopping; detection rates at 5\\% and 10\\% false positives agreed with AUROC in an exploratory check but rest on 56 successes. The attainable AUROC from early states was not estimated. On shuffled labels, task-level intervals covered the null value in 85--90\\% of runs rather than 95\\% (Section~\\ref{sec:extra}), so they, like the attempt-level intervals, may be too narrow, and the variance"),
]


def tex_pairs(disc):
    out = []
    for mo, mn, to, tn in disc:
        if to is None:  # tex sentence identical to md except the section reference
            out.append((mo, mn.replace("(Section 4.8).", tn).replace("(Section 4.8),", tn)))
        else:
            out.append((to, tn))
    return out


edit("docs/paper_draft_en.md", [("\n## 5 Discussion\n", "\n" + en_md_48 + "## 5 Discussion\n")] + [(mo, mn) for mo, mn, _, _ in EN_DISC])
edit("paper/main_en.tex", [("\\subsection{Early termination}\n", "\\subsection{Early termination}\\label{sec:early}\n"),
                           ("\n\\section{Discussion}\n", "\n" + en_tex_48s + "\\section{Discussion}\n")] + tex_pairs(EN_DISC))

# ---------------------------------------------------------------- Korean
KO48 = [
    ("리뷰에 따라 앞 250개 episode의 step 0–2에서 점검 여섯 가지를 더 했다(탐색; 계획은 실행 전 고정, 분석마다 1회 실행; 홀드아웃 데이터는 쓰지 않음). "
     "신뢰구간은 장면 단위 cluster bootstrap(1,000회)이고, 따로 적지 않으면 값은 세 step의 평균이다(표 8). 학습 fold 전체로 학습하면 네 방법 모두 앞선 값을 그대로 재현했다."),
    ("**학습 표본 수.** 75% 프로브와 5단을 각 바깥 학습 fold의 25, 50, 75, 100%(약 50–200개 episode, 100% 미만은 5번 뽑기)로 학습했다. "
     "프로브 − 5단은 −0.002, −0.011, −0.021, −0.025였고 신뢰구간은 모두 0을 포함했다. 학습 데이터가 늘어도 격차가 줄지 않았으므로, 이 범위에서는 프로브가 학습 episode 수에 제한되지 않았음을 시사한다. 더 큰 학습 집합은 시험하지 않았다. "
     "50% 프로브는 모든 크기에서 5단보다 낮았다(−0.104 ~ −0.158)."),
    ("**낮은 오탐 구간.** 오탐률 10%에서 프로브는 실패의 0.268을, 5단은 0.407을 검출했다(차이 −0.139, −0.273 ~ +0.075). "
     "오탐률 5%의 검출률, 오탐률 10%까지의 부분 AUROC, AUPRC의 차이도 신뢰구간이 모두 0을 포함했고 AUROC와 같은 방향이었다. "
     "성공이 56개라 오탐률 5%는 성공 약 3개에 해당하므로 이 추정은 불안정하다. 조기 중단 규칙의 성공 손실 대 절감률은 그림 5에 그렸다."),
    ("**게임 단위 분할.** fold를 방 배치가 아니라 게임 단위로 나누면 AUROC가 프로브에서 −0.006(−0.037 ~ +0.025), 5단에서 −0.027(−0.057 ~ +0.001) 달라졌다. "
     "장면 정보만 담은 합성 특징에서 본 부풀림은 실제 방법에서 나타나지 않았다. 탐욕 데이터는 게임당 episode가 하나라 이 분할은 episode 단위 분할이기도 하다."),
    ("**과제 수준 구간 포함률.** 라벨을 섞은 데이터(20 seed × 3 step)에서 과제 수준 구간이 귀무값을 포함한 비율은 프로브 90.0%(54/60; Wilson 95% CI 79.9–95.3%), "
     "5단 85.0%(51/60), 둘의 차이 88.3%(53/60)로 95%보다 낮았다. 따라서 과제 수준 구간도 다소 좁을 수 있다. 60회는 같은 episode를 공유하므로 이 포함률의 구간도 근사이다."),
    ("**action 토큰 위치.** 고른 action의 마지막 토큰에서 hidden state를 읽으면(actor에 teacher forcing, [3]이 읽는 위치) AUROC가 50% 깊이 0.746, 75% 깊이 0.734였다. "
     "75% action 토큰 프로브 − 5단은 −0.019(−0.085 ~ +0.046)였고 우수한 step은 없어, 읽는 위치를 바꿔도 결론은 같았다. "
     "프롬프트 위치에서 약했던 50% 깊이(0.594)가 action 토큰에서는 75%와 비슷하게 강해, 깊이의 효과가 위치에 달려 있음을 시사한다. action 토큰은 행동을 고른 뒤라 프롬프트 토큰보다 늦은 정보를 담는다."),
    ("**신경망 텍스트 인코더.** TF-IDF 대신 같은 프롬프트의 문장 인코더 임베딩(bge-base-en-v1.5; 510 토큰 조각의 평균)을 쓰면 단독 0.594, 4단 특징과 함께 0.637로 5단보다 −0.115(−0.187 ~ −0.052) 낮았다. "
     "이 인코더 설정은 TF-IDF보다 약했으므로 더 강한 텍스트 baseline이 있는지는 이것으로 시험되지 않았고, 프로브가 이보다 높았던 것(+0.090, 0.039 ~ 0.142)도 프로브가 신경망 텍스트 baseline을 넘는다는 근거가 아니다."),
]
KO_FIG_CAP = ("조기 중단 결과의 탐색적 그림(4.6절): 규칙별 성공 손실 대 절감한 step 비율(Qwen2.5-3B, 500게임, 100 action 상한). "
              "왼쪽: 모든 문턱, 오른쪽: 손실 10%까지. 점: 다른 fold에서 목표 손실 5%로 고른 문턱. 프로브는 75% 깊이 프로브에 step 번호를 더한 것이다. 곡선은 평가 데이터에서 문턱을 바꿔 가며 그렸다.")
KO_FOOT_MD = "step당 *n* = 250(실패 194). 장면 단위 cluster bootstrap 1,000회. 학습 데이터는 각 바깥 학습 fold에 대한 비율, 검출률은 해당 오탐률에서의 참양성률. <!-- 출처: `results/review_b_*.csv`, `results/review_b6_action*.csv`, `results/early_stop_3b_curves.csv`, `results/early_stop_3b_fold.csv` -->"
KO_FOOT_TEX = r"step당 $n = 250$(실패 194). 장면 단위 cluster bootstrap 1,000회. 학습 데이터는 각 바깥 학습 fold에 대한 비율, 검출률은 해당 오탐률에서의 참양성률."


def ko_md_to_tex(s, refs=()):
    import re
    s = s.replace("%", r"\%").replace("**", "")
    for a, b in refs: s = s.replace(a, b)
    s = s.replace("[3]", r"~\cite{ruan2026doomed}").replace("(actor에 teacher forcing, ~\\cite", "(actor에 teacher forcing, \\cite")
    s = s.replace("그림 5", r"그림~\ref{fig:early}").replace("표 8", r"표~\ref{tab:extra}")
    s = s.replace(" × ", r" $\times$ ").replace(" ~ ", r" \textasciitilde{} ")
    s = re.sub(r"(?<![\w.])([−+])(\d+\.\d+)", lambda m: f"${'-' if m.group(1) == '−' else '+'}{m.group(2)}$", s)
    s = s.replace("프로브 − 5단", r"프로브 $-$ 5단")
    s = re.sub(r"(\d)–(\d)", r"\1--\2", s)
    return s


KO_PARAS_TEX = []
for i, p in enumerate(KO48):
    t = ko_md_to_tex(p)
    if i:
        head, rest = t.split(". ", 1)
        t = r"\paragraph{" + head + ".} " + rest
    KO_PARAS_TEX.append(t)
KO_FIG_TEX = "\n".join([r"\begin{figure}[tbp]", r"\centering", r"\includegraphics[width=\linewidth]{figA_early_stop_curves_ko.pdf}",
                        r"\caption{" + ko_md_to_tex(KO_FIG_CAP, [("(4.6절)", r"(\ref{sec:early}절)")]) + r"}\label{fig:early}", r"\end{figure}"])
ko_md_48 = ("### 4.8 추가 점검 (탐색)\n" + "\n\n".join(KO48) + "\n\n" + md_table(KO_FOOT_MD) +
            "\n\n![Fig. 5](../figures/figA_early_stop_curves_ko.png)\n\n*Fig. 5. " + KO_FIG_CAP + "*\n\n")
ko_tex_48s = (r"\subsection{추가 점검}\label{sec:extra}" + "\n" + "\n\n".join(KO_PARAS_TEX) + "\n\n" + tex_table(KO_FOOT_TEX) +
              "\n\n" + KO_FIG_TEX + "\n\n")

KO_DISC = [  # (md old, md new, tex old or None (same), tex new or section-ref replacement)
    ("우리 프로브는 읽는 위치도 다르다(action 마지막 토큰이 아니라 프롬프트 마지막 토큰).",
     "우리 프로브는 읽는 위치도 다르다(action 마지막 토큰이 아니라 프롬프트 마지막 토큰). 다만 action 토큰을 읽어도 탐색 점검에서 결론은 같았다(4.8절).",
     None, None),
    ("다만 reader, 깊이, 프로브 종류(비선형 포함)의 탐색 조합 96개 중 baseline을 넘은 것도 없었다.",
     "다만 reader, 깊이, 프로브 종류(비선형 포함)의 탐색 조합 96개 중 baseline을 넘은 것도 없었고, 학습 데이터를 약 50개에서 200개 episode로 늘려도 5단과의 격차는 줄지 않았다(4.8절).",
     None, None),
    ("학습 데이터를 늘리면 격차가 줄거나 뒤집히는지는 시험하지 않았다.",
     "학습 episode 약 50–200개 범위에서는 격차가 줄지 않았고(4.8절), 더 큰 학습 집합은 시험하지 않았다.",
     None, None),
    ("실제 프로브와 baseline으로 이 차이를 재지는 않았다.",
     "실제 프로브와 5단에서는 변화가 작았다(−0.006, −0.027; 4.8절).",
     None, None),
    ("(포함률 95%가 아니라 80–94%).",
     "(포함률 95%가 아니라 80–94%). 과제 수준 구간은 이보다 낫지만 여전히 명목보다 낮았다(85–90%; 4.8절).",
     None, None),
    ("TF-IDF와 로지스틱 회귀는 약한 텍스트 baseline이며, 같은 프롬프트를 읽는 신경망 텍스트 인코더는 시험하지 않았다. 프로브는 [3]처럼 생성한 action 토큰이 아니라 프롬프트 마지막 토큰을 읽는다.",
     "TF-IDF와 로지스틱 회귀는 약한 텍스트 baseline이고, 시험한 신경망 텍스트 인코더 하나(같은 프롬프트의 bge-base-en-v1.5 임베딩)는 이보다도 약했으므로(4.8절) 더 강한 텍스트 baseline이 격차를 넓힐지는 열려 있다. 주 프로브는 [3]처럼 action 토큰이 아니라 프롬프트 마지막 토큰을 읽으며, action 토큰을 읽어도 결론이 같다는 것은 앞 250개 episode의 탐색 점검으로만 확인했다.",
     None, None),
    ("AUROC는 중단에 중요한 낮은 오탐 구간을 겨냥하지 않으며, 낮은 오탐률에서의 검출률은 보고하지 않았다. 초반 상태로 도달 가능한 AUROC(상한)는 추정하지 않았다. 구간 포함률은 시도 수준에서만 점검했고,",
     "AUROC는 중단에 중요한 낮은 오탐 구간을 겨냥하지 않는다. 오탐률 5%·10%의 검출률은 탐색 점검에서 AUROC와 같은 방향이었지만 성공 56개에 기대고 있다. 초반 상태로 도달 가능한 AUROC(상한)는 추정하지 않았다. 라벨을 섞은 데이터에서 과제 수준 구간은 귀무값을 95%가 아니라 85–90%만 포함해(4.8절) 시도 수준 구간처럼 좁을 수 있고,",
     None, None),
]


def ko_tex_pair(mo, mn):
    import re
    conv = lambda s: re.sub(r"(\d)–(\d)", r"\1--\2", s.replace("%", r"\%").replace(" [3]", r"~\cite{ruan2026doomed}")
                            .replace("(4.8절)", r"(\ref{sec:extra}절)").replace("; 4.8절)", r"; \ref{sec:extra}절)"))
    nn = conv(mn)
    nn = re.sub(r"(?<![\w.])−(\d+\.\d+)", r"$-\1$", nn)
    return conv(mo), nn


edit("docs/paper_draft_ko.md", [("\n## 5 토의\n", "\n" + ko_md_48 + "## 5 토의\n")] + [(mo, mn) for mo, mn, _, _ in KO_DISC])
edit("paper/main_ko.tex", [("\\subsection{조기 중단}\n", "\\subsection{조기 중단}\\label{sec:early}\n"),
                           ("\n\\section{토의}\n", "\n" + ko_tex_48s + "\\section{토의}\n")] + [ko_tex_pair(mo, mn) for mo, mn, _, _ in KO_DISC])

# ---------------------------------------------------------------- short versions
EN_SHORT_PARA = ("Exploratory checks on the first 250 episodes (each run once): the probe minus rung 5 did not shrink as the training data grew from 25\\% to 100\\% of each training fold ($-0.002$ to $-0.025$); "
                 "at a 10\\% false-positive rate the probe detected 0.268 of failures and rung 5 0.407 (difference $-0.139$, $-0.273$ to $+0.075$); splitting by game changed the AUROC by $-0.006$ for the probe and $-0.027$ for rung 5; "
                 "reading the hidden state at the last action token gave the same conclusion (75\\% depth minus rung 5 $-0.019$, $-0.085$ to $+0.046$); and a sentence-encoder text baseline (bge-base-en-v1.5) was weaker than TF-IDF ($-0.115$). "
                 "On shuffled labels, task-level intervals covered the null value in 85--90\\% of runs rather than 95\\%.\n")
edit("paper/main_en_short.tex", [
    ("From step 7 on, rung 4 uses actions older than the six steps shown in the prompt.\n\n\\section{Discussion}",
     "From step 7 on, rung 4 uses actions older than the six steps shown in the prompt.\n" + EN_SHORT_PARA + "\n\\section{Discussion}"),
    ("and our probe reads a different position.", "and our probe reads a different position, although reading the action token gave the same conclusion (Section~\\ref{sec:robust})."),
    ("whether more training data would close or reverse the gap we did not test.",
     "the gap did not shrink between about 50 and 200 training episodes, and larger training sets were not tested."),
    ("(not measured with the real probe).", "(with the real probe and rung 5 the change was small, $-0.006$ and $-0.027$)."),
    ("(coverage 80--94\\% instead of 95\\%).", "(coverage 80--94\\% instead of 95\\%; task level 85--90\\%)."),
    ("TF-IDF is a weak text baseline and neural text encoders were not tested; AUROC does not target the low false-positive region that matters for stopping;",
     "TF-IDF is a weak text baseline and the one neural encoder we tried was weaker still; low false-positive detection agreed with AUROC only in an exploratory check resting on 56 successes; task-level intervals covered the null in 85--90\\% of shuffled-label runs rather than 95\\%;"),
])
KO_SHORT_PARA = ("앞 250개 episode의 탐색 점검(분석마다 1회)에서, 학습 데이터를 각 학습 fold의 25\\%에서 100\\%로 늘려도 프로브 $-$ 5단은 줄지 않았고($-0.002$ \\textasciitilde{} $-0.025$), "
                 "오탐률 10\\%에서 프로브는 실패의 0.268을, 5단은 0.407을 검출했다(차이 $-0.139$, $-0.273$ \\textasciitilde{} $+0.075$). 게임 단위 분할은 AUROC를 프로브에서 $-0.006$, 5단에서 $-0.027$만 바꿨다. "
                 "action 마지막 토큰의 hidden state를 읽어도 결론은 같았고(75\\% 깊이 $-$ 5단 $-0.019$, $-0.085$ \\textasciitilde{} $+0.046$), 문장 인코더(bge-base-en-v1.5) 텍스트 baseline은 TF-IDF보다 약했다($-0.115$). "
                 "라벨을 섞은 데이터에서 과제 수준 구간은 귀무값을 95\\%가 아니라 85--90\\%만 포함했다.\n")
edit("paper/main_ko_short.tex", [
    ("step 7부터는 4단이 프롬프트에 보이는 최근 6 step보다 오래된 action을 쓴다.\n\n\\section{토의}",
     "step 7부터는 4단이 프롬프트에 보이는 최근 6 step보다 오래된 action을 쓴다.\n" + KO_SHORT_PARA + "\n\\section{토의}"),
    ("다만 그들의 환경은 재현하지 않았고 프로브가 읽는 위치도 다르다.",
     "다만 그들의 환경은 재현하지 않았고 프로브가 읽는 위치도 다르다(action 토큰을 읽어도 결론은 같았다, \\ref{sec:robust}절)."),
    ("학습 데이터를 늘리면 격차가 줄거나 뒤집히는지는 시험하지 않았다.",
     "학습 episode 약 50--200개 범위에서는 격차가 줄지 않았고, 더 큰 학습 집합은 시험하지 않았다."),
    ("(실제 프로브로는 재지 않음).", "(실제 프로브와 5단에서는 $-0.006$, $-0.027$로 작았다)."),
    ("(포함률 95\\%가 아니라 80--94\\%).", "(포함률 95\\%가 아니라 80--94\\%, 과제 수준은 85--90\\%)."),
    ("TF-IDF는 약한 텍스트 baseline이며 신경망 텍스트 인코더는 시험하지 않았다. AUROC는 중단에 중요한 낮은 오탐 구간을 겨냥하지 않고, 초반 상태로 도달 가능한 AUROC는 추정하지 않았다.",
     "TF-IDF는 약한 텍스트 baseline이며, 시험한 신경망 인코더 하나는 이보다도 약했다. 낮은 오탐 구간의 검출률은 성공 56개로 한 탐색 점검에서만 AUROC와 같은 방향이었고, 과제 수준 구간은 라벨을 섞은 데이터에서 귀무값을 85--90\\%만 포함했다. 초반 상태로 도달 가능한 AUROC는 추정하지 않았다."),
])

for p, t in STAGED.items():
    io.open(p, "w", encoding="utf-8", newline="\n").write(t)
print("written", len(STAGED), "files")
