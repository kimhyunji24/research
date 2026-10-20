"""2026-10-21 third review, part A (text only, no new analysis): claims narrowed to the evidence, facts disclosed,
citations corrected, limitations and a deviations table (A8) added. Applies to the md drafts, the full .tex files and
the short .tex files (English and Korean). Every anchor is asserted to occur exactly once."""
import io
import re
import sys

R = "/root/research/"
F = {"en_md": "docs/paper_draft_en.md", "ko_md": "docs/paper_draft_ko.md", "en_tex": "paper/main_en.tex",
     "ko_tex": "paper/main_ko.tex", "en_short": "paper/main_en_short.tex", "ko_short": "paper/main_ko_short.tex"}
T = {k: io.open(R + v, encoding="utf-8").read() for k, v in F.items()}
for k in T:
    assert "tab:a8" not in T[k] and "Table A8" not in T[k] and "표 A8" not in T[k], k

KEYS = {1: "pham2026agentstop", 2: "wang2026failfast", 3: "ruan2026doomed", 4: "kadavath2022know", 5: "kuhn2023semantic",
        6: "kossen2024sep", 7: "moreno2026noanswer", 8: "shi2026earlyeval", 9: "mehta2026commit", 10: "bulut2026problem",
        11: "lugoloobi2026failures", 12: "shridhar2021alfworld", 13: "yao2023react", 14: "khan2026companion",
        15: "pan2026monitoring", 16: "poulis2026truth", 17: "marks2023geometry", 18: "burns2022latent",
        19: "ghasemabadi2025gnosis", 20: "dubey2026realtime", 21: "advani2026false"}


def tex(s):
    s = s.replace("%", "\\%")
    s = re.sub(r"−(\d[\d.,]*)", r"$-\1$", s)
    s = s.replace("–", "--")
    s = s.replace(" ~ ", " \\textasciitilde{} ")
    s = re.sub(r"\s?\[(\d+)\]", lambda m: "~\\cite{" + KEYS[int(m.group(1))] + "}", s)
    s = s.replace("Appendix Table A8", "Appendix Table~\\ref{tab:a8}").replace("부록 표 A8", "부록 표~\\ref{tab:a8}")
    s = re.sub(r"^- \*\*(.+?)\*\* ", r"\\item \\textbf{\1} ", s, flags=re.M)
    s = re.sub(r"\*\*(.+?)\*\* ", r"\\paragraph{\1} ", s)
    return s


def rep(k, old, new):
    assert T[k].count(old) == 1, (k, old[:80], T[k].count(old))
    T[k] = T[k].replace(old, new)


def after_sentence(k, anchor, add):
    """Insert `add` right after the sentence that contains `anchor`."""
    assert T[k].count(anchor) == 1, (k, anchor[:80], T[k].count(anchor))
    i = T[k].index(anchor) + len(anchor)
    m = re.compile(r"\.(?=\s|$)").search(T[k], i)
    j = m.end()
    T[k] = T[k][:j] + add + T[k][j:]


def both(lang, kind, *args, new_is_text=True):
    """Apply to the md draft and the full tex of one language; md text is converted for tex."""
    md, tx = f"{lang}_md", f"{lang}_tex"
    if kind == "rep":
        old, new = args
        rep(md, old, new)
        rep(tx, tex(old), tex(new))
    elif kind == "after":
        anchor, add = args
        after_sentence(md, anchor, add)
        after_sentence(tx, tex(anchor), tex(add))


# ======================= English: md + full tex =======================
both("en", "rep", "In the pre-registered comparison the probe was below the full baseline",
     "In the pre-registered comparison (probe at 50% depth) the probe was below the full baseline")
both("en", "rep", "Used to stop failing episodes, the probe was not shown to save more actions than a fixed limit",
     "Used to stop failing episodes with at most 10% of successes lost (the pre-specified range), the probe was not shown to save more actions than a fixed limit")
both("en", "rep", "In this setting, hidden states did not add to visible information;",
     "At the best exploratory depth the differences on the same games were −0.034 to −0.011. In this setting we found no gain from hidden states over visible information, and the held-out intervals exclude gains above about 0.10 AUROC, although smaller gains cannot be ruled out;")
both("en", "rep", "Under the same parameter-proportional approximation, a 0.5B monitor would add about 13–16% to a 3B actor, and a 7B monitor would cost about 2.4 times as much as the actor.",
     "Per token, that monitor is 15–52 times cheaper than those policies. A separate 0.5B monitor that reads every token a 3B actor reads would instead add about 13–16% of the actor's compute, and a 7B monitor would cost about 2.4 times as much as the actor.")
both("en", "rep", "Their probe, replay procedure and task-grouped evaluation are close to ours.",
     "Their replay procedure and task-grouped evaluation are close to ours, but their probe reads the last token of the generated action, whereas ours reads the last prompt token before the candidate actions are scored.")
both("en", "after", "results for them concern external monitoring, not self-monitoring",
     " Because a hidden state at the last prompt token is a deterministic function of the prompt, the question is not whether it holds information absent from the prompt but whether the model's computation makes failure-relevant features linearly readable beyond what the visible-information baselines extract.")
both("en", "after", "In total 800 rollouts were used",
     " The 100 games were drawn from the 500 greedy games and can overlap with the first 250; at the attempt level the probe and the baselines were cross-fitted by scene on the sampled rollouts only.")
both("en", "rep", "and ran it once. Every later analysis",
     "and ran it once. The 50% depth was fixed in the plan without a recorded rationale, and the plan did not state whether H1 required all three rounds or any of them; all three differences fell below zero, so the verdict does not depend on that rule. Every later analysis")
both("en", "after", "the decision rule had been committed beforehand and was not changed",
     " Appendix Table A8 lists all deviations from the registered plans and all exposures to results before a decision.")
both("en", "after", "is its parameter count relative to the actor's",
     " This assumes that the monitor reads every token the actor reads. [2] reports a lower overall overhead (0.1–0.6%) than its per-token ratio, so its figure also depends on how many tokens its monitor reads.")
both("en", "after", "whereas at step 1 only rung 5 did",
     " These verdicts rest on the permutation test; the 95% bootstrap interval of rung 4 at step 3 (0.492–0.644) includes 0.5.")
both("en", "after", "so a practical gain from self-monitoring was not shown in this setting",
     " This verdict holds for the pre-specified range of up to 10% lost successes; the secondary results below depend on the loss range and on how the threshold is chosen.")
both("en", "after", "Adding the hidden state to rung 5 did not improve it",
     " With about 250 training episodes, adding 2,048 dimensions to rung 5 can also overfit, so this check is weak.")
both("en", "after", "was inferior before correction for the 11 steps",
     " From step 7 on, the progress features of rung 4 use actions older than the six steps shown in the prompt, so at these steps the baselines see information that the probe input does not.")
both("en", "rep", "our results suggest that comparisons against task-blind monitors overstate what hidden states add.",
     "our results suggest that comparisons against task-blind monitors may overstate what hidden states add, although our probe also reads a different position (the last prompt token rather than the last action token).")
both("en", "rep", "Several design choices changed conclusions in ways that matter for similar studies. Splitting by game instead of by floor plan inflated a scene-only synthetic feature to an AUROC of 0.97.",
     "Several design choices can change conclusions in similar studies. Splitting by game instead of by floor plan inflated a scene-only synthetic feature to an AUROC of 0.97; we did not measure the difference with the real probe and baselines.")
both("en", "after", "The 7B reader was 4-bit quantized, which is confounded with size",
     " Readers also differ in hidden size (896, 1,536, 2,048 and 3,584 dimensions), which changes the ratio of dimensions to training episodes.")
both("en", "rep", "Early failure was largely a property of the game rather than of the attempt,",
     "In our sample, whose games were stratified by greedy outcome, early failure was largely a property of the game rather than of the attempt,")
both("en", "rep", "Using the probe to stop doomed episodes was not shown to save more steps than a fixed step limit.",
     "Using the probe to stop doomed episodes was not shown to save more steps than a fixed step limit at up to 10% lost successes.")
# figure 3 caption (md caption and the tex caption)
both("en", "after", "(18; none superior)",
     " The comparisons share games and baselines and are not independent, so their count is not a pooled estimate.")
# complementary reading paragraph
COMP_EN = ("**A complementary reading.** Read at 75% depth from a single token and trained on about 250 episodes per round without hand-made features, "
           "the probe came within 0.011–0.034 AUROC of the engineered rung-5 baseline on the first 250 episodes, and on the hold-out it was ahead at steps 0 and 1, though not significantly. "
           "Much of the failure-relevant information is therefore linearly present in the representation; whether more training data would close or reverse the gap we did not test.\n\n")
rep("en_md", "**Practical implication.**", COMP_EN + "**Practical implication.**")
rep("en_tex", "\\paragraph{Practical implication.}", tex(COMP_EN) + "\\paragraph{Practical implication.}")
# limitations: new items after "Post hoc choices"
LIM_EN = ("\n- **Comparison design.** TF-IDF with logistic regression is a weak text baseline; neural text encoders reading the same prompt were not tested. "
          "The probe reads the last prompt token, not the generated action token used in [3]. From step 7 on, the progress features of rung 4, and of rungs 4t and 5t in the early-termination analysis, use actions older than the six steps shown in the prompt."
          "\n- **Metrics and statistics.** AUROC does not target the low false-positive region that matters for stopping, and we did not report true-positive rates at low false-positive rates. "
          "The attainable AUROC from early states was not estimated. Interval coverage was checked only at the attempt level, and the variance across cross-validation seeds is not in the intervals. "
          "The 18 pre-registered, pre-specified and confirmatory comparisons share games and baselines and are not independent."
          "\n- **Registration.** Plans were registered as dated commits in the project repository rather than with an external registry; deviations and early exposures are listed in Appendix Table A8.")
PH_EN = "the margin was also defined for between-game AUROC and is on a different scale at the attempt level."
rep("en_md", PH_EN, PH_EN + LIM_EN)
rep("en_tex", PH_EN, PH_EN + tex(LIM_EN))

# ======================= Korean: md + full tex =======================
both("ko", "rep", "사전 등록한 비교에서 프로브는", "사전 등록한 비교(50% 깊이 프로브)에서 프로브는")
both("ko", "rep", "실패할 episode를 멈추는 데 쓰면, 프로브가 고정 상한보다 action을 더 아낀다는 것은 보이지 않았다",
     "성공 손실 10% 이하(사전에 정한 범위)에서 실패할 episode를 멈추는 데 쓰면, 프로브가 고정 상한보다 action을 더 아낀다는 것은 보이지 않았다")
both("ko", "rep", "이 설정에서 hidden state는 겉보기 정보에 더하는 것이 없었으며,",
     "탐색에서 가장 좋은 깊이의 차이는 같은 게임에서 −0.034 ~ −0.011이었다. 이 설정에서 hidden state의 이득은 찾지 못했고, 홀드아웃 신뢰구간은 약 0.10 AUROC를 넘는 이득을 배제하지만 그보다 작은 이득은 배제하지 못한다. 따라서")
both("ko", "rep", "같은 파라미터 비례 근사로 계산하면 0.5B 모니터는 3B actor에 약 13–16%를 더하고, 7B 모니터는 actor의 약 2.4배가 든다.",
     "토큰당으로는 이 모니터가 그 policy들보다 15–52배 싸다. 반면 3B actor가 읽는 토큰을 모두 읽는 별도 0.5B 모니터는 actor 계산량의 약 13–16%를 더하고, 7B 모니터는 actor의 약 2.4배가 든다.")
both("ko", "rep", "이들의 프로브, 재생 절차, 과제 단위 평가는 우리와 가깝다.",
     "이들의 재생 절차와 과제 단위 평가는 우리와 가깝지만, 이들의 프로브는 생성한 action의 마지막 토큰을 읽고 우리 프로브는 후보 action을 점수화하기 전 프롬프트의 마지막 토큰을 읽는다.")
both("ko", "after", "이들의 결과는 자기 감시가 아니라 외부 감시에 관한 것이다",
     " 프롬프트 마지막 토큰의 hidden state는 프롬프트의 결정적 함수이므로, 질문은 hidden state에 프롬프트 밖의 정보가 있는지가 아니라 모델의 계산이 실패와 관련된 특징을 겉보기 정보 baseline이 뽑는 것 이상으로 선형으로 읽히게 만드는지이다.")
both("ko", "after", "모두 800개 rollout을 사용했다",
     " 이 100게임은 탐욕 게임 500개에서 뽑았으므로 앞 250개와 겹칠 수 있고, 시도 수준에서 프로브와 baseline은 샘플링 rollout만으로 장면 단위 교차적합했다.")
both("ko", "rep", "같은 episode에서 그 뒤에 한 분석은 모두 탐색적이라고 표시한다.",
     "50% 깊이는 근거를 기록하지 않은 채 계획에 고정했고, H1이 세 라운드 모두를 요구하는지 하나만 요구하는지는 계획에 없었다. 세 차이가 모두 0 아래였으므로 판정은 그 규칙에 달려 있지 않다. 같은 episode에서 그 뒤에 한 분석은 모두 탐색적이라고 표시한다.")
both("ko", "after", "판정 규칙은 그 전에 commit했고 바꾸지 않았다",
     " 등록한 계획에서 벗어난 점과 판정 전 결과 노출은 모두 부록 표 A8에 적었다.")
both("ko", "after", "그러면 같은 토큰을 읽는 별도 모니터의 추가 비용은 actor 대비 파라미터 수의 비율이다",
     " 이는 모니터가 actor가 읽는 토큰을 모두 읽는다고 가정한 것이다. [2]는 토큰당 비율보다 낮은 전체 비용(0.1–0.6%)을 보고하므로, 그 값은 모니터가 읽는 토큰 수에도 달려 있다.")
both("ko", "after", "step 1에서는 5단만 그랬다",
     " 이 판정은 순열 검정에 따른 것이며, step 3에서 4단의 95% bootstrap 구간(0.492–0.644)은 0.5를 포함한다.")
both("ko", "after", "이 설정에서 자기 감시의 실용 이득은 확인되지 않았다",
     " 이 판정은 사전에 정한 성공 손실 10% 이하 범위에 대한 것이며, 아래 보조 결과는 손실 범위와 문턱 선택 방식에 따라 달라진다.")
both("ko", "after", "5단에 hidden state를 더해도 나아지지 않았다",
     " 학습 episode가 약 250개라 5단에 2,048차원을 더하면 과적합할 수도 있어, 이 점검은 약하다.")
both("ko", "after", "11개 step에 대한 보정 전 기준으로 열등이었다",
     " step 7부터는 4단의 진행도 특징이 프롬프트에 보이는 최근 6 step보다 오래된 action을 쓰므로, 이 step들에서는 baseline이 프로브 입력에 없는 정보를 본다.")
both("ko", "rep", "다만 과제를 보지 못하는 모니터와의 비교는 hidden state가 더하는 정보를 과대평가할 수 있음을 시사한다.",
     "다만 과제를 보지 못하는 모니터와의 비교는 hidden state가 더하는 정보를 과대평가할 수 있음을 시사한다. 우리 프로브는 읽는 위치도 다르다(action 마지막 토큰이 아니라 프롬프트 마지막 토큰).")
both("ko", "rep", "몇몇 설계 선택은 비슷한 연구에서 결론을 바꿀 만큼 중요했다. 방 배치가 아니라 게임 단위로 나누면 장면 정보만 담은 합성 특징의 AUROC가 0.97까지 부풀었다.",
     "몇몇 설계 선택은 비슷한 연구에서 결론을 바꿀 수 있다. 방 배치가 아니라 게임 단위로 나누면 장면 정보만 담은 합성 특징의 AUROC가 0.97까지 부풀었다. 실제 프로브와 baseline으로 이 차이를 재지는 않았다.")
both("ko", "after", "7B reader는 4비트로 양자화되어 크기 효과와 교락된다",
     " reader마다 hidden 차원도 달라(896, 1,536, 2,048, 3,584) 학습 episode 대비 차원 비율이 바뀐다.")
both("ko", "rep", "초반의 실패는 대체로 시도가 아니라 게임의 성질이었고,",
     "탐욕 결과로 층화한 우리 표본에서 초반의 실패는 대체로 시도가 아니라 게임의 성질이었고,")
both("ko", "rep", "프로브로 실패할 episode를 멈추는 것이 고정 step 상한보다 step을 더 아낀다는 것은 보이지 않았다.",
     "성공 손실 10% 이하에서 프로브로 실패할 episode를 멈추는 것이 고정 step 상한보다 step을 더 아낀다는 것은 보이지 않았다.")
both("ko", "after", "(18개, 우수 0개)",
     " 이 비교들은 게임과 baseline을 공유해 독립이 아니므로, 개수는 합친 추정치가 아니다.")
COMP_KO = ("**보완적 해석.** 75% 깊이에서 토큰 하나를 읽고 손으로 만든 특징 없이 라운드당 약 250개 episode로 학습한 프로브는, 앞 250개 episode에서 공들여 만든 5단 baseline과 AUROC 0.011–0.034 차이 안에 있었고, "
           "홀드아웃의 step 0과 1에서는 유의하지 않지만 앞섰다. 실패와 관련된 정보의 상당 부분이 표현 안에 선형으로 들어 있다는 뜻이다. 학습 데이터를 늘리면 격차가 줄거나 뒤집히는지는 시험하지 않았다.\n\n")
rep("ko_md", "**실용적 함의.**", COMP_KO + "**실용적 함의.**")
rep("ko_tex", "\\paragraph{실용적 함의.}", tex(COMP_KO) + "\\paragraph{실용적 함의.}")
LIM_KO = ("\n- **비교 설계.** TF-IDF와 로지스틱 회귀는 약한 텍스트 baseline이며, 같은 프롬프트를 읽는 신경망 텍스트 인코더는 시험하지 않았다. "
          "프로브는 [3]처럼 생성한 action 토큰이 아니라 프롬프트 마지막 토큰을 읽는다. step 7부터는 4단의 진행도 특징(조기 중단 분석의 4t·5t단도)이 프롬프트에 보이는 최근 6 step보다 오래된 action을 쓴다."
          "\n- **지표와 통계.** AUROC는 중단에 중요한 낮은 오탐 구간을 겨냥하지 않으며, 낮은 오탐률에서의 검출률은 보고하지 않았다. 초반 상태로 도달 가능한 AUROC(상한)는 추정하지 않았다. "
          "구간 포함률은 시도 수준에서만 점검했고, 교차검증 seed 사이의 분산은 구간에 들어 있지 않다. 사전 등록·사전 지정·확증 비교 18개는 게임과 baseline을 공유해 독립이 아니다."
          "\n- **등록.** 계획은 외부 등록처가 아니라 프로젝트 저장소의 날짜 있는 commit으로 등록했다. 벗어난 점과 이른 노출은 부록 표 A8에 적었다.")
PH_KO = "마진은 게임 간 AUROC 기준으로 정한 것이라 시도 수준에서는 척도가 다르다."
rep("ko_md", PH_KO, PH_KO + LIM_KO)
rep("ko_tex", PH_KO, PH_KO + tex(LIM_KO))

# ======================= Table A8 (deviations), md and full tex =======================
A8_ROWS = [
    ("Task-level freeze", "Sample size raised from 250 to 500 episodes to create the confirmatory hold-out; policy, prompt and code unchanged",
     "2026-10-15, before the hold-out was collected", "Hold-out analysed as one separate comparison"),
    ("Task-level freeze", "Repository history rewritten to clean commit messages; the frozen commit hash changed (tree identical, mapping kept)",
     "2026-10-15", "None on data or code; registration rests on dated commits and the mapping"),
    ("Confirmatory comparison", "Non-inferiority margin 0.05 chosen after the exploratory analysis",
     "Before the hold-out was evaluated", "Non-inferior labels are weak (Table 2 note)"),
    ("Attempt level", "Games stratified by greedy outcome, decided after the pilot",
     "Before the main set was collected", "Variance explained by the game is not a population value"),
    ("Attempt level", "Supplementary analyses and interpretation rules added (equivalence band, near-0.5 rule, Holm adjustment)",
     "2026-10-17, before the results were opened", "Primary verdicts reported unchanged; rules applied in addition"),
    ("Attempt level", "Another session saw the first two lines of the results log (run size and the eta-squared point estimate)",
     "Before the attempt-level results were opened", "Reported; no other result was seen"),
    ("100-action re-run", "Interim counts seen twice during collection (3 late successes among the first 116 games); a partial collection was discarded after a shutdown",
     "2026-10-17, before the decision", "Decision rule unchanged (committed beforehand)"),
    ("Early termination", "Loop-detector definition clarified, then corrected so that the count excludes the action itself",
     "2026-10-20, before code was run and results opened", "None on the other rules"),
    ("Model extension", "Inclusion rule changed so that every model passing the pilot criteria enters",
     "2026-10-20, before the SmolLM3 pilot", "Substitutes cannot be chosen after results; no further models added"),
    ("Model pilots", "Reference time of the time criterion was not defined; judged against a reference value",
     "2026-10-17, after the first pilots", "Exclusions rest on the success criterion"),
    ("Second prompt (ReAct)", "Methods of the secondary success-rate comparison (Wilson CI, game-level bootstrap, exact McNemar) were not in the plan",
     "2026-10-20, fixed in the script before running", "Secondary result only"),
]
md_tab = "\n| Table A8. Deviations from the registered plans and exposures to results before a decision | Change or exposure | When | Effect on reported results |\n|---|---|---|---|\n"
md_tab += "".join(f"| {a} | {b} | {c} | {d} |\n" for a, b, c, d in A8_ROWS)
NOTE_EN = "Compiled from the change logs of the plan documents; items already described in the text are repeated here so that all deviations appear in one place.\n"
NOTE_KO = "계획 문서들의 변경 이력에서 모았다. 본문에 이미 적은 항목도 한곳에서 보이도록 다시 실었다.\n"
rep("en_md", "\n## References", md_tab + "\n" + NOTE_EN + "\n## References")
rep("ko_md", "\n## 참고문헌", md_tab + "\n" + NOTE_KO + "\n## 참고문헌")
tex_tab = ("\n\\begin{table}[h]\n\\centering\\small\n\\caption{Deviations from the registered plans and exposures to results before a decision.}\\label{tab:a8}\n"
           "\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
           "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}p{2.1cm}L>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{2.8cm}@{}}\n"
           "\\toprule\nPlan & Change or exposure & When & Effect on reported results \\\\\n\\midrule\n")
ETA, CONF = "$\\eta^2$", "Table~\\ref{tab:confirm} note"
rows_tex = []
for a, b, c, d in A8_ROWS:
    rows_tex.append(tex(a) + " & " + tex(b).replace("eta-squared", ETA) + " & " + tex(c) + " & " + tex(d).replace("Table 2 note", CONF) + " \\\\\n")
tex_tab += "\\addlinespace\n".join(rows_tex)
tex_tab += "\\bottomrule\n\\end{tabularx}\n\\par\\smallskip\\raggedright\\footnotesize\n"
END = "\n\\end{document}\n"
for k, note in (("en_tex", NOTE_EN), ("ko_tex", NOTE_KO)):
    assert T[k].endswith(END), k
    T[k] = T[k][: -len(END)] + "\n" + tex_tab + note.strip() + "\n\\end{table}\n" + END
# md table uses "eta-squared" in words; keep the same words in md
# ======================= short versions =======================
S = "en_short"
rep(S, "In the pre-registered comparison the probe was below the full baseline", "In the pre-registered comparison (probe at 50\\% depth) the probe was below the full baseline")
rep(S, "Used to stop failing episodes, the probe was not shown to save more actions than a fixed limit.",
    "Used to stop failing episodes with at most 10\\% of successes lost, the probe was not shown to save more actions than a fixed limit.\nWe found no gain from hidden states; the held-out intervals exclude gains above about 0.10 AUROC, but smaller gains cannot be ruled out.")
rep(S, "whereas under the same approximation a 0.5B monitor would add about 13--16\\% to a 3B actor.",
    "and is 15--52 times cheaper per token than those policies, whereas a separate 0.5B monitor that reads every token a 3B actor reads would add about 13--16\\% of the actor's compute.")
rep(S, "ALFWorld served only as a boundary test (probe AUC 0.815 with Qwen-2.5-7B).",
    "ALFWorld served only as a boundary test (probe AUC 0.815 with Qwen-2.5-7B). Their probe reads the last token of the generated action; ours reads the last prompt token before the candidates are scored.")
rep(S, "the 3B reader reproduced the actor's hidden states exactly.",
    "the 3B reader reproduced the actor's hidden states exactly. Because this hidden state is a deterministic function of the prompt, the question is whether the model's computation makes failure-relevant features linearly readable beyond what the visible-information baselines extract.")
rep(S, "800 rollouts in total.", "800 rollouts in total. These games were drawn from the 500 greedy games and can overlap with the first 250; at the attempt level all methods were cross-fitted on the sampled rollouts only.")
rep(S, "(95\\% CI of the AUROC difference above zero).",
    "(95\\% CI of the AUROC difference above zero). The 50\\% depth was fixed without a recorded rationale, and the plan did not state whether H1 required all three rounds; all three differences fell below zero.")
rep(S, "All registered items and interpretation tables are in Appendix Table~\\ref{tab:a7}.",
    "All registered items and interpretation tables are in Appendix Table~\\ref{tab:a7}, and deviations from the plans in Table~\\ref{tab:a8}.")
rep(S, "adds one dot product with a 2,048-dimensional vector per step. We did not measure energy or latency.",
    "adds one dot product with a 2,048-dimensional vector per step. This assumes that a separate monitor reads every token the actor reads; the lower overall figure in~\\cite{wang2026failfast} also depends on how many tokens its monitor reads. We did not measure energy or latency.")
rep(S, "(within-game permutation $p \\le 0.018$),", "(within-game permutation $p \\le 0.018$; the bootstrap interval of rung 4 at step 3, 0.492--0.644, includes 0.5),")
rep(S, "the primary difference was $-0.032$ ($-0.183$ to $+0.119$), inconclusive.",
    "the primary difference was $-0.032$ ($-0.183$ to $+0.119$), inconclusive. This holds for the pre-specified range; over the whole loss range the probe saved more than the cap ($+0.056$) but less than rung 5t.")
rep(S, "and probes on the running mean of hidden states were not better either.",
    "and probes on the running mean of hidden states were not better either. From step 7 on, rung 4 uses actions older than the six steps shown in the prompt.")
rep(S, "our results suggest that such comparisons overstate what hidden states add, since a ladder that sees the task closed the gap. We did not reproduce their environments.",
    "our results suggest that such comparisons may overstate what hidden states add, since a ladder that sees the task closed the gap; we did not reproduce their environments, and our probe reads a different position.")
rep(S, "\\paragraph{Practical implication.}",
    "\\paragraph{A complementary reading.} Read at 75\\% depth from a single token and trained on about 250 episodes per round, the probe came within 0.011--0.034 AUROC of the engineered rung-5 baseline, and on the hold-out it was ahead at steps 0 and 1, though not significantly. Much of the failure-relevant information is linearly present in the representation; whether more training data would close or reverse the gap we did not test.\n\n\\paragraph{Practical implication.}")
rep(S, "Several design choices changed conclusions in ways that matter for similar studies. Splitting by game instead of by floor plan inflated a scene-only synthetic feature to an AUROC of 0.97.",
    "Several design choices can change conclusions in similar studies. Splitting by game instead of by floor plan inflated a scene-only synthetic feature to an AUROC of 0.97 (not measured with the real probe).")
rep(S, "The main prompt was written by us and adjusted on a different split before data collection; an external prompt was tested separately.",
    "The main prompt was written by us and adjusted on a different split before data collection; an external prompt was tested separately.\n"
    "TF-IDF is a weak text baseline and neural text encoders were not tested; AUROC does not target the low false-positive region that matters for stopping; the attainable AUROC from early states was not estimated; the 18 comparisons share games and baselines and are not independent; readers differ in hidden size (896 to 3,584 dimensions); and plans were registered as dated repository commits, not with an external registry (deviations in Appendix Table~\\ref{tab:a8}).")
rep(S, "and using them to stop doomed episodes was not shown to save more steps than a fixed limit.",
    "and using them to stop doomed episodes was not shown to save more steps than a fixed limit at up to 10\\% lost successes.")
rep(S, "\\input{floats_en/tab_a7}\n", "\\input{floats_en/tab_a7}\n\\input{floats_en/tab_a8}\n")

S = "ko_short"
rep(S, "사전 등록한 비교에서 프로브는", "사전 등록한 비교(50\\% 깊이 프로브)에서 프로브는")
rep(S, "실패할 episode를 멈추는 데 쓰면, 프로브가 고정 상한보다 action을 더 아낀다는 것은 보이지 않았다.",
    "성공 손실 10\\% 이하에서 실패할 episode를 멈추는 데 쓰면, 프로브가 고정 상한보다 action을 더 아낀다는 것은 보이지 않았다.\nhidden state의 이득은 찾지 못했고, 홀드아웃 신뢰구간은 약 0.10 AUROC를 넘는 이득을 배제하지만 그보다 작은 이득은 배제하지 못한다.")
rep(S, "같은 근사로 0.5B 모니터는 3B actor에 약 13--16\\%를 더한다.",
    "토큰당으로는 그 policy들보다 15--52배 싸다. 반면 3B actor가 읽는 토큰을 모두 읽는 별도 0.5B 모니터는 actor 계산량의 약 13--16\\%를 더한다.")
rep(S, "ALFWorld는 경계 시험으로만 쓰였다(Qwen-2.5-7B에서 프로브 AUC 0.815).",
    "ALFWorld는 경계 시험으로만 쓰였다(Qwen-2.5-7B에서 프로브 AUC 0.815). 이들의 프로브는 생성한 action의 마지막 토큰을 읽고, 우리 프로브는 후보를 점수화하기 전 프롬프트의 마지막 토큰을 읽는다.")
rep(S, "3B reader는 actor의 hidden state를 그대로 재현했다.",
    "3B reader는 actor의 hidden state를 그대로 재현했다. 이 hidden state는 프롬프트의 결정적 함수이므로, 질문은 모델의 계산이 실패와 관련된 특징을 겉보기 정보 baseline이 뽑는 것 이상으로 선형으로 읽히게 만드는지이다.")
rep(S, "모두 800개 rollout이다.", "모두 800개 rollout이다. 이 게임들은 탐욕 게임 500개에서 뽑아 앞 250개와 겹칠 수 있고, 시도 수준에서 모든 방법은 샘플링 rollout만으로 교차적합했다.")
rep(S, "(AUROC 차이의 95\\% CI가 0 위).",
    "(AUROC 차이의 95\\% CI가 0 위). 50\\% 깊이는 근거를 기록하지 않은 채 정했고, H1이 세 라운드 모두를 요구하는지는 계획에 없었으나 세 차이가 모두 0 아래였다.")
rep(S, "등록한 항목과 해석표는 부록 표~\\ref{tab:a7}에 있다.", "등록한 항목과 해석표는 부록 표~\\ref{tab:a7}에, 계획에서 벗어난 점은 표~\\ref{tab:a8}에 있다.")
rep(S, "2,048차원 벡터와의 내적 하나를 더할 뿐이다. 에너지와 지연은 측정하지 않았다.",
    "2,048차원 벡터와의 내적 하나를 더할 뿐이다. 이는 별도 모니터가 actor가 읽는 토큰을 모두 읽는다고 가정한 것이며, \\cite{wang2026failfast}의 더 낮은 전체 비용은 모니터가 읽는 토큰 수에도 달려 있다. 에너지와 지연은 측정하지 않았다.")
rep(S, "(게임 안 순열 $p \\le 0.018$),", "(게임 안 순열 $p \\le 0.018$; step 3에서 4단의 bootstrap 구간 0.492--0.644는 0.5를 포함),")
rep(S, "주 비교 차이는 $-0.032$($-0.183$ \\textasciitilde{} $+0.119$)로 결론 없음이었다.",
    "주 비교 차이는 $-0.032$($-0.183$ \\textasciitilde{} $+0.119$)로 결론 없음이었다. 이는 사전에 정한 범위에 대한 것이며, 손실 전체 범위에서는 프로브가 상한보다 더 아꼈지만($+0.056$) 5t보다는 덜 아꼈다.")
rep(S, "이전 step hidden state의 누적 평균을 쓴 프로브도 낫지 않았다.",
    "이전 step hidden state의 누적 평균을 쓴 프로브도 낫지 않았다. step 7부터는 4단이 프롬프트에 보이는 최근 6 step보다 오래된 action을 쓴다.")
rep(S, "우리 결과는 과제를 보는 사다리가 그 차이를 메웠으므로 그런 비교가 hidden state의 기여를 과대평가할 수 있음을 시사한다. 그들의 환경은 재현하지 않았다.",
    "우리 결과는 과제를 보는 사다리가 그 차이를 메웠으므로 그런 비교가 hidden state의 기여를 과대평가할 수 있음을 시사한다. 다만 그들의 환경은 재현하지 않았고 프로브가 읽는 위치도 다르다.")
rep(S, "\\paragraph{실용적 함의.}",
    "\\paragraph{보완적 해석.} 75\\% 깊이에서 토큰 하나를 읽고 라운드당 약 250개 episode로 학습한 프로브는 공들여 만든 5단 baseline과 AUROC 0.011--0.034 차이 안에 있었고, 홀드아웃의 step 0과 1에서는 유의하지 않지만 앞섰다. 실패와 관련된 정보의 상당 부분이 표현 안에 선형으로 들어 있다는 뜻이며, 학습 데이터를 늘리면 격차가 줄거나 뒤집히는지는 시험하지 않았다.\n\n\\paragraph{실용적 함의.}")
rep(S, "몇몇 설계 선택은 비슷한 연구에서 결론을 바꿀 만큼 중요했다. 방 배치가 아니라 게임 단위로 나누면 장면 정보만 담은 합성 특징의 AUROC가 0.97까지 부풀었다.",
    "몇몇 설계 선택은 비슷한 연구에서 결론을 바꿀 수 있다. 방 배치가 아니라 게임 단위로 나누면 장면 정보만 담은 합성 특징의 AUROC가 0.97까지 부풀었다(실제 프로브로는 재지 않음).")
rep(S, "주 프롬프트는 직접 작성해 데이터 수집 전에 다른 분할에서 고쳤고, 외부 프롬프트는 따로 시험했다.",
    "주 프롬프트는 직접 작성해 데이터 수집 전에 다른 분할에서 고쳤고, 외부 프롬프트는 따로 시험했다.\n"
    "TF-IDF는 약한 텍스트 baseline이며 신경망 텍스트 인코더는 시험하지 않았다. AUROC는 중단에 중요한 낮은 오탐 구간을 겨냥하지 않고, 초반 상태로 도달 가능한 AUROC는 추정하지 않았다. 비교 18개는 게임과 baseline을 공유해 독립이 아니며, reader마다 hidden 차원이 다르다(896--3,584). 계획은 외부 등록처가 아니라 저장소의 날짜 있는 commit으로 등록했다(벗어난 점은 부록 표~\\ref{tab:a8}).")
rep(S, "프로브로 실패할 episode를 멈추는 것이 고정 상한보다 step을 더 아낀다는 것은 보이지 않았다.",
    "성공 손실 10\\% 이하에서 프로브로 실패할 episode를 멈추는 것이 고정 상한보다 step을 더 아낀다는 것은 보이지 않았다.")
rep(S, "\\input{floats_ko/tab_a7}\n", "\\input{floats_ko/tab_a7}\n\\input{floats_ko/tab_a8}\n")

if "--dry" in sys.argv:
    print("dry run ok")
    sys.exit(0)
for k, v in F.items():
    io.open(R + v, "w", encoding="utf-8", newline="\n").write(T[k])
    print("wrote", v)
