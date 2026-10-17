# [Title: write last — no colon, name ALFWorld/Qwen2.5 and the two evaluation levels]

<!-- Draft started 2026-10-17. Target (changed 2026-10-17): main conference or journal paper; venue TBD. Extra experiments needed: plan section 7. Plan: docs/paper_revision_plan_2026-10-17.md.
     Writing order (per plan): Method -> Results -> Discussion -> Conclusion -> Introduction -> Abstract -> Title.
     [TODO-RESULT] marks text that depends on the within-game analysis, which had not been read when this draft was written. -->

## Abstract
[TODO-RESULT: one paragraph, written last. What was done + main results with 95% CIs. No abbreviations, no citations.]

## 1 Introduction
[TODO: write after Results/Discussion. Known facts -> open question (task difficulty vs. in-progress failure) -> what differs here -> aim and hypotheses.]

## 2 Related Work
[TODO: verify sources in docs/literature_check_2026-10-15.md against the originals before citing numbers.]

## 3 Method

### 3.1 Environment and agent
We used ALFWorld (TextWorld version, `alfworld` 0.4.2) with the 3,553 games of the training split and all six task types.
An episode ended when the task was completed or after 30 actions; episodes that did not complete the task were labeled as failures (label 1).
The agent (actor) was Qwen2.5-3B-Instruct in bfloat16. At each step the prompt contained a fixed system message (rules and one worked example episode),
the task sentence, the last six observation–action pairs, the current observation, the actions that had recently produced no effect, and the list of admissible actions.

The actor selected actions by constrained scoring rather than free generation. Each admissible action *a* (excluding `help`, `inventory` and `look`)
was scored by the summed log-probability of the continuation `Action: a<|im_end|>` given the prompt, computed in one batched pass that reuses the prompt's key–value cache.
In the greedy policy the highest-scoring action was taken. In the sampling policy an action was drawn from softmax(score / *T*) with *T* = 1,
i.e. the model's distribution renormalized over admissible actions; the random generator was seeded by (sample seed, episode index, step), so rollouts are reproducible and resumable.
Free generation was not used because small models frequently produced inadmissible actions in preliminary runs (Appendix [TODO]).

### 3.2 Representations: separating the actor from the reader
Hidden states were taken at the last prompt token, immediately before the action is scored, at relative depths of 25, 50, 75 and 100% of the layers
(layers 9, 18, 27 and 36 for the 3B model). To vary model size without changing the trajectories, we passed the actor's stored prompts through *reader* models by teacher forcing:
Qwen2.5-0.5B-, 1.5B- and 3B-Instruct in bfloat16, and Qwen2.5-7B-Instruct with 4-bit NF4 weights (bfloat16 compute), the largest that fits in 12 GB of GPU memory.
All four models share the tokenizer and chat template; the re-tokenized prompt length matched the stored length at every step, and on the steps we checked (five) the 3B reader reproduced the actor's stored hidden states exactly.
Because readers other than 3B read another model's trajectories, results for them concern external monitoring, not self-monitoring.

### 3.3 Data
**Task-level data (greedy).** We collected 500 greedy episodes on distinct games in the environment's deterministic order.
The first 250 episodes were used for the pre-registered and exploratory analyses; episodes 250–499 were collected afterwards as a confirmatory hold-out
for a single comparison fixed before collection (Section 3.6). Reader hidden states were extracted for the first 250 episodes.

**Attempt-level data (sampling).** Under greedy decoding the outcome is a function of the game, so task-level labels cannot separate task difficulty from in-progress failure.
We therefore re-ran 100 games eight times each with the sampling policy: a pilot of 20 games (10 games the greedy policy solved, 10 it failed; sample seed 0)
and a main set of 80 new games (40 solved, 40 failed by the greedy policy, drawn at random with seed 1 from the remaining greedy games; sample seed 1).
Stratifying by the greedy outcome was decided after the pilot and is reported as a selection step. In total 800 rollouts were used.
Re-collecting the first 24 main-set rollouts on a fresh installation reproduced them exactly (actions, outcomes and hidden states).

### 3.4 Evaluation units
**Task level (between games).** For each round *r* we kept one row per episode still running at step *r* and computed the AUROC for predicting failure.
Within a round the step index is constant, which removes episode length as a predictor. The pre-registered rounds 1–3 are steps 0–2 (the first three actions);
every episode was still running at these steps.

**Attempt level (within games).** All rollouts of a game share the step-0 prompt, so step 0 was excluded and steps 1–3 were evaluated.
Only games with at least one success and one failure among their eight rollouts contribute. The metric is the within-game pairwise AUROC:
over all (failed, successful) rollout pairs from the same game, the fraction in which the failed rollout received the higher failure score (ties count one half),
pooled across games with weights proportional to the number of pairs.

Two properties of this design were checked before the attempt-level results were read. First, sampled rollouts of a game often had not diverged yet:
of the 470 evaluation pairs per step (39 games with both outcomes, 28 scenes), 70.4%, 66.4% and 52.3% had identical prompts at steps 1, 2 and 3.
The probe input is then identical, so these pairs are ties and cap the probe's pairwise AUROC at 0.648, 0.668 and 0.739.
Second, rungs 3–5 include the repetition count and the probability of the action sampled at the current step, which lie after the probe's input in time
and are exactly where rollouts diverge. We therefore pre-specified two secondary analyses: (i) the same metric restricted to pairs whose prompts differ,
and (ii) time-aligned rungs 4t and 5t in which those two features are taken from the previous step. Secondary results are reported after, and do not replace, the primary result.

### 3.5 Predictors
**Probe.** Standardization followed by L2-regularized logistic regression with balanced class weights.
The regularization strength *C* ∈ {10⁻⁴, 10⁻³, 10⁻², 10⁻¹, 1} was selected in a nested three-fold cross-validation inside each training fold.
A one-hidden-layer MLP (64 units, α = 10⁻², early stopping, fixed hyperparameters) was evaluated as an exploratory probe.

**Baseline ladder.** Each rung adds information available without hidden states:
(1) step index; (2) + task type; (3) + recent repetition count, probability of the chosen action and number of candidates;
(4) + progress features computed from earlier actions (distinct locations visited, open/take/transform/move counts, whether the target object has been seen);
(5) + TF-IDF (word 1–2-grams) of the full user message, which contains the task sentence, room description, recent history, current observation and admissible actions.
Rung 3 uses quantities computed after scoring the current step, slightly later than the probe's input; this favors the baseline.
The probe is compared with rung 5 (pre-registered) and, at the attempt level, with rungs 4 and 5.

**Difficulty oracle (attempt-level analysis).** Each game's greedy outcome from the task-level data, which is independent of the sampled rollouts.
It is constant within a game, so its within-game AUROC is 0.5 by construction (verified as an implementation check).
We also report the share of outcome variance explained by the game (η²) and, on rows pooled across games, the AUROC gain of a logistic stack of [oracle, predictor] over the oracle alone.

### 3.6 Cross-fitting, statistics and pre-registration
All predictors were cross-fitted with five-fold stratified group splits by **scene** (the ALFWorld floor plan; 86 scenes in the first 250 games),
repeated with two seeds and averaged, so that every row is scored by a model that never saw its scene. Splitting by game instead inflated AUROC to 0.97
on a synthetic feature that encodes only the scene. Standardization, TF-IDF and model selection were fit inside each training fold.

Task-level 95% confidence intervals use a scene-level cluster bootstrap (1,000 resamples); differences between a probe and a baseline are computed on the same resamples.
The null distribution re-runs the full pipeline, including nested selection, on permuted labels (100 permutations).
Attempt-level intervals use a game-level bootstrap (1,000 resamples), and the null permutes labels within games with scores held fixed (1,000 permutations).

Before any probe was fit we fixed the code commit, configuration and primary analysis (3B actor and reader, 50% depth, linear probe, rounds 1–3, comparison with rung 5),
and ran it once. Every later analysis on the same episodes is labeled exploratory. A single exploratory configuration (75% depth, linear) was then tested on the confirmatory
hold-out, training on all 500 episodes and evaluating only episodes 250–499. The decision rule was: superior if the lower 95% bound of the difference exceeds 0,
non-inferior if it exceeds −0.05, inferior if the upper bound is below 0, otherwise inconclusive. The margin of 0.05 was chosen after the exploratory analysis.
The attempt-level analysis (75% depth, linear, vs. rungs 4 and 5, steps 1–3) and its interpretation table were fixed before the sampled rollouts were collected.
All changes to the frozen procedures are logged with dates in the repository.

### 3.7 Compute
One NVIDIA RTX 4070 (12 GB) under WSL2 (Ubuntu 22.04), Python 3.11, PyTorch 2.11 (CUDA 12.8), Transformers 5.19.0.
Collecting one episode took about 8–11 s; the 640 main-set rollouts took 88 min.

## 4 Results
### 4.1 Task level
**Data.** The greedy actor completed 56 of the first 250 episodes (22.4%, Wilson 95% CI 17.7–28.0%) and 101 of all 500.
The 250 games covered 86 scenes. Successful episodes took 4–28 actions (median 12), so all 250 episodes were still running at steps 0–3,
and the failure rate in every pre-registered round was 0.776.

**Baselines.** Information visible without hidden states already predicted failure before the first action (Table 1, Fig. 2).
Task type alone gave an AUROC of 0.603, and the full rung-5 baseline reached 0.701 at step 0, 0.732 at step 1 and 0.823 at step 2.
Progress features (rung 4) accounted for most of the rise from step 2 onward (0.673 → 0.834); at steps 2 and 3, rung 5 was not higher than rung 4.
The step-index rung was exactly 0.500 in every round, as intended. Permuted-label AUROCs averaged 0.50 (95% range about 0.38–0.62).

**Pre-registered comparison.** In the single pre-registered run (3B reader, 50% depth, linear probe), the probe scored below the rung-5 baseline in all three rounds,
and every 95% interval of the difference lay below zero (Table 1, Fig. 3): −0.181 (−0.276 to −0.069) at step 0, −0.133 (−0.230 to −0.031) at step 1 and
−0.159 (−0.240 to −0.073) at step 2. Hypothesis H1 was therefore not supported in the pre-registered cell; this matches the third outcome of the interpretation table fixed in advance
("even the 3B reader does not beat rung 5").

**Exploratory analyses (same 250 episodes).** We then evaluated all 96 combinations of four readers, four depths, two probe types and three rounds (Fig. 4; Appendix Table [TODO]).
No combination exceeded the baseline by more than 0.004, and the two positive differences (+0.004 and +0.001) had intervals spanning zero.
The 75% depth was the best or near-best depth for every reader; for the 3B reader the linear probe at 75% depth was within −0.034 to −0.011 of the baseline (all intervals include zero).
Averaged over steps 0–2, the best-depth linear AUROC was 0.660 (0.5B), 0.717 (1.5B), 0.727 (3B) and 0.717 (7B, 4-bit; best depth 100%), i.e. no monotonic increase with reader size;
the 7B result is confounded with quantization. Averaged over depths, the fixed MLP was not better than the linear probe (differences −0.035 to −0.004).
Of 36 pairwise reader comparisons, 4 had intervals excluding zero, close to the about 2 expected by chance, and their signs were inconsistent.
These analyses were chosen after seeing the pre-registered result and are not evidence on their own.

**Confirmatory hold-out.** The one configuration carried forward (3B reader, 75% depth, linear) was evaluated on the 250 new episodes
(205 failures, failure rate 0.82; Table 2). The differences from rung 5 were +0.039 (−0.030 to +0.100), +0.035 (−0.030 to +0.096) and −0.029 (−0.087 to +0.029) at steps 0–2.
No round met the criterion for superiority and none was inferior; steps 0 and 1 met the non-inferiority rule, whose margin was set after exploration, and step 2 was inconclusive.
The rung-5 baseline was lower on the hold-out than on the first 250 episodes (0.607 vs. 0.701 at step 0), which indicates sample-to-sample variation of this size.

| Table 1 (draft). Failure prediction AUROC on the first 250 episodes (pre-registered rounds). | step 0 | step 1 | step 2 |
|---|---|---|---|
| Rung 1: step index | 0.500 | 0.500 | 0.500 |
| Rung 2: + task type | 0.603 | 0.603 | 0.603 |
| Rung 3: + repetition, chosen-action probability | 0.659 | 0.633 | 0.671 |
| Rung 4: + progress | 0.673 | 0.700 | 0.834 |
| Rung 5: + TF-IDF of prompt | 0.701 | 0.732 | 0.823 |
| Probe, 3B reader, 50% depth (pre-registered) | 0.520 | 0.599 | 0.665 |
| Probe − rung 5 (95% CI) | −0.181 (−0.276, −0.069) | −0.133 (−0.230, −0.031) | −0.159 (−0.240, −0.073) |

*n* = 250 episodes (194 failures) per round; CIs from a scene-level cluster bootstrap (1,000 resamples). Interval half-widths for single AUROCs were about 0.07–0.09.

| Table 2 (draft). Confirmatory hold-out (episodes 250–499; 3B reader, 75% depth, linear). | step 0 | step 1 | step 2 |
|---|---|---|---|
| Probe AUROC (95% CI) | 0.645 (0.547, 0.736) | 0.716 (0.619, 0.810) | 0.729 (0.633, 0.815) |
| Rung 5 AUROC | 0.607 | 0.681 | 0.758 |
| Probe − rung 5 (95% CI) | +0.039 (−0.030, +0.100) | +0.035 (−0.030, +0.096) | −0.029 (−0.087, +0.029) |
| Decision (margin 0.05, set post hoc) | non-inferior | non-inferior | inconclusive |

*n* = 250 episodes (205 failures). Training used all 500 episodes; evaluation used only the new 250.

Figures (to be produced): Fig. 2 baseline ladder by step with 95% CI bands; Fig. 3 probe − rung 5 differences with 95% CIs, marking pre-registered, exploratory and confirmatory estimates;
Fig. 4 best-depth AUROC by reader size (7B marked as 4-bit).
### 4.2 Attempt level
[TODO-RESULT]
### 4.3 How much the game determines the outcome
[TODO-RESULT: η², oracle.]
### 4.4 Independent replication of the attempt-level result
[TODO: pre-registered replication on new games (sample size from power simulation). Report discovery and replication separately; do not pool.]
### 4.5 Generalization to a second environment and model family
[TODO: one more environment (ScienceWorld / WebShop / BabyAI-text) and one more family (Llama-3.2 / Gemma).]
### 4.6 Comparison with prior failure-prediction and uncertainty methods
[TODO: token log-probability/entropy, verbalized confidence, LLM judge; chosen from docs/literature_check_2026-10-15.md.]
### 4.7 Early termination
[TODO: steps saved vs. successes lost per stopping policy, including later rounds.]
### 4.8 Robustness checks (summary; details in Appendix)
[TODO: divergence-aligned analysis, stronger probes (C grid, PCA, layer combination, MLP search), stronger text baseline, time-aligned baselines, CV-seed variance, scene-level permutation, Holm-adjusted exploratory table, 3B free-generation comparison.]

## 5 Discussion
[TODO-RESULT]

## 6 Limitations
- **Agent setting.** Actions were chosen by scoring admissible candidates, not by free generation, so the agent is a constrained selector. We did not compare with a free-generation agent.
- **Scope.** One environment (ALFWorld), one model family (Qwen2.5), one prompt and one actor size (3B). We make no claim beyond this setting.
- **External monitoring.** Readers other than 3B read trajectories produced by another model. The 7B reader was 4-bit quantized, which is confounded with size.
- **Sample size.** With 250 episodes per evaluation set (about 50 successes), intervals for an AUROC difference were about ±0.06–0.10; small differences and reader-size effects cannot be resolved.
- **Post hoc choices.** The 75% depth and the non-inferiority margin of 0.05 were chosen after the exploratory analysis; the margin was also defined for between-game AUROC and is on a different scale at the attempt level.
- **Policy shift.** Attempt-level results come from a sampling policy (*T* = 1), not the greedy policy used at the task level, and from games selected by their greedy outcome.
- **Early rounds.** Pre-registered rounds cover only the first three actions; later rounds are conditioned on survival and were analyzed only exploratorily.
- **Truncation.** Episodes were cut at 30 actions. Among 15 episodes unsolved at 30 actions, none succeeded by 50 (95% CI 0–20%), which does not rule out an effect of truncation.
- **Prompt tuning.** The prompt was adjusted while inspecting episodes from a different split (valid_unseen) before data collection.

## 7 Conclusion
[TODO-RESULT: 3–4 sentences, no numbers.]

## References
[TODO]
