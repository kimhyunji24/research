# 인수인계 (2026-10-15)

## 지금 상태
- 브랜치 `probes-baseline` (main에는 아직 합치지 않았다). 동결 기록: `docs/freeze_2026-10-15.md`.
- **본 수집 완료 (2026-10-15 17:52 KST, 약 47분)**: 3B 250 episode, 코드 commit `89058c2`.
  WSL `~/research/data/main_3b/Qwen2.5-3B-Instruct/`, 같은 내용을 Windows `C:\research\data\main_3b\`에도 복사했다(git 제외).
  `data/decision_rerun`, `data/maxstep50`도 같은 곳에 복사했다. 이어서 수집이 필요하면 `--resume`.
- WSL: Ubuntu-22.04, 저장소 `~/research`, venv `.venv`, 게임 데이터 `/root/alfworld_data`
  (`export ALFWORLD_DATA=/root/alfworld_data`). 기본 사용자가 root다.

## 이번에 확인한 것 (표본 수와 한계 포함)
| 항목 | 결과 | 한계 |
|---|---|---|
| 3B max_steps 50, 20 episode (10/8 프롬프트 v2) | 성공 5/20. 30 step 이내 미성공 15개 중 31~50 step 성공 0개 (95% CI 0~20%) | 표본 작음 |
| 3B decision.yaml 재수집, 100 episode | 성공 26/100 (95% Wilson CI 18~35%), 실패 74. 성공 episode는 4~28 step | 이전 20/100과 같은 episode인지 확인 필요 |
| probes 예비 실행 (50% layer, step 1~3, n=100, 실패 74) | baseline3 AUROC 0.60/0.63/0.72, linear 0.55/0.66/0.66(CI 약 ±0.13) | **해석 불가**: 셔플 대조군이 0.33~0.41로 0.5가 아니다 |
| **본 수집 3B, 250 episode** (`main_3b`) | 성공 56/250 (22.4%, 95% Wilson CI 17.7~28.0%), 실패 194. 서로 다른 250 게임. 성공 episode는 4~28 step(중앙값 12). steps 6,490행, hidden (6490, 4, 2048) float16, hid_idx 정합, NaN 없음 | 성공 목표 약 50개는 달성. task type별 성공: look 10/23, pick_place 17/48, clean 13/59, cool 8/25, heat 4/36, two_obj 4/59 (heat, two_obj는 표본 작음) |
| 라운드별 n (`main_3b`, 0-based step 0~3) | 각 250, 실패 194, 양성(실패) 비율 0.776. step 4는 n=241 | 성공은 모두 4 step 이상이라 step 0~3은 전원 생존 |
| `main_3b`의 처음 100 episode vs `decision_rerun` | 게임 순서와 성공 여부 모두 동일 | 같은 머신에서 2회 실행했을 때의 결정성만 확인. 두 데이터는 독립 표본이 아니다 |
| `score_norm: sum`의 길이 선호 (`main_3b`, 6,490 step) | 선택된 후보의 평균 토큰 수 8.98, 전체 후보 평균 8.73. 선택된 후보보다 짧은 후보의 평균 비율 0.36(0.5면 순위상 선호 없음). 가장 짧은 후보가 선택된 비율 0.23. mean 정규화로 바꾸면 12.7%의 step에서 다른 action이 선택됨 | 짧은 action을 체계적으로 선호한다는 증거는 이 데이터에서 보이지 않는다. 다만 후보 구성(go to 위주)과 task 진행의 영향을 분리하지 못했다 |
| **baseline 1~5단, `main_3b`** (50% layer 불필요, hidden 미사용, n=250, 실패 194, 장면 86개, 장면 단위 5-fold x2 + nested C, 순열 100회) | step 0: 0.500 / 0.603 / 0.659 / 0.673 / 0.701. step 1: 0.500 / 0.603 / 0.633 / 0.700 / 0.732. step 2: 0.500 / 0.603 / 0.671 / 0.834 / 0.823. step 3: 0.500 / 0.603 / 0.700 / 0.814 / 0.818 (1~5단 순). 2~5단의 95% CI(장면 cluster bootstrap)는 약 ±0.07~0.08, 순열 null 평균 0.50(95% 구간 약 0.38~0.62), p_perm 0.01~0.03 | 프로브가 이겨야 할 기준선이 step 0에서 이미 0.70이다(과제·방 묘사 텍스트만으로). 4단의 진행도가 step 2부터 0.83으로 급상승. 5단이 4단보다 낮은 라운드가 있다(과적합 가능). `results/baselines_main_3b.csv` |
| 순열 대조군 보정 | 다회(100) 순열 평균이 0.498~0.507로 0.5. 예비 실행의 0.33~0.41은 단일 순열 공유의 우연으로 보이나 확정은 아님 | 예비 실행과 절차가 달라(장면 분할, nested C) 같은 조건에서 재현해 본 것은 아니다 |
| Qwen2.5 0.5B/1.5B/3B/7B 토크나이저, chat template | 샘플 프롬프트에서 동일 (vocab 151,665) | 실제 프롬프트 전체 검증은 재추출 스크립트에서 |
| 저장한 prompt를 3B에 다시 통과 | 마지막 토큰 hidden이 저장값과 일치(5 step, diff 0.0) | 다른 크기는 미확인 |

`results/probes_decision_rerun.csv`는 위 probes 예비 실행 전체 출력이다. **연구 결과가 아니라 코드 점검용**이다.

## 코드 점검에서 나온 문제 (미해결)
1. `score_norm: sum`의 길이 선호: `main_3b`에서 확인했고 체계적 선호는 보이지 않았다(위 표). 동결된 설정이므로 바꾸지 않는다.
   다만 mean 정규화와 12.7%의 step에서 다른 action을 고르므로, 논문에는 sum 정규화를 사용했다고 명시한다.
2. 설계 문서 4절의 "episode당 forward 한 번"은 틀렸다. 프롬프트가 step마다 달라서 reader는 **step당 prefill 한 번**이다.
3. (수정 완료, 2026-10-15) `probes/`: 장면 단위 StratifiedGroupKFold, nested C, 순열 대조군 다회 반복, explained variance 출력,
   k가 fold 표본 수 이상이면 건너뛰기, 장면 cluster bootstrap과 짝지은 차이 CI. 계획은 `docs/analysis_plan_2026-10-15.md`.
   **hidden state 프로브는 아직 `main_3b`에 돌리지 않았다**(라운드 기준 확정 후).
4. "라운드 1"의 기준(첫 번째 action이 step 0)을 정하지 않았다. 사전 등록 전 확정 필요.
5. teacher-forced 재추출 스크립트(0.5B/1.5B/3B, 선택 7B 4-bit)는 아직 없다. baseline 4~5단은 구현했다
   (목표 물체 파싱 성공 96%. 4단의 `obj_seen`은 프롬프트 창 6 step 안에서만 확인 가능).

## 다음 순서 (설계 문서 12절 기준)
1. (완료) `main_3b` 수집: 성공 56개.
2. teacher-forced 추출 스크립트 작성: `steps.jsonl`의 `prompt`를 reader 모델에 통과, 마지막 토큰의 25/50/75/100% layer 저장.
   prompt 토큰 수가 `prompt_tokens`와 같은지 검증한다.
3. **사용자 결정**: "라운드 1"의 기준(첫 번째 action이 step 0인지)을 정한 뒤, 3B reader(저장된 hidden)로
   `python probes/run_probes.py data/main_3b/Qwen2.5-3B-Instruct --layer 1 --rounds <정한 step들>`을 한 번 실행.
4. (완료) baseline 4~5단 구현과 `main_3b` 실행.
5. 선택: 1.5B 궤적 수집(탐색적 H5), 7B 4-bit reader.

## 규칙 (사용자 지시)
- 결과를 과대 해석하지 않고 표본 수와 신뢰구간을 같이 적는다. 불확실한 것은 "확인 필요"로 적는다.
- 기존 `data/`를 덮어쓰지 않는다. 새 출력은 새 폴더.
- commit은 사용자가 확인한 뒤에 한다(2026-10-15에 "모든 것을 저장·commit·push" 지시가 있었음).

## 주 지표 1회 실행 결과 (2026-10-15, 라운드 정의 확정 후 한 번만 실행)
조건: 3B 궤적, 3B reader(저장된 hidden), 50% layer(3B layer 18), linear(표준화+L2, nested C), PCA 없음, 장면 단위 5-fold x2, 순열 100회.
n=250, 실패 194(양성 비율 0.776), 장면 86개. `results/probe_main_3b_primary*.csv`.

| 라운드(step) | 프로브 AUROC (95% CI) | p_perm | baseline 5단 | 프로브 - baseline 5단 (95% CI) |
|---|---|---|---|---|
| 1 (step 0) | 0.520 (0.44~0.61) | 0.31 | 0.701 | -0.181 (-0.276 ~ -0.069) |
| 2 (step 1) | 0.599 (0.52~0.68) | 0.04 | 0.732 | -0.133 (-0.230 ~ -0.031) |
| 3 (step 2) | 0.665 (0.58~0.75) | 0.01 | 0.823 | -0.159 (-0.240 ~ -0.073) |

- 사전 등록한 비교(프로브 vs baseline 5단)에서 세 라운드 모두 프로브가 낮고 CI가 0을 포함하지 않는다. H1은 이 설정에서 지지되지 않았다.
  설계 문서 8절의 해석 표로는 "3B 읽기도 5단을 못 이김"에 해당한다.
- 제한: linear 하나, 50% layer 하나, n=250(양성 56개 중 성공 클래스는 소수)이다. "정보가 없다"가 아니라
  "이 설정의 선형 프로브가 텍스트+진행도 baseline보다 못하다"까지만 말할 수 있다. 다른 layer, MLP, PCA는 탐색적이며 아직 실행하지 않았다.
- `C_median`이 라운드 2에서 격자 끝(1.0)에 있어 격자가 충분히 넓지 않을 수 있다(확인 필요).
- 짝지은 비교는 baseline 5단 대비다. 5단이 4단보다 낮은 라운드가 있어 baseline 중 최고(4단)와 비교하면 차이가 더 크다.

## 탐색적 추가 분석 결과 (2026-10-15, 새 수집 없음, 앞 250 episode, 3B 궤적, linear, nested C)
`results/extra_main_3b_{A,B,C}_*.csv`, 요약 `results/summarize_extra.py`. 모두 탐색적이며 확증 근거가 아니다.

- **A. reader 간 짝지은 차이(H2)**: 6쌍 x step 3 x layer 2 = 36개 비교 중 장면 cluster CI가 0을 벗어난 것은 4개뿐이고(우연 기대 약 2개),
  부호가 일관되지 않는다. 75% layer에서 3B−0.5B가 step 0(+0.095), step 2(+0.119)로 0.5B가 낮고, 0.5B−1.5B/7B도 step 2에서 약 −0.12.
  50% layer에서는 모든 쌍의 CI가 0을 포함하고, 3B−7B는 step 0/1에서 −0.08(7B가 더 높음)로 75% layer(+0.085)와 반대다. 단조로운 크기 효과의 증거는 없다.
  7B는 4-bit 양자화 교락이 있다.
- **B. 후반 라운드(step 0~10)**: 프로브(3B, 75%, linear) - baseline 5단 차이는 -0.034~+0.009이고 CI가 0을 포함한다(step 9만 -0.114, CI -0.219~-0.023).
  TPR@10%FPR은 프로브가 대체로 낮다(step 0 0.29 vs 0.42, step 5 0.44 vs 0.55). 살아남은 episode는 성공으로 끝나지 않은 것이라 후반 라운드의 양성 비율이 올라가고(0.78 -> 0.87) 선택 효과가 있다.
- **C. 시간축**: 이전 step hidden의 누적 평균을 더해도 개선이 없다(current+cummean ≈ current, cummean만은 더 낮음). baseline 5단 대비 차이는 -0.05~+0.01, CI가 0을 포함한다.
- 결론: 75% layer 선형 프로브는 라운드 0~10 전반에서 baseline 5단과 구분되지 않거나 약간 낮고, 이긴 곳은 없다. reader 크기와 시간축 정보도 뚜렷한 이득이 없다.

## 다중 rollout 파일럿 결과 (2026-10-15, 20게임 x 8회 = 160 episode, T=1.0, 층화: 탐욕 성공 10 + 실패 10)
`results/pilot_multi_rollout_summary.txt`, 게임 목록 `results/games_pilot*.json`. 프로브는 돌리지 않았다.
- 성공 66/160 = 41.2% (95% CI 33.9~49.0%). 탐욕 결과로 층화한 표본이라 전체 성공률과 비교할 수 없다.
- 성공과 실패가 모두 나온 게임 8/20 = 40%(판정 규칙 30% 이상 충족). 8회 모두 성공 4, 모두 실패 8.
- 탐욕 성공 게임 10개의 샘플링 성공률 64/80 = 0.80(갈리는 게임 6개), 탐욕 실패 게임 10개는 2/80 = 0.025(갈리는 게임 2개).
  즉 결과의 대부분은 게임이 정하고 게임 내 변동은 일부 게임에서만 있다.
- 같은 게임 안의 서로 다른 경로 비율 평균 0.80. 소요 약 22분(episode당 약 8초).
- 본 규모 예상: 100게임이면 갈리는 게임 약 40개, 쌍 수는 게임당 (실패 수 x 성공 수)로 제한된다.
