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
| Qwen2.5 0.5B/1.5B/3B/7B 토크나이저, chat template | 샘플 프롬프트에서 동일 (vocab 151,665) | 실제 프롬프트 전체 검증은 재추출 스크립트에서 |
| 저장한 prompt를 3B에 다시 통과 | 마지막 토큰 hidden이 저장값과 일치(5 step, diff 0.0) | 다른 크기는 미확인 |

`results/probes_decision_rerun.csv`는 위 probes 예비 실행 전체 출력이다. **연구 결과가 아니라 코드 점검용**이다.

## 코드 점검에서 나온 문제 (미해결)
1. `score_norm: sum`의 길이 선호: `main_3b`에서 확인했고 체계적 선호는 보이지 않았다(위 표). 동결된 설정이므로 바꾸지 않는다.
   다만 mean 정규화와 12.7%의 step에서 다른 action을 고르므로, 논문에는 sum 정규화를 사용했다고 명시한다.
2. 설계 문서 4절의 "episode당 forward 한 번"은 틀렸다. 프롬프트가 step마다 달라서 reader는 **step당 prefill 한 번**이다.
3. `probes/` 수정 필요: 셔플 대조군을 여러 번 반복해 null 분포 보고, explained variance 출력, baseline과 프로브의 `C` 통일(또는 nested CV),
   k가 fold 표본 수 이상이면 건너뛰기.
4. "라운드 1"의 기준(첫 번째 action이 step 0)을 정하지 않았다. 사전 등록 전 확정 필요.
5. baseline 4단(진행도 추출기), 5단(TF-IDF), teacher-forced 재추출 스크립트(0.5B/1.5B/3B, 선택 7B 4-bit)는 아직 없다.

## 다음 순서 (설계 문서 12절 기준)
1. (완료) `main_3b` 수집: 성공 56개.
2. teacher-forced 추출 스크립트 작성: `steps.jsonl`의 `prompt`를 reader 모델에 통과, 마지막 토큰의 25/50/75/100% layer 저장.
   prompt 토큰 수가 `prompt_tokens`와 같은지 검증한다.
3. `probes/` 수정(위 3번) 후 reader별 라운드별 AUROC.
4. baseline 4~5단 구현.
5. 선택: 1.5B 궤적 수집(탐색적 H5), 7B 4-bit reader.

## 규칙 (사용자 지시)
- 결과를 과대 해석하지 않고 표본 수와 신뢰구간을 같이 적는다. 불확실한 것은 "확인 필요"로 적는다.
- 기존 `data/`를 덮어쓰지 않는다. 새 출력은 새 폴더.
- commit은 사용자가 확인한 뒤에 한다(2026-10-15에 "모든 것을 저장·commit·push" 지시가 있었음).
