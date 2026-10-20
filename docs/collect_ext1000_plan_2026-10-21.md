# 새 episode 1,000개 수집과 학습 크기 확장 계획 (2026-10-21, 초안, 탐색)

> **상태: 2026-10-21 사용자 승인(문서 그대로). 이 승인 기록을 commit한 뒤에 수집한다.**

## 왜, 그리고 어떤 규칙의 예외인가
- 5차 리뷰 (1)·(3): 500개 sweep에서 학습 약 50→400개의 격차 변화는 −0.010(−0.038 ~ +0.022)으로 탐지되지 않았지만, 프로브(0.651→0.707)와 5단(0.635→0.701) 모두 약 400개에서도 오르고 있어 곡선이 포화되지 않았다. 토의의 "셋째 설명(표본 부족)"을 더 큰 학습 집합으로 직접 본다.
- **표본 확대 금지 규칙의 예외**이다. 사용자 요청(2026-10-21, "write a plan for collecting 1,000+ new episodes as an exception to the no-enlargement rule; don't collect until I approve the plan")에 따라 계획만 쓴다. 이 예외는 이 계획 하나에만 적용하고, 다른 분석의 표본 확대로 넓히지 않는다.
- B 결과, 확증 결과, 500개 sweep 결과를 모두 본 뒤 정한 **탐색** 분석이다. 사전 등록·확증 판정을 바꾸지 않으며, 결과는 "시사한다"로만 쓴다.

## 수집
- 정책·프롬프트·코드: `configs/main_3b.yaml`과 같다(Qwen2.5-3B-Instruct, bfloat16, 후보 점수화 `policy: score`, `score_norm: sum`, 탐욕 `temperature: 0`, `max_steps: 30`, `relative_layers: [0.25, 0.5, 0.75, 1.0]`, split train, seed 0). 새 config `configs/ext1000_3b.yaml`은 `out_dir`와 `game_list`만 다르다. `.venv`는 동결된 그대로(transformers 5.19, torch 2.11).
- 게임: ALFWorld train 3,553게임 중 main_3b의 500게임을 뺀 3,053게임에서 seed 0으로 1,000게임을 비복원 추출한다(층화 없음, 탐욕 결과를 보기 전). 새 스크립트 `collect/make_new_game_list.py`가 목록과 해시를 `data/ext1000_3b/games.json`에 쓴다. main_3b와 겹치는 게임이 0인지 확인한다.
- 저장: `data/ext1000_3b/Qwen2.5-3B-Instruct/`(새 폴더, 기존 `data/` 덮어쓰지 않음). 예상 크기는 hidden 약 0.45 GB(step당 4깊이 × 2,048차원 float16, episode당 평균 약 26 step 기준), steps.jsonl 약 0.1 GB. 디스크 여유 914 GB.
- 시간과 GPU: episode당 약 11초(확증 수집 기준, `docs/confirmatory_plan_2026-10-15.md`)로 약 3.1시간. GPU 작업은 사용자가 있을 때만 하므로 250개씩 4번(각 약 47분)으로 나누고 `--resume`으로 잇는다. 중간 묶음의 결과(성공률 포함)는 열람하지 않고, 진행 줄 수만 확인한다.
- 수집 전 점검(결과 아님): main_3b의 처음 3게임을 같은 설정으로 `/tmp`에 다시 돌려 action 열과 hidden이 저장값과 같은지 확인한다(결정성). 다르면 수집하지 않고 원인을 찾는다.
- 수집 후 점검: episode 1,000개, 게임 중복 0, main_3b와 겹침 0, hidden 행 수 = steps 행 수, `prompt_tokens` 기록, step 0–2 행 수. sha256을 기록하고, 분석 직전에 캐시를 거치지 않고 다시 확인한다(2026-10-20 WSL ext4 손상 경험).

## 분석 (수집 전에 고정)
- 데이터: main_3b 500 + 새 1,000 = 1,500 episode, step 0–2. 새 episode 중 step 2 이전에 끝난 episode가 있으면 그 step에서 빠지며, 행 수를 보고한다.
- 절차: `scripts/2026-10-21/sweep500.py`와 같다. 장면 단위 `StratifiedGroupKFold` 5-fold × 반복 2(seed 0, 1)로 평가 fold를 고정하고, 바깥 학습 fold(약 1,200 episode)에서 비율 f ∈ {1/24, 1/12, 1/6, 1/3, 2/3, 1}(약 50, 100, 200, 400, 800, 1,200 episode)만큼 라벨 층화로 뽑아 학습한다. f < 1은 뽑기 5회 평균, nested C(`C_GRID`). 새 스크립트 `scripts/2026-10-2x/sweep1500.py`(날짜는 실행일).
- 방법: P75, P50, R4, R5(기존 정의). 장면 cluster bootstrap 1,000회, 짝지은 차이.
- 주 지표: 대비 **(P75 − R5)[약 1,200] − (P75 − R5)[약 400]**(step 평균, 95% CI). 약 400까지는 이미 본 범위이므로, 그 너머에서의 변화만 주 지표로 삼는다.
- 보조: (P75 − R5)[약 1,200] − (P75 − R5)[약 50], 크기별 P75 − R5와 각 방법의 AUROC, P50 − R5, step별 값, Holm 보정(크기별 차이 family). 평가 행을 main_3b 500개와 새 1,000개로 나눈 값(기술).
- 해석 규칙:
  - 주 대비의 CI 하한 > 0: "약 400개에서 1,200개로 학습을 늘리자 5단과의 격차가 줄었다. 프로브가 학습 표본에 제한되었을 수 있다"로 보고하고 토의의 셋째 설명을 그에 맞게 고친다.
  - CI가 0 포함: "약 400–1,200개 범위에서도 격차 변화는 탐지되지 않았다"로 보고하고, CI 폭을 함께 적는다.
  - CI 상한 < 0: "학습을 늘리자 격차가 벌어졌다"로 보고한다.
  - 약 1,200개의 P75 − R5 CI 하한이 0보다 커도 탐색 결과이므로 "우수"라는 판정어는 쓰지 않는다. 1,200개 너머로 외삽하지 않는다.
- 하지 않는 것: 다른 actor·프롬프트·온도·환경, 새 reader, 시도 수준(게임 내) 수집, 확증 분석 재실행. 새 episode로 사전 등록·확증 비교를 다시 하지 않는다.

## 출력과 원고
- 출력 `results/sweep1500_{lc,diff,contrast,half}.csv`, 로그 `results/collect_ext1000.log`, `results/sweep1500.log`(시작·종료 시각, HEAD). 출력이 있으면 멈춘다. 실행 전 라벨을 섞은 smoke 점검.
- 원고: 4.8절 학습 크기 단락과 표 8·9, 토의 셋째 설명·보완적 해석, 축약판 한 문장을 결과에 맞게 고친다. 초록은 주 대비의 CI 하한이 0보다 클 때만 고친다. 한계의 "약 400개 너머 미시험" 문장을 바꾼다. 표 A8에 "표본 확대 규칙의 예외, 결과를 본 뒤 계획" 행을 추가한다.
- 백업: 새 `data/ext1000_3b/`(약 0.6 GB)를 USB에 복사한다.

## 변경 이력
- 2026-10-21: 초안(사용자 승인 전, 수집 안 함).
- 2026-10-21: 사용자 승인, 내용 변경 없음. 승인 문구: "I approve docs/collect_ext1000_plan_2026-10-21.md as written, as an exception to the no-enlargement rule. Log the approval, commit everything so far (text fixes, sweep, plan), then run the determinism check and the first batch of 250 while I'm here." 표본 확대 금지 규칙의 예외는 이 계획에만 적용한다. 수집과 분석 결과는 아직 보지 않았다.
