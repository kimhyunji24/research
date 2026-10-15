# 분석 계획 보충 (2026-10-15, 본 수집 이후 · 프로브를 `main_3b`에 돌리기 전)

설계 문서 10절 사전 등록과 `docs/freeze_2026-10-15.md`를 보충한다. **`main_3b`의 hidden state 프로브는 이 문서를 commit한 뒤에만 돌린다.**
이 문서를 쓰는 시점에 `main_3b`에서 본 것은 episode·step 수, 성공 수, task type별 분포, 후보 길이 통계뿐이다(AUROC는 보지 않았다).
`decision_rerun`(n=100)은 이미 프로브를 돌려 본 데이터이고 `main_3b`의 처음 100 episode와 같으므로 독립 표본이 아니다.

## 왜 바꾸는가
예비 실행(n=100)에서 무작위 라벨 대조군이 0.33~0.41로 나왔다. 원인 후보는 (1) 층화하지 않은 GroupKFold에서 fold마다 양성 비율이
달라 풀링한 AUROC에 음의 편향이 생기는 것, (2) 게임 단위 분할이 같은 장면(scene)의 게임을 train/test에 나눠 넣는 것이다.
둘 다 합성 데이터로 점검한다(`probes/synth_check.py`). 원인은 확인 필요이며, 확인 전에는 이 보정이 충분하다고 단정하지 않는다.

## 확정하는 절차
| 항목 | 값 |
|---|---|
| 분할 단위 | **장면(scene)**: 게임 경로 디렉터리 이름의 마지막 숫자(예: `...-CounterTop-18` → 18). `main_3b`에서 86개 |
| 교차적합 | `StratifiedGroupKFold(5, shuffle)` x 2회 반복(seed 0, 1), 행별 점수는 반복 평균. 모든 행이 한 번씩 평가에 쓰인다 |
| 정규화 | 선형 모델 `C`는 nested: 바깥 train fold 안에서 3-fold 장면 단위 AUROC로 `C ∈ {1e-4,1e-3,1e-2,1e-1,1}` 선택. baseline과 프로브가 같은 격자 |
| MLP | 은닉 64, alpha 1e-2, early stopping 고정(탐색적 지표) |
| 표준화/PCA/TF-IDF | 모두 Pipeline 안에서 fold별로 학습 |
| 대조군 | 라운드 행의 라벨을 무작위로 섞고 **같은 절차 전체**(nested `C` 포함)를 반복. 평균, 2.5/97.5 분위, 경험적 p=(1+#{null>=관측})/(1+N) |
| 신뢰구간 | 장면 단위 cluster bootstrap 1000회(주). episode 단위 bootstrap도 같이 보고. 프로브-baseline 차이는 같은 재표집으로 짝지어 계산 |
| 다중 비교 | 주 지표: 라운드(3) x reader(3)의 프로브 - baseline 5단 차이에 Holm. 나머지(MLP, PCA 스윕, 다층 결합)는 탐색적 |
| PCA | 기본 끔. k는 {8,32,64,128}, 각 설정의 평균 explained variance 비율을 함께 출력. fold 학습 표본 수 이상이면 건너뛴다 |

## baseline 사다리 정의 (행 = 라운드 r에 살아 있는 episode)
| 단 | 입력 |
|---|---|
| 1 | step 번호 (라운드 안에서는 상수 → AUROC 0.5가 되어야 한다) |
| 2 | +task type (6종 one-hot) |
| 3 | +최근 6 step 안의 반복 횟수(현재 선택한 action 포함), 선택된 후보 확률 `p_chosen`, 후보 수 `n_cand` |
| 4 | +진행도: 지금까지 방문한 서로 다른 장소 수, open 횟수, 목표 물체 take 횟수/여부, 변환 action(heat/cool/clean/use) 횟수, move 횟수, 관찰 텍스트에서 목표 물체를 본 적이 있는지(프롬프트 창 6 step 안에서만 확인 가능). 모두 현재 step **이전** action에서만 계산(`obj_seen`은 프롬프트의 관찰 텍스트) |
| 5 | +프롬프트 user 메시지 전체의 TF-IDF(1~2-gram, min_df=2, sublinear). 과제 문장, 초기 방 묘사, 최근 history, 현재 관찰, admissible 목록을 포함 |

한계: 3단의 `repeat`, `p_chosen`, `n_cand`는 현재 step의 정책 계산 이후 값이라 hidden state 시점(프롬프트 마지막 토큰)보다 약간 늦다.
baseline을 더 강하게 만드는 방향이므로 프로브에 불리한 쪽의 보수적 선택이다. 목표 물체 추출은 정규식이며 파싱 실패율을 보고한다.

## 아직 미결정 (프로브 실행 전에 사용자가 정해야 함)
- "라운드 1"이 첫 번째 action(step 0)인지. 현재 코드의 `--rounds`는 0-based step이다.
- 주 지표에 쓸 layer(50% = 인덱스 1, 3B 기준 layer 18)와 reader 순서는 설계 문서 10절을 따른다.

## 이번 단계에서 하는 것과 하지 않는 것
- 한다: 코드 수정, 합성 데이터 검증, `main_3b`에서 **baseline 1~5단만** 실행(프로브 AUROC는 보지 않는다).
- 하지 않는다: `main_3b` hidden state 프로브 실행, 0.5B/1.5B reader 추출, 표본 크기 확대 결정.
