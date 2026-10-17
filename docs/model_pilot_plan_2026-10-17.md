# 모델 확장 파일럿 계획 (2026-10-17, 파일럿 실행 전에 기록)

## 왜 하는가
지금까지의 모든 결과는 actor 하나(Qwen2.5-3B-Instruct)에서 나왔다. 연구 질문을 "에이전트 hidden state 실패 신호는 과제 난이도를 읽는가
진행 중 실패를 읽는가, 그 답은 모델 크기·계열에 따라 어떻게 바뀌는가"로 넓히면서(2026-10-17 사용자 결정), 같은 환경(ALFWorld)에서
actor를 최소 2개 추가한다. 이 문서는 **어떤 모델을 본 실험에 넣을지 정하는 파일럿 규칙**만 고정한다. 본 수집과 분석 절차는 파일럿 후,
본 수집 전에 별도로 사전 기록한다.

## 후보
| 우선 | 모델 | HF revision | 역할 | 비고 |
|---|---|---|---|---|
| 1 | Qwen/Qwen3-4B (thinking 끔) | `1cfa9a7208912126459214e8b04321603b3df60c` | 같은 계열 계보, 더 큼·최신 | bf16 가중치 7.49GB |
| 1 | meta-llama/Llama-3.2-3B-Instruct | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | 같은 크기, 다른 계열 | gated. 2026-10-17 17시대 기준 접근 승인 대기(403) |
| 대체 | HuggingFaceTB/SmolLM3-3B | `a07cc9a04f16550a088caea529712d1d335b0ac1` | 다른 계열 대체 | 대체 1순위 |
| 대체 | microsoft/Phi-3.5-mini-instruct | `2fe192450127e6a83f7441aef6e3ca586c338b77` | 다른 계열 대체 | 대체 2순위, 3.8B |
| 선택 | Qwen/Qwen3-1.7B (thinking 끔) | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 크기 사다리의 작은 점, Doomed from the Start와 같은 모델 | 파일럿만 하고 본 실험 포함 여부는 기준으로 |

- revision은 2026-10-17 HF API의 main 해시이며 설정 파일(`configs/pilot_*.yaml`의 `revision`)에 고정했다.
- 1.5B 이하를 크기로 미리 빼지 않는다. 아래 기준을 통과하지 못하면 빠진다.
- Qwen2.5 0.5B/1.5B의 낮은 성공률(2026-10-08: 0/20, 2/100)은 이 기준을 정하기 전에 알고 있었다(기록).
- Llama 접근 승인이 늦어지는 것은 기준 탈락이 아니다. 승인될 때까지 기다리고, 주말 안에 승인되지 않으면 사용자가 대체로 넘어갈지 정한다.

## 파일럿 절차
- 정책: `main_3b`와 같다. 프롬프트 v2, 후보 점수화 `score_norm: sum`, 탐욕, `max_steps: 30`, train split, seed 0.
- 게임: `main_3b` 처음 20 episode의 게임. `--game_list results/games_pilot_models_first20.json --repeats 1`
  (sha256 `b7e2f9149fb439c5c21612e10b0cff996dda66dec2b242e652f416639b9c22ef`).
  게임 목록을 쓰는 이유: 2026-10-17 PC A의 새 WSL 설치에서 `--num_episodes`만 주면 환경이 내보내는 게임 순서가 `main_3b`와 달랐다
  (처음 5게임 모두 다름, 모델 없이 환경만으로 확인). 같은 설치 안에서는 결정적이지만 설치 간에는 같지 않다. 게임 목록을 쓴 r2·시간 한도 100은 영향이 없다.
- 설정: `configs/pilot_llama32_3b.yaml`, `configs/pilot_qwen3_4b.yaml`, `configs/pilot_qwen3_1p7b.yaml`(main_3b.yaml 대비 모델, revision, 아래 모델별 키, 출력 폴더만 다름).
- 출력: `data/pilot_models/<모델>`(새 폴더).

### 코드 변경 (`collect/rollout.py`, 기본값은 기존 동작과 같음)
| 설정 키 | 기본값(= 기존) | 용도 | 파일럿 값 |
|---|---|---|---|
| `turn_end` | `<\|im_end\|>` | 후보 `Action: <a><turn_end>`의 assistant 턴 종료 토큰. 시작 시 chat template으로 일치 여부를 assert | Llama `<\|eot_id\|>`, Qwen3 `<\|im_end\|>` |
| `chat_template_kwargs` | 없음 | chat template 인자 | Qwen3 `enable_thinking: false`, Llama `date_string: "26 Jul 2024"` |
| `cand_batch` | 0(후보 한 번에) | 후보를 나눠 점수 계산(KV cache 메모리) | 8 |
| `revision` | 없음 | HF commit 고정 | 위 표 |

- Llama 3 chat template은 날짜를 따로 주지 않으면 시스템 메시지에 실행 날짜를 넣는다. 그러면 실행하는 날마다 prompt가 달라지므로, 템플릿 기본값 "26 Jul 2024"로 고정한다(템플릿 렌더링 확인은 접근 승인 후, 확인 필요).
- Qwen3의 `enable_thinking: false`는 생성 프롬프트 끝에 빈 `<think>\n\n</think>\n\n` 블록을 넣는다. hidden은 그 뒤의 prompt 마지막 토큰에서 뽑는다.
- `cand_batch`가 필요한 이유: 후보마다 prompt KV cache를 복제한다. Qwen2.5-3B는 KV head 2개라 최악(prompt 1,502 토큰 x 후보 63개)에도 약 3.5GB지만,
  Llama-3.2-3B(KV head 8, 28층)는 약 11GB, Qwen3-4B(KV head 8, 36층)는 약 14GB라 12GB GPU에서 메모리가 부족하다(계산, `main_3b` 최대값 기준).
  8개씩 나누면 각각 약 1.4GB, 1.8GB다. 나눠 계산하면 GEMM 배치 크기가 달라져 수치가 아주 조금 다를 수 있으나, 같은 모델 안에서는 고정값이라 결정적이다.
  Qwen2.5-3B 기존 데이터와의 일치는 `cand_batch: 0`(기본) 경로로만 주장한다.

### 회귀 점검 (2026-10-17, 파일럿 전, 통과)
- 수정한 코드로 Qwen2.5-3B(`configs/main_3b.yaml`, 기본값)를 `main_3b` 처음 5게임(`--game_list`)에 돌렸다:
  episode 5개·124 step에서 성공·step 수 불일치 0, prompt·action·후보·후보 로그확률·후보 토큰 수·p_chosen 불일치 0, hidden(4 layer) 최대 차이 0.0.
- 첫 시도는 `--num_episodes 5`로 돌려 다른 게임이 나와 비교할 수 없었다(위 게임 순서 문제). 그 출력 `data/regress_qwen25`는 쓰지 않는다.

## 포함 기준 (결과 보기 전 고정)
1. **성공**: 20 episode 중 성공 3개 이상(15%). 근거: 본 수집 250 episode에서 성공이 약 35개 이상이어야 과제 단위 AUROC 차이의 CI 반폭이
   기존 3B 결과(성공 56개, 반폭 약 0.06~0.10)보다 크게 나빠지지 않는다(추정, 확인 필요). 20개로 추정한 성공률은 거칠다는 한계를 기록한다.
2. **실행 가능성**: 12GB GPU에서 메모리 부족 없이 20 episode 완료(`cand_batch: 8`), episode당 평균 시간이 3B의 3배 이내.
3. **유효 행동**: 후보 점수화라 항상 유효(점검만).
4. **저장 정합성(점검)**: hidden shard에 NaN·inf 0(float16 저장이라 큰 활성값이 넘칠 수 있음). 실패하면 저장 형식을 고친 뒤 다시 돌리고 기록한다.
- 기준을 통과한 우선 1 모델 2개를 본 실험에 넣는다. 우선 1이 하나라도 탈락하면 대체 후보 중 통과한 모델로 채운다.
  대체 순서(2026-10-17 사용자 확정): SmolLM3-3B 먼저, 탈락하면 Phi-3.5-mini. 대체 모델도 같은 기준 1~4를 적용한다.
- 탈락 모델의 파일럿 결과(성공 수, 시간, 메모리)는 논문 부록에 보고한다.

## 파일럿 결과 (2026-10-17, 위 기준을 commit `a1ead7f`에 고정한 뒤 실행)
PC A(RTX 4070, 드라이버 591.59, WSL 3.0.3), 코드 `a1ead7f`, `HF_HUB_OFFLINE=1`, 게임 `results/games_pilot_models_first20.json`(main_3b 처음 20게임), 출력 `data/pilot_models/<모델>`.

| 모델 | 실행 | 기준 1 성공(≥3/20) | 기준 2 실행 가능성 | 기준 3 유효 행동 | 기준 4 hidden | 판정 |
|---|---|---|---|---|---|---|
| Qwen3-4B | 17:39~17:44 | **7/20** (35%, Wilson 95% CI 18~57%) 통과 | 20/20 완료, 메모리 부족 0, 14.9초/episode, 최대 할당 10.24GB 통과 | 475/475 | 475행 = step 475, NaN 0, inf 0, 최대 절댓값 98.5 | **포함** |
| Qwen3-1.7B | 17:49~17:52 | **1/20** (5%, Wilson 95% CI 1~24%) 탈락 | 20/20 완료, 메모리 부족 0, 8.5초/episode, 최대 할당 5.03GB 통과 | 574/574 | 574행 = step 574, NaN 0, inf 0, 최대 절댓값 304.0 | **제외**(기준 1) |
| Llama-3.2-3B | 미실행 | - | - | - | - | 접근 승인 대기(탈락 아님) |

- 같은 20게임에서 Qwen2.5-3B(`main_3b` 기록)는 5/20 성공했다(참고, 비교 검정 아님).
- 기준 2의 "3B의 3배"는 기준 시간을 문서에 정하지 않았다. 참고값은 회귀 점검의 Qwen2.5-3B 6.9초/episode(5게임, 34.7초)와 초안 3.7절의 8~11초/episode다.
  가장 엄격한 6.9초 기준(3배 = 20.8초)으로도 두 모델 모두 통과한다.
- 성공 episode의 step 수: Qwen3-4B 3, 4, 7, 10, 15, 19, 27 / Qwen3-1.7B 4.
- task type별 성공: Qwen3-4B look_at 3/4, pick_and_place 1/3, clean 0/5, cool 1/1, heat 1/4, two_obj 1/3. Qwen3-1.7B look_at 1/4, 나머지 0.
- 20개 표본이라 성공률 추정이 거칠다(CI 폭 약 40%p). 기준 1은 포함 여부만 정하며, 모델 간 성공률 차이를 주장하지 않는다.
- 결과: 우선 1 중 Qwen3-4B 포함. Llama-3.2-3B는 승인 후 같은 절차로 파일럿한다. Llama가 탈락하면 대체 1순위 SmolLM3-3B를 파일럿한다.
  Qwen3-1.7B는 탈락 모델로 부록에 보고한다.
- 데이터 `data/pilot_models`는 저장소 밖(gitignore)이고 PC A에만 있다. 같은 설정으로 다시 돌리면 재현된다고 기대하지만 확인하지 않았다.

## 본 실험 규모 (예정, 파일럿 후 별도 사전 기록에서 확정)
- 모델당 탐욕 본 수집 250 episode + 다중 rollout 50게임 x 8회(T=1.0, 게임은 그 모델의 탐욕 결과로 25/25 층화, seed 1).
- 분석은 Qwen2.5-3B와 같은 절차(과제 단위 사전 등록 비교, 게임 내 쌍 AUROC, η², 난이도 오라클). 모델별로 따로 보고하고 기존 결과와 합치지 않는다.
- 본 수집도 게임 목록으로 고정한다(위 게임 순서 문제). 예상 GPU: 모델당 약 2~3시간(추정, 파일럿 시간으로 갱신).

## 변경 이력
- 2026-10-17: 최초 기록(파일럿 전, fork 세션 초안).
- 2026-10-17 (파일럿 전, 주 세션): revision 고정, 게임 목록 사용(설치 간 게임 순서 불일치 발견), 코드 변경과 회귀 점검 결과, `cand_batch`·`chat_template_kwargs` 추가,
  기준 4(NaN·inf) 추가, 대체 순서 확정(SmolLM3-3B 먼저, 사용자 결정), Llama 접근 대기는 탈락 아님을 명시. 대체 모델의 설정 파일은 필요해질 때 만든다.
- 2026-10-17 (파일럿 후): Qwen3-4B·Qwen3-1.7B 파일럿 결과 절 추가. 기준은 바꾸지 않았다. 기준 2의 기준 시간이 정의되지 않았음을 기록하고 참고값으로 판정했다.
