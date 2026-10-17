# 데이터·수치 출처 기록 (2026-10-17, PC 리셋 전)

이 PC가 리셋되면 다시 알 수 없는 정보를 남긴다. 10-16~17 작업의 데이터와 문서 수치가 어디서 왔는지 추적하기 위한 것이다.

## 환경 (10-16~17 수집·분석에 쓴 것)
| 항목 | 값 |
|---|---|
| 하드웨어 | RTX 4070 12GB, 드라이버 591.59, Windows 11 Pro 10.0.26200 |
| WSL | 3.0.3, 커널 6.18.54.1 (2.7.12/6.18.33에서는 ext4 손상·CUDA 오류가 나서 사용하지 않음) |
| 배포판 | Ubuntu 22.04.5, 기본 사용자 root |
| Python | 3.11.17 (uv) |
| 패키지 전체 | `results/env_freeze_2026-10-17.txt` (uv pip freeze, 103개) |
| 주요 패키지 | torch 2.11.0+cu128, transformers 5.19.0, tokenizers 0.23.3, accelerate 1.15.0, safetensors 0.8.0, scikit-learn 1.9.1, numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, alfworld 0.4.2, textworld 1.7.0, bitsandbytes 0.50.2 |
| 모델 | Qwen/Qwen2.5-3B-Instruct, HF snapshot `aa8e72537993ba99e69dfaafa59ed015b17504d1` (10-16 다운로드) |
| ALFWorld 데이터 | `alfworld-download`(0.4.2), `json_2.1.1` tw-pddl 4,027개. train `game.tw-pddl`을 경로 정렬 후 이어붙인 sha256 `d47d94c64582e092b8f35c8c754f393bc9b3a440fd68eacd3d91907a84940e4f` |

- 10-15 이전 데이터(`main_3b`, reader 4종, 파일럿 등)를 만들 때의 패키지 버전과 모델 revision은 기록이 없다(확인 불가).
  다만 10-16 환경에서 다시 수집한 다중 rollout 앞 24 episode가 10-15 데이터와 action·hidden까지 비트 단위로 같았고,
  시간 한도 100 재수집의 앞 7게임도 처음 30 step이 `main_3b`와 비트 단위로 같았다. 같은 가중치·같은 계산 경로였다는 간접 증거다.
- 10-17의 분석 결과(`results/within_game*`)는 위 scikit-learn 1.9.1 환경에서 계산했다. 10-15 결과(`results/probe_main_3b_primary*`, `confirm_main_3b*` 등)를 계산한 버전은 기록이 없다.

## 데이터
| 데이터 | 만든 방법 | 확인 수단 |
|---|---|---|
| `data/` 10-15분(main_3b 500, reader 4종, 파일럿 160, maxstep50, decision_rerun, 다중 rollout 중단 24) | `data-backup` 브랜치(Git LFS), commit `e8e4c98` | `MANIFEST.txt` 크기, `git lfs fsck` |
| `data/multi_rollout_main_r2` (640) | `docs/multi_rollout_plan_2026-10-15.md` 변경 이력의 명령, 코드 `collect/rollout.py`(마지막 변경 `4b31038`) | `results/sha256_multi_rollout_main_r2.txt` (643개) |
| `data/horizon100` (미완, 리셋 시 소실) | `docs/horizon100_plan_2026-10-17.md`, 설정 `configs/horizon100.yaml`, `configs/alfworld_h100.yaml` | 사전 기록 점검 1~3 (`scripts/2026-10-17/h100_check.py`) |
| `data/horizon100_games/games.json` | `scripts/2026-10-17/make_h100_list.py` | sha256 `c3b5838e…`(계획서에 전체 값) |

## 문서 수치를 만든 스크립트 (`scripts/2026-10-17/`)
모두 저장소 루트(`~/research`)에서 실행하는 일회성 점검 스크립트다. 원래 세션 임시 폴더에 있던 것을 그대로 옮겼다.
| 스크립트 | 만든 수치·용도 | 기록된 곳 |
|---|---|---|
| `wsl_setup.sh` | WSL 환경 구성과 data 복사·바이트 비교 | Notion 10-16~17 1절 |
| `check_manifest.sh`, `cmp_data.sh`, `verify_record.py`, `disk_check.sh`, `cuda_test.py` | 복구 데이터·설치 패키지·디스크·CUDA 점검 | 같은 곳 |
| `compare_24.py` | r2 앞 24 episode의 백업 대비 비트 단위 재현성 | `multi_rollout_plan` 변경 이력 |
| `check_r2.py`, `hid_merge_check.py`, `backup_r2.sh` | r2 정합성, 합친 hidden.npy = shard, Windows 백업과 sha256 목록 | 같은 곳 |
| `design_check.py` | 갈리는 게임 39·장면 28, 프롬프트 동일 쌍 70.4/66.4/52.3%, 생존 | `multi_rollout_plan` "보조 분석과 해석 규칙" |
| `tie_check.py` | 동점 쌍 889개에서 hidden 동일, baseline 시점 이후 특징 상이 | 같은 곳 |
| `test_supp.py` | `probes/within_game_supp.py` 합성 데이터 단위 점검 | 같은 곳 |
| `null_calib.sh`, `agg_null.py` | 무작위 라벨 20 seed null 보정 표 (출력은 /tmp, 보관 안 함) | `multi_rollout_plan` "null 보정 점검" |
| `stop_facts.py`, `cost_facts.py` | 탐색 구간의 실패 길이, cap 규칙 손실·절감, step당 토큰 | Notion 10-16~17 7절 |
| `make_h100_list.py` | 시간 한도 100 게임 목록 | `horizon100_plan` |
| `h100_check.py` | 시간 한도 100 사전 기록 점검 1, 2 | `horizon100_plan` |

## 2026-10-17 오후 재구성 (리셋 후, PC A·PC B)
| 항목 | PC A (원래 PC) | PC B (PC_07) |
|---|---|---|
| GPU / 드라이버 | RTX 4070 12GB / 591.59 | RTX 4070 12GB / 591.59 |
| WSL / 커널 | 3.0.3.0 / 6.18.54.1 (system.vhd 교체 후 설치, RESTART.md 2절) | 3.0.3 / 6.18.54.1 |
| 배포판 | Ubuntu 22.04.5, `[user] default=root` | Ubuntu 22.04.5, `[boot] systemd=true` + `[user] default=root` |
| Python / 패키지 | 3.11.17 (uv 0.13.1), freeze 103개 10-17과 동일 | 3.11.17, freeze 103개 동일 |
| 모델 | Qwen2.5-3B `aa8e7253`(HF main과 같음, 오프라인 로드) | 같음 |
| ALFWorld 데이터 | train 이어붙인 sha256 `d47d94c6…` 일치 | 일치 |
| 점검 | data 50/50 바이트, git fsck, RECORD 15,407/15,407, 3GB 해시, ext4 0, CUDA | 같음(전체 패키지 RECORD 1건 불일치는 PyPI 원본 wheel의 문제) |

| 데이터 | 만든 곳·방법 | 확인 수단 |
|---|---|---|
| `data/multi_rollout_main_r2` (10-17 재수집) | PC A, 위 환경, 코드 `a3fd9bb` | `results/sha256_multi_rollout_main_r2.txt` 643/643 일치(10-16 수집과 비트 단위 동일) |
| `data/horizon100` | PC B, 코드 `a3fd9bb`, 14:51~18:12 | `results/sha256_horizon100.txt` 404개, 사전 기록 점검 1~3(PC A·PC B 모두 통과) |
| `data/pilot_models/{Qwen3-4B,Qwen3-1.7B}` | PC A, 코드 `a1ead7f`, `configs/pilot_*.yaml`(revision 고정) | 저장소 밖, 결과는 `docs/model_pilot_plan_2026-10-17.md` |

- 10-15 결과(`results/probe_main_3b_primary*`, `confirm_main_3b*`)는 10-17 환경에서 같은 코드로 다시 돌렸을 때 AUROC·CI·순열 p·C가 완전히 같았다
  (`results/repro_y30_*`, 결과 대체 없음). 따라서 10-15 계산 환경을 몰라도 이 결과들은 10-17 환경으로 재현된다.
- 10-17 오후 수치를 만든 코드(모두 저장소 루트에서 실행):
| 코드 | 만든 수치·용도 | 기록된 곳 |
|---|---|---|
| `probes/sens_y100.py` | 시간 한도 100 민감도 분석(y100), 같은 환경 재현 점검(`--label y30`) | `horizon100_plan` 결과 절 |
| `scripts/2026-10-17/h100_check3.py` | 시간 한도 100 사전 기록 점검 3(정합성, 성공 여부 출력 안 함) | 같은 곳 |
| `scripts/2026-10-17/h100_judge.py` | 늦은 성공 비율 L, Wilson CI, 구간·task type별, maxstep50 대조 | 같은 곳 |
| `scripts/2026-10-17/cmp_repro.py` | 원래 vs y30 재현 vs y100 비교(환경 효과와 라벨 효과 분리) | 같은 곳 |
| `scripts/2026-10-17/patch_rollout.py` | `collect/rollout.py` 모델별 설정 추가(일회성 패치, `a1ead7f`에 반영됨) | `model_pilot_plan` |
| `scripts/2026-10-17/regress_cmp2.py` | 수정 코드 회귀 점검(main_3b 처음 5게임, 124 step 일치) | 같은 곳 |
| `scripts/2026-10-17/judge_pilot.py` | 모델 파일럿 포함 기준 1~4 판정 | 같은 곳 |
