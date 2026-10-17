# 재시작 가이드 (컴퓨터가 초기화된 뒤)

2026-10-17 갱신: 실습실 PC(리셋·자동 종료) 두 대에서 복구한 경험을 반영했다. 기준 환경은 `docs/provenance_2026-10-17.md`.

## 0. 먼저 확인
- 실습실 관리 프로그램(MaestroNet Agent7, Exit Pro) 때문에 종료하면 초기화되고, 입력이 없으면 자동 종료될 수 있다. 긴 GPU 작업 전에 관리자에게 예외를 문의한다.
- 보존 폴더 확인: 여러 폴더에 둔 `_reset_marker_research.txt`가 남은 곳(Notion 2026-10-16 페이지 9절). PC A에는 없었다(10-17).
- 이 저장소에서 commit하는 세션은 하나만 둔다.

## 1. 저장소
읽을 순서: `docs/NEXT_SESSION_PROMPT.md` -> `docs/critical_review_2026-10-15.md` -> `docs/multi_rollout_plan_2026-10-15.md` -> `docs/horizon100_plan_2026-10-17.md`
-> `docs/model_pilot_plan_2026-10-17.md` -> `docs/early_stop_plan_2026-10-17.md` -> `docs/paper_draft_en.md`. 이전 해시는 `docs/commit_map.md`.

## 2. 환경 (Windows + WSL2, RTX 4070 12GB, 드라이버 591.59)
PowerShell(관리자):
```powershell
winget install --id Git.Git -e --scope user      # Git + Git LFS (실제 설치 위치는 C:\Program Files\Git)
wsl --update                                     # 반드시 먼저. 3.0.3 이상 확인(wsl --version). 2.7.12에서 ext4 손상·CUDA 오류 2회
wsl --install -d Ubuntu-22.04 --no-launch
```
- 설치가 `CreateVm/HCS/ERROR_FILE_CORRUPT`로 실패하면(10-17 PC A): Hyper-V가 `C:\Program Files\WSL\system.vhd`를 열지 못하는 것이다(파일 해시는 정상).
  `wsl --shutdown` -> `Stop-Service WslService` -> `system.vhd`를 `system.vhd.bad`로 이름 변경 -> 같은 파일의 복사본을 원래 이름으로 넣기(해시 확인) -> `Start-Service WslService` -> 다시 설치.
- PowerShell에서 `wsl.exe -- bash -c "..."`로 따옴표·`\n`이 든 명령을 넘기면 깨진다. 스크립트 파일을 Windows 쪽에 두고 `tr -d '\r'`로 변환해 실행하거나 base64로 넘긴다.
  Git Bash의 wsl.exe는 경로 변환 때문에 실패한다.

WSL 안(root):
```bash
printf '[user]\ndefault=root\n' > /etc/wsl.conf            # 기본 사용자 root (Ubuntu 앱을 직접 실행하면 일반 사용자가 생기니 주의)
apt-get update && apt-get install -y git git-lfs curl build-essential unzip rsync
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=$HOME/.local/bin:$PATH
git clone https://github.com/kimhyunji24/research.git ~/research && cd ~/research && git checkout probes-baseline   # 코드는 WSL에서 clone(Windows clone은 CRLF라 .sh가 깨짐)
uv venv --python 3.11.17 .venv && . .venv/bin/activate
uv pip install -r results/env_freeze_2026-10-17.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match   # 버전 고정(비트 단위 재현용)
export ALFWORLD_DATA=/root/alfworld_data && alfworld-download     # 약 2.3GB. train game.tw-pddl 이어붙인 sha256 d47d94c6...(provenance)
```
- 모델: Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`로 받고 `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/refs/main`에
  그 해시를 둔 뒤 수집은 `HF_HUB_OFFLINE=1`로 한다. 다른 모델은 `configs/pilot_*.yaml`의 `revision`.
- 점검(하나라도 실패하면 수집하지 않음): data 바이트 비교, `git fsck`, `python scripts/2026-10-17/verify_record.py .venv/lib/python3.11/site-packages`,
  `bash scripts/2026-10-17/disk_check.sh`(3GB 해시 재확인, ext4 경고 0), `python scripts/2026-10-17/cuda_test.py`.

## 3. 데이터 복구
**백업에서 받는다**(브랜치 `data-backup`, Git LFS, 약 730MB). Windows에서 autocrlf를 끄고 받은 뒤 WSL로 복사한다(켜면 JSON에 CR이 붙는다):
```powershell
git -c core.autocrlf=false clone --branch data-backup --single-branch https://github.com/kimhyunji24/research.git research_data
```
```bash
rsync -a /mnt/c/Users/<사용자>/Desktop/research_data/data/ ~/research/data/   # 그다음 MANIFEST.txt 크기, git lfs fsck, 바이트 비교
```
LFS에 없는 데이터(2026-10-17 기준 사용자 USB `research_backup_2026-10-17`, `horizon100_final`에 sha256과 함께 보관):
| 데이터 | 검증 | 다시 만들기(GPU, RTX 4070) |
|---|---|---|
| `data/multi_rollout_main_r2` (640) | `sha256sum -c results/sha256_multi_rollout_main_r2.txt` (643개) | `python collect/rollout.py --config configs/multi_rollout.yaml --game_list data/multi_rollout_main_games/games_main.json --repeats 8 --sample_seed 1 --out_dir data/multi_rollout_main_r2` (약 90분) |
| `data/horizon100` (399) | `sha256sum -c results/sha256_horizon100.txt` (404개) | `python scripts/2026-10-17/make_h100_list.py` 후 `python collect/rollout.py --config configs/horizon100.yaml --game_list data/horizon100_games/games.json --repeats 1 --out_dir data/horizon100` (약 3.5시간) |
| `data/pilot_models/*` | - | `configs/pilot_*.yaml`의 사용 줄(모델당 3~5분) |

- **게임 순서 주의(10-17 발견)**: `--num_episodes`만 주면 환경이 내보내는 게임 순서가 WSL 설치마다 다를 수 있다(새 설치에서 `main_3b`와 처음 5게임이 모두 달랐다).
  같은 게임을 다시 돌릴 때는 항상 `--game_list`를 쓴다. `main_3b` 자체를 다시 만들어야 하면 `data/main_3b/.../episodes.jsonl`의 task_id 순서로 목록을 만든다.
- 샘플링 rollout은 (sample_seed, episode, step)으로 결정되며, 같은 GPU·드라이버·패키지에서 비트 단위로 재현됐다(r2 643/643, 10-17).
- 중단된 수집은 `--resume`. 게임 순서가 이전과 다르면 중단한다.

## 4. 주요 분석 명령
```bash
python probes/run_probes.py data/main_3b/Qwen2.5-3B-Instruct --layer 2 --rounds 0 1 2 --kinds linear --rungs 5 --eval_from_episode 250 --out results/confirm_main_3b.csv   # 확증(이미 1회 실행, 재실행 금지)
python probes/within_game.py data/multi_rollout_pilot/Qwen2.5-3B-Instruct data/multi_rollout_main_r2/Qwen2.5-3B-Instruct --layer 2 --steps 1 2 3 --game_meta data/multi_rollout_pilot_games/games_pilot_meta.json data/multi_rollout_main_games/games_main_meta.json --out results/within_game   # 게임 내 주 분석(이미 1회 실행)
python probes/sens_y100.py [--max_episode 250] [--label y30] -- <run_probes.py 인자>   # 시간 한도 100 민감도(y100)·재현 점검(y30)
python probes/extra_analysis.py data/main_3b/Qwen2.5-3B-Instruct --out results/extra_main_3b    # reader 쌍, 후반 라운드, 시간축
```

## 5. commit과 push
- commit 작성자: `kimhyunji24 <hyunji4560@gmail.com>`. Claude·세션 표시는 넣지 않는다.
- WSL 안에서는 GitHub 인증이 안 된다. Windows clone에서 WSL 저장소를 fetch한 뒤 push한다(처음 한 번 브라우저 로그인, 대화형 터미널 필요):
```powershell
git -C <Windows clone> -c safe.directory=* fetch \wsl.localhost\Ubuntu-22.04\root\research probes-baseline
git -C <Windows clone> push origin <sha>:refs/heads/probes-baseline
```

## 6. 주의
- 긴 작업은 wsl.exe 연결을 유지한 채 돌린다(연결이 끊기면 WSL VM이 유휴 종료되면서 작업도 죽는다).
- 수집(GPU)과 프로브(CPU) 병행 시 CPU 경쟁으로 수집이 느려진다. `taskset`/`nice`로 코어를 나눈다.
- Qwen2.5-3B 수집은 VRAM 약 10.6GB라 다른 GPU 작업과 동시에 못 한다. KV head가 많은 모델(Llama-3.2-3B, Qwen3-4B)은 `cand_batch: 8`.
- 7B reader는 4-bit(nf4)만 가능(bf16은 12GB 초과).
