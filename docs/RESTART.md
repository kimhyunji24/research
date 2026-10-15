# 재시작 가이드 (컴퓨터가 초기화된 뒤)

## 1. 저장소
```bash
git clone https://github.com/kimhyunji24/research.git && cd research && git checkout probes-baseline
```
읽을 순서: `docs/NEXT_SESSION_PROMPT.md` -> `docs/critical_review_2026-10-15.md` -> `docs/HANDOFF.md` -> `docs/freeze_2026-10-15.md` -> `docs/confirmatory_plan_2026-10-15.md`
-> `docs/multi_rollout_plan_2026-10-15.md` -> `docs/paper_draft_2026-10-15.md`. 이전 해시는 `docs/commit_map.md`.

## 2. 환경 (Windows + WSL2, RTX 4070 12GB)
```bash
wsl --install -d Ubuntu-22.04            # PowerShell(관리자). 재부팅이 필요할 수 있다
# WSL 안에서
apt-get update && apt-get install -y git curl build-essential unzip
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=$HOME/.local/bin:$PATH
git clone https://github.com/kimhyunji24/research.git ~/research && cd ~/research && git checkout probes-baseline
uv venv --python 3.11 .venv && . .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers==5.19.0 accelerate scikit-learn numpy pandas pyyaml tqdm alfworld==0.4.2 bitsandbytes
export ALFWORLD_DATA=/root/alfworld_data && alfworld-download     # 약 2.3GB
```

## 3. 데이터 복구

**먼저 백업에서 받는다**(브랜치 `data-backup`, Git LFS, 약 730MB, 2026-10-15 업로드):
```bash
git lfs install
git clone --branch data-backup --single-branch https://github.com/kimhyunji24/research.git ../research_data
cp -r ../research_data/data ./data     # 코드 저장소의 data/로 복사
```
`MANIFEST.txt`에 파일 목록과 크기가 있다. 백업이 없거나 깨졌을 때만 아래로 다시 만든다.
| 데이터 | 명령 | 시간(RTX 4070) |
|---|---|---|
| `data/main_3b` (500 episode, 확증 포함) | `python collect/rollout.py --config configs/main_3b.yaml --num_episodes 500` | 약 95분 |
| `data/main_3b/readers/*` (0.5B, 1.5B, 7B 4-bit) | `python collect/extract_reader.py --run data/main_3b/Qwen2.5-3B-Instruct --reader Qwen/Qwen2.5-0.5B-Instruct` (1.5B 동일, 7B는 `--load_4bit`) | 8분 / 8분 / 28분 |
| `data/multi_rollout_pilot` | `python collect/rollout.py --config configs/multi_rollout.yaml --game_list results/games_pilot.json --repeats 8 --out_dir data/multi_rollout_pilot` | 약 22분 |
| `data/multi_rollout_main` (본 규모, 새 80게임) | `bash results/run_multi_main.sh` (게임 목록 `results/games_main.json`, `sample_seed=1`) | 약 90~110분 |

- 환경의 게임 순서와 탐욕 정책은 결정적이다(`main_3b` 처음 100 episode가 재수집과 일치함). 같은 결과가 나와야 한다.
- 샘플링 rollout은 (sample_seed, episode, step)으로 결정되어 같은 설정이면 재현된다(확인은 안 함).
- 중단된 수집은 `--resume`. 게임 순서가 이전과 다르면 중단한다.

## 4. 주요 분석 명령
```bash
python probes/run_probes.py data/main_3b/Qwen2.5-3B-Instruct --layer 2 --rounds 0 1 2 --kinds linear --rungs 5 --eval_from_episode 250 --out results/confirm_main_3b.csv   # 확증
python probes/within_game.py data/multi_rollout_pilot/Qwen2.5-3B-Instruct data/multi_rollout_main/Qwen2.5-3B-Instruct --layer 2 --steps 1 2 3 --out results/within_game    # 게임 내
python probes/extra_analysis.py data/main_3b/Qwen2.5-3B-Instruct --out results/extra_main_3b    # reader 쌍, 후반 라운드, 시간축
```

## 5. 주의
- 수집(GPU)과 프로브(CPU) 병행 시 CPU 경쟁으로 수집이 5배 느려진다. `taskset`/`nice`로 코어를 나누거나 순서대로 실행한다.
- 3B 수집은 VRAM 10.6GB라 다른 GPU 작업과 동시에 못 한다.
- 7B는 4-bit(nf4)만 가능(bf16은 12GB 초과).
- 기본 사용자가 root이며 사용자 계정은 만들지 않았다.
