# 다중 rollout 본 규모: 새 80게임 x 8회 (T=1.0, sample_seed=1). 로그는 ~/multi_rollout_main.log
set -u
cd ~/research && . .venv/bin/activate && export ALFWORLD_DATA=/root/alfworld_data
cp /mnt/c/research/collect/rollout.py collect/ && cp /mnt/c/research/configs/multi_rollout.yaml configs/
python -W ignore collect/rollout.py --config configs/multi_rollout.yaml \
  --game_list data/multi_rollout_main_games/games_main.json --repeats 8 --sample_seed 1 \
  --out_dir data/multi_rollout_main > ~/multi_rollout_main.log 2>&1
tail -n 3 ~/multi_rollout_main.log | cut -c1-300
