cd ~/research && . .venv/bin/activate
cp /mnt/c/research/collect/make_game_list.py collect/
python collect/make_game_list.py --from_run data/main_3b/Qwen2.5-3B-Instruct --n 80 --seed 1 --stratify --max_episode 500 \
  --exclude data/multi_rollout_pilot_games/games_pilot.json --out data/multi_rollout_main_games/games_main.json
cp data/multi_rollout_main_games/games_main.json data/multi_rollout_main_games/games_main_meta.json /mnt/c/research/results/
