# 확증 수집(main_3b 500 episode)이 끝나면 다중 rollout 파일럿(20게임 x 8회, 층화)을 자동으로 시작한다.
set -u
cd ~/research && . .venv/bin/activate && export ALFWORLD_DATA=/root/alfworld_data
while pgrep -f "collect/rollout.py --config configs/main_3b.yaml" > /dev/null; do sleep 20; done
echo "confirm collection finished $(date): $(wc -l < data/main_3b/Qwen2.5-3B-Instruct/episodes.jsonl) episodes"
cp /mnt/c/research/collect/rollout.py /mnt/c/research/collect/make_game_list.py collect/
cp /mnt/c/research/configs/multi_rollout.yaml configs/
python collect/make_game_list.py --from_run data/main_3b/Qwen2.5-3B-Instruct --n 20 --seed 0 --stratify --max_episode 250 \
  --out data/multi_rollout_pilot_games/games_pilot.json
python -W ignore collect/rollout.py --config configs/multi_rollout.yaml --game_list data/multi_rollout_pilot_games/games_pilot.json \
  --repeats 8 --out_dir data/multi_rollout_pilot 2>&1 | grep -E "^\[Qwen|^\{|Error|Traceback" | tail -200
echo "pilot finished $(date)"
