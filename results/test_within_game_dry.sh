# 코드 경로 점검: 파일럿 데이터에서 baseline2(게임 안 상수)와 난이도 오라클만. 프로브는 돌리지 않는다.
cd ~/research && . .venv/bin/activate && cp /mnt/c/research/probes/*.py probes/
cp /mnt/c/research/results/games_pilot_meta.json data/multi_rollout_pilot_games/ 2>/dev/null
python -W ignore probes/within_game.py data/multi_rollout_pilot/Qwen2.5-3B-Instruct --methods baseline2 --steps 1 \
  --game_meta data/multi_rollout_pilot_games/games_pilot_meta.json --n_perm 20 --n_boot 100 2>&1 | tail -12
