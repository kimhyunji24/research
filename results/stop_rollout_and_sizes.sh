cd ~/research
P=$(pgrep -f "collect/rollout.py --config configs/multi_rollout.yaml")
echo "rollout pids: $P"; for p in $P; do kill -TERM $p; done; sleep 5
echo "남은 rollout 프로세스: $(pgrep -f 'collect/rollout.py' | wc -l)"
f=data/multi_rollout_main/Qwen2.5-3B-Instruct
echo "multi_rollout_main: episodes $(wc -l < $f/episodes.jsonl), shards $(ls $f/hidden_shards | wc -l), steps $(wc -l < $f/steps.jsonl)"
bash /mnt/c/research/results/pause_7b.sh CONT
echo "--- data 크기"; du -sh data/* | sort -h; du -sh data/main_3b/readers/* 2>/dev/null
echo "--- confirm"; tail -n 3 results/confirm_main_3b.log
echo "--- git"; git status --short | head -8
