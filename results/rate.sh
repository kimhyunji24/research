f=~/research/data/multi_rollout_main/Qwen2.5-3B-Instruct/episodes.jsonl
a=$(wc -l < $f); t0=$(date +%s); sleep ${1:-60}; b=$(wc -l < $f); t1=$(date +%s)
echo "$(date +%H:%M:%S) episodes $a -> $b in $((t1-t0))s => $(( (t1-t0)*1000/((b-a)>0?(b-a):1) ))ms/episode, 남은 $((640-b))개"
