# 7B 탐색 프로브(탐색적)만 일시 정지. 재개: bash results/pause_7b.sh CONT
SIG=${1:-STOP}
P=$(pgrep -f "probes/run_probes.py data/main_3b/Qwen2.5-3B-Instruct --hidden data/main_3b/readers/Qwen2.5-7B")
for p in $P; do kill -$SIG $p; done
# 이 작업의 loky worker(자식 프로세스)도 같이
for p in $P; do for c in $(pgrep -P $p); do kill -$SIG $c; done; done
echo "$SIG: $P"
