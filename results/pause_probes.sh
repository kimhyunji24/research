# CPU 경쟁으로 수집이 느려질 때 프로브 작업을 일시 정지한다. 재개: bash results/pause_probes.sh cont
SIG=${1:-STOP}
pkill -$SIG -f "probes/run_probes.py" ; pkill -$SIG -f "loky" ; pkill -$SIG -f "run_explore_probes.sh"
sleep 1; ps -eo pid,stat,pcpu,args | grep -E "run_probes|loky|rollout" | grep -v grep | cut -c1-110
