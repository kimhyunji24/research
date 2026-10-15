# 확증 수집(rollout)을 코어 0-3에 고정하고, 일시 정지했던 프로브 작업을 코어 8-15, nice 15로 제한한 뒤 재개한다.
RP=$(pgrep -f "collect/rollout.py --config configs/main_3b.yaml" | head -1)
[ -n "$RP" ] && taskset -a -pc 0-3 $RP | tail -1
for p in $(pgrep -f "probes/run_probes.py|loky|run_explore_probes.sh"); do
  taskset -a -pc 8-15 $p >/dev/null 2>&1; renice -n 15 -p $p >/dev/null 2>&1
done
pkill -CONT -f "probes/run_probes.py"; pkill -CONT -f "loky"; pkill -CONT -f "run_explore_probes.sh"
sleep 2; ps -eo pid,stat,psr,ni,pcpu,args | grep -E "run_probes|rollout" | grep -v grep | cut -c1-100 | head -6
