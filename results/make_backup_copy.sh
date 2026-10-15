# data/를 Windows 쪽 백업 폴더로 복사한다(hidden_shards는 hidden.npy와 중복이라 제외, 단 중단된 multi_rollout_main은 포함).
set -e
B=/mnt/c/research_backup
mkdir -p $B/data
cd ~/research/data
for d in main_3b decision_rerun maxstep50 multi_rollout_pilot multi_rollout_main multi_rollout_pilot_games multi_rollout_main_games; do
  rm -rf $B/data/$d; cp -r $d $B/data/$d
done
find $B/data -name hidden_shards -not -path '*multi_rollout_main/*' -prune -exec rm -rf {} +
cd $B && du -sh data data/* | sort -h && find data -type f | wc -l
