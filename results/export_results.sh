# WSL의 results/ 산출물(CSV, 로그)을 Windows 저장소의 results/로 복사한다(데이터 원본 data/는 복사하지 않음).
cd ~/research/results
for f in explore_*.csv explore_*.log confirm_main_3b* probes_*.csv probe_main_3b_primary* baselines_main_3b* extra_main_3b* pilot_multi_rollout_summary.txt; do
  [ -f "$f" ] && cp -u "$f" /mnt/c/research/results/ && echo "copied $f"
done
