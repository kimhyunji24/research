# 탐색적 확장 실행 스크립트 (docs/analysis_plan_2026-10-15.md 마지막 절). WSL에서: bash results/run_explore.sh
set -u
cd ~/research && . .venv/bin/activate && export ALFWORLD_DATA=/root/alfworld_data
RUN=data/main_3b/Qwen2.5-3B-Instruct
F="python -W ignore"
for r in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct; do
  $F collect/extract_reader.py --run $RUN --reader $r 2>&1 | grep -v -i 'warn\|unauth\|Loading' | tail -6
done
$F collect/extract_reader.py --run $RUN --reader Qwen/Qwen2.5-7B-Instruct --load_4bit 2>&1 | grep -v -i 'warn\|unauth\|Loading' | tail -6
echo "=== extraction done $(date)"
