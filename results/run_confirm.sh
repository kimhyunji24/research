# 확증 분석 (docs/confirmatory_plan_2026-10-15.md): 3B reader, 75% layer, linear vs baseline 5단, 평가는 episode>=250만.
set -u
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd ~/research && cp /mnt/c/research/probes/*.py probes/ && . .venv/bin/activate
python -c "import numpy as np;h=np.load('data/main_3b/Qwen2.5-3B-Instruct/hidden.npy',mmap_mode='r');print('hidden',h.shape)"
echo "--- 코드 경로 점검 (baseline 1단, 순열 2회; 결과는 의미 없음)"
taskset -c 8-23 nice -n 10 python -W ignore probes/run_probes.py data/main_3b/Qwen2.5-3B-Instruct --baselines_only --rungs 1 --rounds 0 \
  --eval_from_episode 250 --n_perm 2 --n_boot 20 --n_jobs 4 2>&1 | grep -E "episodes=|step +method|baseline1" | head -5
echo "--- 확증 실행"
taskset -c 8-23 nice -n 10 python -W ignore probes/run_probes.py data/main_3b/Qwen2.5-3B-Instruct --layer 2 --rounds 0 1 2 \
  --kinds linear --pca 0 --rungs 5 --eval_from_episode 250 --n_perm 100 --n_jobs 8 \
  --out results/confirm_main_3b.csv > results/confirm_main_3b.log 2>&1
tail -n 14 results/confirm_main_3b.log
