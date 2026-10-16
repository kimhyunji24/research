#!/usr/bin/env bash
# 무작위 라벨 seed 20개로 within_game_supp.py의 null 보정 점검. 출력은 /tmp/null_calib (results/에 쓰지 않음)
cd ~/research
mkdir -p /tmp/null_calib
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
seq 1 20 | xargs -P 7 -I{} sh -c '.venv/bin/python -W ignore probes/within_game_supp.py \
  data/multi_rollout_pilot/Qwen2.5-3B-Instruct data/multi_rollout_main_r2/Qwen2.5-3B-Instruct \
  --layer 2 --steps 1 2 3 --n_boot 200 --n_perm 200 --shuffle_seed $((1000+{})) \
  --out /tmp/null_calib/s{} > /tmp/null_calib/s{}.log 2>&1; echo "seed {} exit=$?"'
ls /tmp/null_calib/*_diff.csv | wc -l
