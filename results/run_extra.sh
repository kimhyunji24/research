# 탐색적 추가 분석(reader 짝지은 차이, 후반 라운드, 시간축). 수집과 CPU를 나눠 쓰므로 코어와 스레드를 제한한다.
# WSL에서: bash results/run_extra.sh
set -u
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd ~/research && cp /mnt/c/research/probes/*.py probes/ && . .venv/bin/activate
taskset -c 16-23 nice -n 15 python -W ignore probes/extra_analysis.py data/main_3b/Qwen2.5-3B-Instruct --out results/extra_main_3b > results/extra_main_3b.log 2>&1
tail -n 40 results/extra_main_3b.log
