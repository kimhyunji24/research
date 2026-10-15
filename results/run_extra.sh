# 탐색적 추가 분석(reader 짝지은 차이, 후반 라운드, 시간축). WSL에서: bash results/run_extra.sh
set -u
cd ~/research && cp /mnt/c/research/probes/*.py probes/ && . .venv/bin/activate
python -W ignore probes/extra_analysis.py data/main_3b/Qwen2.5-3B-Instruct --out results/extra_main_3b > results/extra_main_3b.log 2>&1
tail -n 40 results/extra_main_3b.log
