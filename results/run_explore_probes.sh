# 탐색적 프로브: reader 4종 x layer 4종 x linear/MLP x step 0,1,2. baseline 5단과 짝지은 차이.
set -u
cd ~/research && . .venv/bin/activate
RUN=data/main_3b/Qwen2.5-3B-Instruct
mkdir -p results
declare -A HID=( [3B]=$RUN/hidden.npy [0.5B]=data/main_3b/readers/Qwen2.5-0.5B-Instruct/hidden.npy
                 [1.5B]=data/main_3b/readers/Qwen2.5-1.5B-Instruct/hidden.npy [7B-4bit]=data/main_3b/readers/Qwen2.5-7B-Instruct-4bit/hidden.npy )
for k in 3B 0.5B 1.5B 7B-4bit; do
  [ -f "${HID[$k]}" ] || { echo "skip $k: no hidden"; continue; }
  python -W ignore probes/run_probes.py $RUN --hidden "${HID[$k]}" --layers 0 1 2 3 --kinds linear mlp --rungs 5 \
    --rounds 0 1 2 --n_perm 100 --out results/explore_${k}.csv > results/explore_${k}.log 2>&1
  echo "done $k $(date)"
done
