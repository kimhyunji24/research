#!/usr/bin/env bash
# RESTART.md 2-3절 재현 (2026-10-16)
set -euo pipefail
printf '[user]\ndefault=root\n' > /etc/wsl.conf

echo "== apt"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git git-lfs curl build-essential unzip rsync >/dev/null

echo "== uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

echo "== code"
if [ ! -d ~/research ]; then
  git clone https://github.com/kimhyunji24/research.git ~/research
fi
cd ~/research
git switch probes-baseline 2>/dev/null || git switch -c probes-baseline
git log --oneline -1

echo "== data (copy from Windows restore, data/ must not exist yet)"
if [ -e data ]; then echo "data/ already exists, skip copy"; else
  rsync -a /mnt/c/Users/catholic/Desktop/research_data/data/ ./data/
fi
sync; echo 3 > /proc/sys/vm/drop_caches
src=/mnt/c/Users/catholic/Desktop/research_data
bad=0; ok=0
while IFS=$'\t' read -r size path; do
  [[ "$size" =~ ^[0-9]+$ ]] || continue
  if cmp -s "$src/$path" "$path"; then ok=$((ok+1)); else echo "DIFF $path"; bad=$((bad+1)); fi
done < <(tr -d '\r' < "$src/MANIFEST.txt")
echo "data byte-compare same=$ok diff=$bad"
git fsck --no-progress && echo "git fsck OK"
# 변경 이력(Windows clone에서 편집, CRLF 제거해 반영)
tr -d '\r' < /mnt/c/Users/catholic/Desktop/research/docs/multi_rollout_plan_2026-10-15.md > docs/multi_rollout_plan_2026-10-15.md
git diff --stat

echo "== venv"
uv venv --python 3.11 .venv
. .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers==5.19.0 accelerate scikit-learn numpy pandas pyyaml tqdm alfworld==0.4.2 bitsandbytes
python -c "import torch, transformers, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('transformers', transformers.__version__)"

echo "== alfworld data"
export ALFWORLD_DATA=/root/alfworld_data
grep -q ALFWORLD_DATA ~/.bashrc || echo 'export ALFWORLD_DATA=/root/alfworld_data' >> ~/.bashrc
if [ ! -d /root/alfworld_data/json_2.1.1 ]; then alfworld-download; fi
du -sh /root/alfworld_data
sync
echo "ext4 warnings/errors: $(dmesg | grep -cE 'EXT4-fs (warning|error)')"
echo "== done"
