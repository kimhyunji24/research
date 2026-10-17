#!/usr/bin/env bash
# multi_rollout_main_r2를 Windows 새 폴더로 백업하고 sha256으로 검증한다(기존 백업 폴더는 건드리지 않음)
set -euo pipefail
dst=/mnt/c/Users/catholic/Desktop/research_data_new
[ -e "$dst/data/multi_rollout_main_r2" ] && { echo "already exists, stop"; exit 1; }
mkdir -p "$dst/data"
cd ~/research
find data/multi_rollout_main_r2 -type f | sort | xargs sha256sum > /root/r2.sha256
cp /root/r2.sha256 "$dst/SHA256SUMS_multi_rollout_main_r2.txt"
cp -r data/multi_rollout_main_r2 "$dst/data/"
cp ~/multi_rollout_main_r2.log "$dst/multi_rollout_main_r2.log"
cd "$dst" && sha256sum -c --quiet SHA256SUMS_multi_rollout_main_r2.txt && echo "backup verified: $(wc -l < SHA256SUMS_multi_rollout_main_r2.txt) files"
du -sh "$dst/data/multi_rollout_main_r2"
