#!/bin/bash
# ReAct 수집 종료 뒤: 정합성 점검 -> sha256 목록 -> USB 복사·검증. 분석·commit 없음, 결과(성공률 등) 출력 없음.
cd ~/research && . .venv/bin/activate
echo "##### meta"; cat ~/react.meta
echo "##### log tail (errors only)"; grep -iE "traceback|error|out of memory|killed" ~/react.log | tail -5; echo "oom lines: $(grep -ci 'out of memory' ~/react.log)"
echo "ext4 warn/err: $(dmesg | grep -cE 'EXT4-fs (warning|error)')"
echo "##### 1. 정합성"
python -I /mnt/c/Users/catholic/AppData/Local/Temp/ws/check_react.py data/react_qwen25_3b/Qwen2.5-3B-Instruct results/games_main3b_first250.json
echo "##### 2. sha256"
out=results/sha256_react_qwen25_3b.txt
[ -e $out ] && { echo "$out exists, stop"; exit 1; }
find data/react_qwen25_3b -type f | sort | xargs sha256sum > $out
echo "files: $(wc -l < $out)  size: $(du -sh data/react_qwen25_3b | cut -f1)"
echo "##### 3. USB"
dst=/mnt/d/research_backup_2026-10-17/react_qwen25_3b
mountpoint -q /mnt/d || { mkdir -p /mnt/d; mount -t drvfs D: /mnt/d; }
[ -d /mnt/d/research_backup_2026-10-17 ] || { echo "USB not mounted, stop"; exit 1; }
[ -e "$dst" ] && { echo "$dst exists, stop"; exit 1; }
mkdir -p "$dst/data"
cp $out "$dst/SHA256SUMS_react_qwen25_3b.txt"
cp -r data/react_qwen25_3b "$dst/data/"
cp ~/react.log "$dst/react_qwen25_3b.log"; cp ~/react.meta "$dst/react_qwen25_3b.meta"
sync; echo 3 > /proc/sys/vm/drop_caches
cd "$dst" && sha256sum -c --quiet SHA256SUMS_react_qwen25_3b.txt && echo "backup verified: $(wc -l < SHA256SUMS_react_qwen25_3b.txt) files" || echo "BACKUP VERIFY FAILED"
