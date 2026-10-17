#!/usr/bin/env bash
# WSL ~/research/data 와 Windows 복구본을 바이트 단위로 비교
src=/mnt/c/Users/catholic/Desktop/research_data
cd ~/research
ok=0; bad=0
while IFS=$'\t' read -r size path; do
  [[ "$size" =~ ^[0-9]+$ ]] || continue
  if cmp -s "$src/$path" "$path"; then ok=$((ok+1)); else echo "DIFF/ERR $path"; bad=$((bad+1)); fi
done < <(tr -d '\r' < "$src/MANIFEST.txt")
echo "same=$ok diff=$bad"
echo "== git fsck (code repo)"
git fsck --no-progress 2>&1 | tail -5
echo "== kernel fs warnings so far: $(dmesg | grep -c 'EXT4-fs')"
