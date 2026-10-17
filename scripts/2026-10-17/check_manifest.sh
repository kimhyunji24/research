#!/usr/bin/env bash
cd ~/research
bad=0; n=0
while IFS=$'\t' read -r size path; do
  [[ "$size" =~ ^[0-9]+$ ]] || continue
  n=$((n+1))
  a=$(stat -c %s "$path" 2>/dev/null || echo missing)
  [ "$a" = "$size" ] || { echo "MISMATCH $path $size $a"; bad=$((bad+1)); }
done < <(tr -d '\r' < /mnt/c/Users/catholic/Desktop/research_data/MANIFEST.txt)
echo "entries=$n problems=$bad"
du -sh data
