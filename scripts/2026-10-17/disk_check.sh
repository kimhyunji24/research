#!/usr/bin/env bash
# WSL ext4 쓰기/읽기 점검: 큰 파일 1개 + 작은 파일 다수 디렉터리, 캐시 비운 뒤 해시 재확인
set -u
d=/root/disk_check; rm -rf "$d"; mkdir -p "$d/many"
echo "kernel ext4 warnings before: $(dmesg | grep -c 'EXT4-fs')"
dd if=/dev/urandom of="$d/big.bin" bs=1M count=3072 status=none
h1=$(sha256sum "$d/big.bin" | cut -d' ' -f1)
for i in $(seq 1 20000); do head -c 2048 /dev/urandom > "$d/many/f$i"; done
(cd "$d/many" && find . -type f -print0 | sort -z | xargs -0 sha256sum > ../many.sha)
sync; echo 3 > /proc/sys/vm/drop_caches
h2=$(sha256sum "$d/big.bin" | cut -d' ' -f1)
[ "$h1" = "$h2" ] && echo "big file: OK" || echo "big file: MISMATCH"
(cd "$d/many" && sha256sum -c --quiet ../many.sha && echo "20000 small files: OK" || echo "small files: FAIL")
ls "$d/many" | wc -l
echo "kernel ext4 warnings after: $(dmesg | grep -c 'EXT4-fs')"
dmesg | grep -E 'EXT4-fs (warning|error)' | tail -3
rm -rf "$d"
