# data-backup (2026-10-15)

코드 저장소(`probes-baseline`)에 없는 수집 데이터의 백업. 큰 파일은 Git LFS(`*.npy`, `*.jsonl`).
복구: 이 브랜치를 받아 `data/` 폴더를 코드 저장소의 `data/`에 복사한다(`docs/RESTART.md` 참고).
`hidden_shards/`는 `hidden.npy`와 중복이라 제외했고, 중단된 `multi_rollout_main`만 shard를 포함한다.
파일 목록과 크기는 `MANIFEST.txt`.
