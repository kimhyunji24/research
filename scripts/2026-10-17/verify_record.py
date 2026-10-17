"""site-packages의 dist-info/RECORD 해시로 설치 파일 무결성 확인 (torch, nvidia-*, transformers)."""
import base64, csv, hashlib, pathlib, sys
sp = pathlib.Path(sys.argv[1])
bad = ok = 0
for rec in sorted(sp.glob("*.dist-info/RECORD")):
    name = rec.parent.name.lower()
    if not (name.startswith(("torch-", "nvidia_", "transformers-", "triton-"))):
        continue
    for row in csv.reader(open(rec)):
        if len(row) < 3 or not row[1].startswith("sha256="):
            continue
        f = sp / row[0]
        if not f.exists():
            continue
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        d = base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()
        if d == row[1][7:]:
            ok += 1
        else:
            bad += 1
            print("BAD", row[0], flush=True)
print(f"ok={ok} bad={bad}")
