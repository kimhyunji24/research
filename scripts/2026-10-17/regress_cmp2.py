"""회귀 점검(게임 목록 기준): task_id로 main_3b episode와 짝지어 prompt, action, 후보 점수, hidden을 비교한다."""
import json, sys, numpy as np
A = "data/main_3b/Qwen2.5-3B-Instruct"; B = sys.argv[1]
ea = {e["task_id"]: e for e in map(json.loads, open(f"{A}/episodes.jsonl"))}
eb = [json.loads(l) for l in open(f"{B}/episodes.jsonl")]
pairs = [(ea[e["task_id"]]["episode"], e["episode"]) for e in eb]
meta = sum(any(ea[e["task_id"]][k] != e[k] for k in ("success", "steps")) for e in eb)
def rows(p, keep):
    out = {}
    for r in map(json.loads, open(p)):
        if r["episode"] in keep: out.setdefault(r["episode"], []).append(r)
    return out
ra = rows(f"{A}/steps.jsonl", {a for a, _ in pairs}); rb = rows(f"{B}/steps.jsonl", {b for _, b in pairs})
hA = np.load(f"{A}/hidden.npy", mmap_mode="r")
keys = ("action", "prompt", "cands", "cand_logp", "cand_ntok", "p_chosen")
diff = dict.fromkeys(keys, 0); n = 0; maxd = 0.0
for a, b in pairs:
    x, y = sorted(ra[a], key=lambda r: r["step"]), sorted(rb[b], key=lambda r: r["step"])
    assert len(x) == len(y), (a, b, len(x), len(y))
    for k in keys: diff[k] += sum(p[k] != q[k] for p, q in zip(x, y))
    hb = np.load(f"{B}/hidden_shards/ep{b:05d}.npy").astype(np.float32)
    maxd = max(maxd, float(np.abs(hA[[r["hid_idx"] for r in x]].astype(np.float32) - hb).max())); n += len(x)
print(f"episodes={len(eb)} steps={n} meta_mismatch={meta} field_mismatch={diff} hidden_max_abs_diff={maxd}")
