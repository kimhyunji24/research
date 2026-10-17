"""백업(multi_rollout_main, 24 ep)과 새 수집(multi_rollout_main_r2)의 episode 0..23 비교: 게임, 성공, step 수, action 열, hidden shard."""
import json, numpy as np, os
A = "data/multi_rollout_main/Qwen2.5-3B-Instruct"
B = "data/multi_rollout_main_r2/Qwen2.5-3B-Instruct"
ea = [json.loads(l) for l in open(f"{A}/episodes.jsonl")]
eb = [json.loads(l) for l in open(f"{B}/episodes.jsonl")][:len(ea)]
assert len(eb) == len(ea), (len(ea), len(eb))
keys = ["task_id", "success", "steps", "rollout_idx"]
meta_diff = [i for i, (x, y) in enumerate(zip(ea, eb)) if any(x[k] != y[k] for k in keys)]

def actions(path, n):
    acts = {}
    for l in open(path):
        r = json.loads(l)
        if r["episode"] < n:
            acts.setdefault(r["episode"], []).append(r.get("action"))
    return acts
aa, ab = actions(f"{A}/steps.jsonl", len(ea)), actions(f"{B}/steps.jsonl", len(ea))
act_diff = [e for e in range(len(ea)) if aa.get(e) != ab.get(e)]
hid_exact, hid_maxdiff = 0, 0.0
for e in range(len(ea)):
    ha = np.load(f"{A}/hidden_shards/ep{e:05d}.npy").astype(np.float32)
    hb = np.load(f"{B}/hidden_shards/ep{e:05d}.npy").astype(np.float32)
    if ha.shape == hb.shape:
        d = float(np.abs(ha - hb).max()); hid_maxdiff = max(hid_maxdiff, d); hid_exact += d == 0.0
    else:
        print("shape diff ep", e, ha.shape, hb.shape)
print(f"episodes compared={len(ea)}")
print(f"meta (game/success/steps) mismatches={len(meta_diff)} {meta_diff}")
print(f"action sequence mismatches={len(act_diff)} {act_diff}")
print(f"hidden shards exactly equal={hid_exact}/{len(ea)}, max abs diff={hid_maxdiff:.4g}")
print("success backup", sum(e['success'] for e in ea), "new", sum(e['success'] for e in eb))
