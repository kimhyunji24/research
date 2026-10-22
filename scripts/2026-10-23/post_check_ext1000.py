"""Post-collection checks for data/ext1000_3b (docs/collect_ext1000_plan_2026-10-21.md). Counts and integrity only; no outcomes printed."""
import hashlib, json, os, collections
import numpy as np

D = "data/ext1000_3b/Qwen2.5-3B-Instruct"
eps = [json.loads(l) for l in open(f"{D}/episodes.jsonl")]
steps = [json.loads(l) for l in open(f"{D}/steps.jsonl")]
H = np.load(f"{D}/hidden.npy", mmap_mode="r")
games = json.load(open("data/ext1000_3b/games.json"))
ref = {json.loads(l)["task_id"] for l in open("data/main_3b/Qwen2.5-3B-Instruct/episodes.jsonl")}
ids = [e["task_id"] for e in eps]
print("step row keys:", sorted(steps[0].keys()))
print("episodes", len(eps), "| episode indices 0..n-1:", [e["episode"] for e in eps] == list(range(len(eps))))
print("unique games", len(set(ids)), "| overlap with main_3b", len(set(ids) & ref), "| set == games.json", set(ids) == set(games), "| order == games.json", ids == games)
print("hidden", H.shape, H.dtype, "| steps rows", len(steps), "| hidden rows == steps rows", H.shape[0] == len(steps))
print("hid_idx == 0..n-1", [s["hid_idx"] for s in steps] == list(range(len(steps))))
print("prompt_tokens present in all rows", all(s.get("prompt_tokens") for s in steps))
print("finite hidden (first/last 2000 rows)", bool(np.isfinite(H[:2000].astype(np.float32)).all() and np.isfinite(H[-2000:].astype(np.float32)).all()))
skey = "step" if "step" in steps[0] else None
if skey:
    c = collections.Counter(s[skey] for s in steps)
    print("rows at step 0,1,2:", c[0], c[1], c[2], "| episodes with >=3 steps:", sum(1 for e in eps if sum(1 for s in steps if s["episode"] == e["episode"]) >= 3) if False else "see below")
    n_by_ep = collections.Counter(s["episode"] for s in steps)
    print("episodes with >=1,>=2,>=3 steps:", sum(v >= 1 for v in n_by_ep.values()), sum(v >= 2 for v in n_by_ep.values()), sum(v >= 3 for v in n_by_ep.values()))
shards = sorted(os.listdir(f"{D}/hidden_shards"))
print("shards", len(shards), "| shards rows total", sum(np.load(f"{D}/hidden_shards/{f}", mmap_mode='r').shape[0] for f in shards))
for f in ("episodes.jsonl", "steps.jsonl", "hidden.npy"):
    h = hashlib.sha256(open(f"{D}/{f}", "rb").read()).hexdigest()
    print("sha256", f, h)
print("games.json sha256", open("data/ext1000_3b/games.json.sha256").read().strip())
