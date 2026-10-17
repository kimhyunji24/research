import json, numpy as np
d = "data/multi_rollout_main_r2/Qwen2.5-3B-Instruct"
eps = [json.loads(l) for l in open(d + "/episodes.jsonl")]
c = np.concatenate([np.load(f"{d}/hidden_shards/ep{e['episode']:05d}.npy") for e in eps])
h = np.load(d + "/hidden.npy", mmap_mode="r")
print("shape", h.shape, c.shape, "identical", np.array_equal(h, c))
