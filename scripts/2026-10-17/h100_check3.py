"""horizon100 사전 기록 점검 3 (정합성). 성공 여부·성공 step은 출력하지 않는다."""
import collections, json, os, numpy as np
H = "data/horizon100/Qwen2.5-3B-Instruct"
games = json.load(open("data/horizon100_games/games.json"))
eps = [json.loads(l) for l in open(f"{H}/episodes.jsonl")]
steps = [json.loads(l) for l in open(f"{H}/steps.jsonl")]
cnt = collections.Counter(e["task_id"] for e in eps)
print("episodes", len(eps), "(expect 399)", "ids contiguous", [e["episode"] for e in eps] == list(range(len(eps))))
print("games", len(cnt), "all in list", set(cnt) == set(games), "each once", set(cnt.values()) == {1})
print("max steps <= 100", max(e["steps"] for e in eps) <= 100)
per_ep = collections.Counter(r["episode"] for r in steps)
print("steps rows", len(steps), "match episodes.steps", all(per_ep[e["episode"]] == e["steps"] for e in eps))
n_hid = nan = 0; shape = None
for e in eps:
    h = np.load(f"{H}/hidden_shards/ep{e['episode']:05d}.npy")
    assert h.shape[0] == e["steps"], (e["episode"], h.shape)
    n_hid += h.shape[0]; shape = (h.shape[1:], h.dtype); nan += int(np.isnan(h.astype(np.float32)).sum())
print("hidden rows", n_hid, "== steps rows", n_hid == len(steps), "shape tail", shape, "NaN", nan)
print("shards on disk", len(os.listdir(f"{H}/hidden_shards")))
if os.path.exists(f"{H}/hidden.npy"):
    m = np.load(f"{H}/hidden.npy", mmap_mode="r"); print("merged hidden.npy rows", m.shape[0], "== rows", m.shape[0] == n_hid)
