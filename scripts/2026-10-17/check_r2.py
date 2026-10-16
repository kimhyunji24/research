"""multi_rollout_main_r2 정합성: episode 수, 게임당 반복, steps/hidden 대응, NaN."""
import collections, json, numpy as np, os
B = "data/multi_rollout_main_r2/Qwen2.5-3B-Instruct"
games = json.load(open("data/multi_rollout_main_games/games_main.json"))
eps = [json.loads(l) for l in open(f"{B}/episodes.jsonl")]
steps = [json.loads(l) for l in open(f"{B}/steps.jsonl")]
print("episodes", len(eps), "episode ids contiguous", [e["episode"] for e in eps] == list(range(len(eps))))
cnt = collections.Counter(e["task_id"] for e in eps)
print("games", len(cnt), "all in list", set(cnt) == set(games), "repeats", sorted(set(cnt.values())))
per_ep = collections.Counter(r["episode"] for r in steps)
print("steps rows", len(steps), "steps match episodes.steps", all(per_ep[e["episode"]] == e["steps"] for e in eps))
n_hid, nan = 0, 0
for e in eps:
    h = np.load(f"{B}/hidden_shards/ep{e['episode']:05d}.npy")
    assert h.shape[0] == e["steps"], (e["episode"], h.shape)
    n_hid += h.shape[0]; nan += int(np.isnan(h.astype(np.float32)).sum())
print("hidden rows", n_hid, "shape tail", h.shape[1:], h.dtype, "NaN", nan)
print("hid_idx max", max(r["hid_idx"] for r in steps), "== rows-1", max(r["hid_idx"] for r in steps) == n_hid - 1)
print("shards on disk", len(os.listdir(f"{B}/hidden_shards")))
