"""Determinism check before the ext1000 collection (docs/collect_ext1000_plan_2026-10-21.md): re-run the first 3 games of
main_3b with configs/ext1000_3b.yaml settings into /tmp and compare actions, success and hidden states with the stored run.

    .venv/bin/python scripts/2026-10-21/det_check.py make --out /tmp/det/games.json
    .venv/bin/python collect/rollout.py --config configs/ext1000_3b.yaml --game_list /tmp/det/games.json --out_dir /tmp/det
    .venv/bin/python scripts/2026-10-21/det_check.py compare --new /tmp/det/Qwen2.5-3B-Instruct
"""
import argparse, json, os
import numpy as np

REF = "data/main_3b/Qwen2.5-3B-Instruct"
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["make", "compare"])
ap.add_argument("--out")
ap.add_argument("--new")
a = ap.parse_args()
ref_eps = [json.loads(l) for l in open(f"{REF}/episodes.jsonl")]
if a.mode == "make":
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump([e["task_id"] for e in ref_eps[:3]], open(a.out, "w"), indent=1)
    print("games:", [e["task_id"].split("/")[-3] for e in ref_eps[:3]])
else:
    ref_steps = [json.loads(l) for l in open(f"{REF}/steps.jsonl")]
    new_eps = [json.loads(l) for l in open(f"{a.new}/episodes.jsonl")]
    new_steps = [json.loads(l) for l in open(f"{a.new}/steps.jsonl")]
    H0, H1 = np.load(f"{REF}/hidden.npy", mmap_mode="r"), np.load(f"{a.new}/hidden.npy")
    ok = True
    for e in new_eps:
        r = next(x for x in ref_eps if x["task_id"] == e["task_id"])
        rs = [s for s in ref_steps if s["episode"] == r["episode"]]
        ns = [s for s in new_steps if s["episode"] == e["episode"]]
        same_act = [s["action"] for s in rs] == [s["action"] for s in ns]
        d = max(float(np.abs(H0[x["hid_idx"]].astype(np.float32) - H1[y["hid_idx"]].astype(np.float32)).max()) for x, y in zip(rs, ns))
        same_tok = all(x["prompt_tokens"] == y["prompt_tokens"] for x, y in zip(rs, ns))
        print(f"{e['task_id'].split('/')[-3][:60]:60s} steps {len(rs)}/{len(ns)} actions_equal={same_act} "
              f"success {r['success']}/{e['success']} prompt_tokens_equal={same_tok} max|hidden diff|={d:.4g}")
        ok &= same_act and r["success"] == e["success"] and same_tok and len(rs) == len(ns) and d == 0.0
    print("DETERMINISM", "OK" if ok else "FAILED")
