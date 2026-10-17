"""horizon100 사용 전 점검 1, 2 (성공 여부는 출력하지 않는다).
1) 완료된 episode마다 step 0~29의 action 열이 main_3b와 같고 hidden(4 layer)이 최대 차이 0
2) maxstep50의 실패 게임은 step 0~49 action 열이 같다(해당 게임이 완료된 경우만)"""
import json, os, numpy as np
H = "data/horizon100/Qwen2.5-3B-Instruct"; M = "data/main_3b/Qwen2.5-3B-Instruct"; X = "data/maxstep50/Qwen2.5-3B-Instruct"
eh = [json.loads(l) for l in open(f"{H}/episodes.jsonl")]
em = {e["task_id"]: e for e in map(json.loads, open(f"{M}/episodes.jsonl"))}
def acts(path, keep):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if r["episode"] in keep: out.setdefault(r["episode"], []).append((r["step"], r["action"], r["hid_idx"]))
    return out
ah = acts(f"{H}/steps.jsonl", {e["episode"] for e in eh})
am = acts(f"{M}/steps.jsonl", {em[e["task_id"]]["episode"] for e in eh})
hm = np.load(f"{M}/hidden.npy", mmap_mode="r")
act_bad, hid_bad, maxd = [], [], 0.0
for e in eh:
    a = sorted(ah[e["episode"]])[:30]; b = sorted(am[em[e["task_id"]]["episode"]])[:30]
    if [x[1] for x in a] != [x[1] for x in b]: act_bad.append(e["episode"]); continue
    hh = np.load(f"{H}/hidden_shards/ep{e['episode']:05d}.npy")[:30].astype(np.float32)
    d = float(np.abs(hh - hm[[x[2] for x in b]].astype(np.float32)).max()); maxd = max(maxd, d)
    if d != 0.0: hid_bad.append(e["episode"])
print(f"check1: episodes done={len(eh)}, first-30 action mismatches={len(act_bad)} {act_bad[:5]}, hidden mismatches={len(hid_bad)}, max abs diff={maxd}")
ex = [json.loads(l) for l in open(f"{X}/episodes.jsonl")]
xf = {e["task_id"]: e["episode"] for e in ex if e["steps"] > 30 or not e["success"]}
ax = acts(f"{X}/steps.jsonl", set(xf.values()))
n = bad = 0
for e in eh:
    if e["task_id"] in xf:
        n += 1
        a = [x[1] for x in sorted(ah[e["episode"]])[:50]]; b = [x[1] for x in sorted(ax[xf[e["task_id"]]])[:50]]
        L = min(len(a), len(b))
        bad += a[:L] != b[:L]
print(f"check2: maxstep50 overlap games done={n} (of {len(xf)}), first-50 action mismatches={bad}")
