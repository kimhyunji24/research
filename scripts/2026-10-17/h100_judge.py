"""시간 한도 100 늦은 성공 판정 (docs/horizon100_plan_2026-10-17.md 해석 규칙, 0238950)."""
import json, math, collections
H = "data/horizon100/Qwen2.5-3B-Instruct"
eps = [json.loads(l) for l in open(f"{H}/episodes.jsonl")]
meta = {m["task_id"]: m for m in json.load(open("data/horizon100_games/games_meta.json"))}
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c - h, c + h
succ = [e for e in eps if e["success"]]
early = [e for e in succ if e["steps"] <= 30]
late = [e for e in succ if e["steps"] > 30]
n = len(eps); k = len(late); lo, hi = wilson(k, n)
print(f"N={n} successes within 100={len(succ)} (<=30 steps: {len(early)}, must be 0) late(31-100)={k}")
print(f"L = {k}/{n} = {k/n:.4f}  Wilson 95% CI {lo:.4f} ~ {hi:.4f}")
rule = "CI upper < 5%: 30-step label is a good proxy" if hi < 0.05 else ("CI lower >= 5%: change label definition to 100 steps" if lo >= 0.05 else "CI includes 5%: cannot conclude; report sensitivity analyses alongside")
print("RULE:", rule)
for half in ("explore", "confirm"):
    sub = [e for e in eps if meta[e["task_id"]]["half"] == half]; kk = sum(e["success"] and e["steps"] > 30 for e in sub); a, b = wilson(kk, len(sub))
    print(f"  {half}: {kk}/{len(sub)} = {kk/len(sub):.4f} (CI {a:.4f}~{b:.4f})")
tt = collections.defaultdict(lambda: [0, 0])
for e in eps: tt[e["task_type"]][1] += 1; tt[e["task_type"]][0] += e["success"] and e["steps"] > 30
print("  by task type:", {t: f"{v[0]}/{v[1]}" for t, v in sorted(tt.items())})
print("  late success steps:", sorted(e["steps"] for e in late))
print("  successes within 50 steps (31-50):", sum(31 <= e["steps"] <= 50 for e in late))
x = [json.loads(l) for l in open("data/maxstep50/Qwen2.5-3B-Instruct/episodes.jsonl")]
xf = {e["task_id"] for e in x if e["steps"] > 30 or not e["success"]}
ov = [e for e in eps if e["task_id"] in xf]
print(f"  maxstep50 overlap games: {len(ov)}, h100 successes in 31-50 among them: {sum(e['success'] and 31 <= e['steps'] <= 50 for e in ov)}, maxstep50 successes in 31-50: {sum(e['success'] and e['steps'] > 30 for e in x if e['task_id'] in xf)}")
print("  first 116 episodes late successes (prior exposure check):", sum(e["success"] and e["steps"] > 30 for e in eps[:116]), "steps", sorted(e["steps"] for e in eps[:116] if e["success"]))
