"""모델 파일럿 포함 기준 1~4 판정 (docs/model_pilot_plan_2026-10-17.md)."""
import json, sys, numpy as np, re
D = sys.argv[1]; log = sys.argv[2]
eps = [json.loads(l) for l in open(f"{D}/episodes.jsonl")]
steps = [json.loads(l) for l in open(f"{D}/steps.jsonl")]
games = json.load(open("results/games_pilot_models_first20.json"))
s = sum(e["success"] for e in eps)
print(f"episodes={len(eps)} all games in list={set(e['task_id'] for e in eps)==set(games)} each once={len({e['task_id'] for e in eps})==len(eps)}")
print(f"C1 success: {s}/{len(eps)} -> {'PASS' if s >= 3 else 'FAIL'} (need >=3)")
fin = [l for l in open(log) if l.startswith("{")]
info = json.loads(fin[-1]) if fin else {}
oom = sum("out of memory" in l.lower() for l in open(log))
sec = info.get("sec"); per = sec / len(eps) if sec else None
print(f"C2 feasibility: completed={len(eps)==20} OOM lines={oom} sec_total={sec} sec/ep={per} peak_mem_gb={info.get('mem_gb')} layers={info.get('layers')}")
print(f"C3 valid actions: valid={sum(r['valid'] for r in steps)}/{len(steps)} ok(admissible)={sum(r['ok'] for r in steps)}/{len(steps)}")
h = np.concatenate([np.load(f"{D}/hidden_shards/ep{e['episode']:05d}.npy") for e in eps])
print(f"C4 hidden: rows={h.shape[0]} == steps {len(steps)}: {h.shape[0]==len(steps)} shape={h.shape[1:]} {h.dtype} NaN={int(np.isnan(h).sum())} inf={int(np.isinf(h).sum())} max_abs={float(np.abs(h.astype(np.float32)).max()):.1f}")
print("success steps:", sorted(e["steps"] for e in eps if e["success"]), "| per task_type:",
      {t: f"{sum(e['success'] for e in eps if e['task_type']==t)}/{sum(e['task_type']==t for e in eps)}" for t in sorted({e['task_type'] for e in eps})})
