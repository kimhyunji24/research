"""조기 중단 설계용 사실(탐색 구간 episode 0~249만): 실패/성공 길이, step cap 단순 규칙의 비용-손실, step당 토큰."""
import json, numpy as np
d = "data/main_3b/Qwen2.5-3B-Instruct"
eps = [json.loads(l) for l in open(f"{d}/episodes.jsonl")][:250]
S = np.array([e["steps"] for e in eps if e["success"]]); F = np.array([e["steps"] for e in eps if not e["success"]])
print(f"n=250 success={len(S)} fail={len(F)}")
print("fail steps: unique", np.unique(F, return_counts=True))
print("success steps: min/median/max", S.min(), np.median(S), S.max(), "| quantiles 80/90/95:", np.percentile(S, [80, 90, 95]))
tot = S.sum() + F.sum()
print(f"total steps={tot}, failures' share={F.sum()/tot:.3f}")
for cap in (10, 12, 15, 18, 20, 24):
    lost = (S > cap).sum(); cost = np.minimum(S, cap).sum() + np.minimum(F, cap).sum()
    print(f"cap={cap:2d}: successes lost {lost}/{len(S)} ({lost/len(S):.1%}), steps saved {1-cost/tot:.1%}")
tok = {}
for l in open(f"{d}/steps.jsonl"):
    r = json.loads(l)
    if r["episode"] < 250:
        tok.setdefault("prompt", []).append(r.get("prompt_tokens", np.nan)); tok.setdefault("n_cand", []).append(r.get("n_cand", np.nan))
print("prompt tokens/step median", np.nanmedian(tok["prompt"]), "| n_cand median", np.nanmedian(tok["n_cand"]))
