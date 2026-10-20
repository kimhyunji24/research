"""New, unused train games for docs/collect_ext1000_plan_2026-10-21.md: the ALFWorld train pool exactly as the rollout's
environment builds it, minus the games already in --exclude_run, sampled without replacement (no stratification).

    .venv/bin/python collect/make_new_game_list.py --exclude_run data/main_3b/Qwen2.5-3B-Instruct --n 1000 --seed 0 \
        --out data/ext1000_3b/games.json
Writes the list and a sha256 of it; stops if the output exists.
"""
import argparse, hashlib, json, os
import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/main_3b.yaml")
ap.add_argument("--exclude_run", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if os.path.exists(a.out): raise SystemExit(f"output exists: {a.out}")
cfg = yaml.safe_load(open(a.config))
from alfworld.agents.environment import get_environment  # noqa: E402
env_cfg = yaml.safe_load(os.path.expandvars(open(cfg["alfworld_config"]).read()))
pool = sorted(get_environment(env_cfg["env"]["type"])(env_cfg, train_eval=cfg["split"]).game_files)
used = {json.loads(l)["task_id"] for l in open(os.path.join(a.exclude_run, "episodes.jsonl"))}
assert used <= set(pool), "excluded games are not in the pool (path format differs)"
rest = [g for g in pool if g not in used]
pick = [rest[i] for i in np.random.default_rng(a.seed).choice(len(rest), a.n, replace=False)]
assert len(set(pick)) == a.n and not (set(pick) & used)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(pick, open(a.out, "w"), indent=1)
h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
open(a.out + ".sha256", "w").write(h + "\n")
print(f"pool {len(pool)}, excluded {len(used)}, remaining {len(rest)}, picked {len(pick)} (seed {a.seed}) -> {a.out} sha256 {h}")
