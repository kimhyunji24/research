"""다중 rollout용 게임 목록을 만든다. 기존 수집 결과(episodes.jsonl)의 게임 중에서 고른다.

사용: python collect/make_game_list.py --from_run data/main_3b/Qwen2.5-3B-Instruct --n 20 --seed 0 [--stratify] --out data/multi_rollout/games_pilot.json
--stratify: 탐욕 결과가 성공인 게임과 실패인 게임을 반반씩 뽑는다(사후 선택임을 기록한다).
--max_episode: 이 번호 미만의 episode에서만 고른다(예: 250, 확증용 새 episode와 섞이지 않게).
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--from_run", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--stratify", action="store_true")
ap.add_argument("--max_episode", type=int, default=250)
ap.add_argument("--out", required=True)
a = ap.parse_args()

E = [json.loads(l) for l in open(os.path.join(a.from_run, "episodes.jsonl"))]
E = [e for e in E if e["episode"] < a.max_episode]
rng = np.random.default_rng(a.seed)
if a.stratify:
    pos = [e for e in E if e["success"]]; neg = [e for e in E if not e["success"]]
    k = a.n // 2
    pick = [pos[i] for i in rng.choice(len(pos), k, replace=False)] + [neg[i] for i in rng.choice(len(neg), a.n - k, replace=False)]
else:
    pick = [E[i] for i in rng.choice(len(E), a.n, replace=False)]
games = [e["task_id"] for e in pick]
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(games, open(a.out, "w"), indent=1)
json.dump([dict(task_id=e["task_id"], greedy_success=e["success"], greedy_steps=e["steps"], task_type=e["task_type"])
           for e in pick], open(a.out.replace(".json", "_meta.json"), "w"), indent=1, ensure_ascii=False)
print(f"{len(games)} games -> {a.out}  (탐욕 성공 {sum(e['success'] for e in pick)}, 실패 {sum(not e['success'] for e in pick)})")
