"""main_3b 500 episode 중 탐욕 실패 게임 목록(시간 한도 100 재수집용). 새 폴더 data/horizon100_games에 쓴다."""
import json, os, collections
E = [json.loads(l) for l in open("data/main_3b/Qwen2.5-3B-Instruct/episodes.jsonl")]
F = [e for e in E if not e["success"]]
c = collections.Counter(e["task_id"] for e in E)
dup = {t: n for t, n in c.items() if n > 1}
print(f"episodes={len(E)} failures={len(F)} (expl {sum(e['episode'] < 250 for e in F)}, conf {sum(e['episode'] >= 250 for e in F)}) "
      f"unique games overall={len(c)} duplicated games={len(dup)}")
games = sorted({e["task_id"] for e in F})
os.makedirs("data/horizon100_games", exist_ok=False)
json.dump(games, open("data/horizon100_games/games.json", "w"), indent=1)
json.dump([dict(task_id=e["task_id"], main_episode=e["episode"], half="explore" if e["episode"] < 250 else "confirm",
                task_type=e["task_type"], greedy_steps_30=e["steps"]) for e in F],
          open("data/horizon100_games/games_meta.json", "w"), indent=1)
print("games written:", len(games))
