"""같은 게임을 반복해서 reset할 수 있는지 확인한다(LLM 없이 환경만)."""
import json, os, sys, yaml, collections
from alfworld.agents.environment import get_environment
cfg = yaml.safe_load(os.path.expandvars(open("configs/alfworld_base.yaml").read()))
E = [json.loads(l) for l in open("data/main_3b/Qwen2.5-3B-Instruct/episodes.jsonl")]
games = [E[i]["task_id"] for i in (0, 1, 2)]
lst = [g for g in games for _ in range(3)]            # [g0,g0,g0,g1,g1,g1,g2,g2,g2]
obj = get_environment(cfg["env"]["type"])(cfg, train_eval="train")
obj.game_files = lst; obj.num_games = len(lst)
env = obj.init_env(batch_size=1)
seq = []
for i in range(9):
    obs, infos = env.reset()
    seq.append(games.index(infos["extra.gamefile"][0]) if infos["extra.gamefile"][0] in games else -1)
print("reset 순서(게임 인덱스):", seq)
print("횟수:", dict(collections.Counter(seq)))
