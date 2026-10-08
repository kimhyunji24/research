"""ALFWorld ReAct rollout + hidden state 저장.

각 step에서 action 생성 직전 마지막 입력 토큰의 hidden state(25/50/75/final layer)를 저장한다.
사용: python collect/rollout.py --config configs/pilot.yaml [--model Qwen/Qwen2.5-1.5B-Instruct]
"""
import argparse, json, os, re, time
import numpy as np
import torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = (
    "You are an agent in a household text environment. Each turn you see an observation and the "
    "list of admissible actions. Reply in the form:\nThought: <short reasoning>\nAction: <one admissible action>"
)


def layer_indices(n_layers, rel):
    # hidden_states 길이는 n_layers+1 (embedding 포함). 1.0 -> final layer.
    return [max(1, min(n_layers, round(r * n_layers))) for r in rel]


def parse_action(text, admissible):
    m = re.search(r"Action:\s*(.+)", text)
    cand = (m.group(1) if m else text).strip().splitlines()[0].strip().lower()
    return cand, cand in {a.lower() for a in admissible}


def make_prompt(tok, task, trace, obs, admissible):
    lines = [f"Task: {task}"]
    for o, a in trace[-6:]:  # 최근 6 step만 유지해 길이 통제
        lines.append(f"Observation: {o}\nAction: {a}")
    lines.append(f"Observation: {obs}\nAdmissible actions: {', '.join(admissible)}")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--model")
    ap.add_argument("--num_episodes", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.model: cfg["model_name"] = a.model
    if a.num_episodes: cfg["num_episodes"] = a.num_episodes
    torch.manual_seed(cfg["seed"])

    from alfworld.agents.environment import get_environment
    env_cfg = yaml.safe_load(os.path.expandvars(open(cfg["alfworld_config"]).read()))
    env = get_environment(env_cfg["env"]["type"])(env_cfg, train_eval=cfg["split"]).init_env(batch_size=1)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=getattr(torch, cfg["dtype"])).to(cfg["device"]).eval()
    layers = layer_indices(model.config.num_hidden_layers, cfg["relative_layers"])

    tag = cfg["model_name"].split("/")[-1]
    out = os.path.join(cfg["out_dir"], tag); os.makedirs(out, exist_ok=True)
    meta_f = open(os.path.join(out, "steps.jsonl"), "w")
    hid, t0, wins = [], time.time(), 0
    n_tok = 0

    for ep in range(cfg["num_episodes"]):
        obs, infos = env.reset()
        text = obs[0]
        task = text.split("Your task is to:")[-1].strip()
        gamefile = infos["extra.gamefile"][0]
        trace, won = [], False
        for step in range(cfg["max_steps"]):
            adm = infos["admissible_commands"][0]
            prompt = make_prompt(tok, task, trace, text, adm)
            ids = tok(prompt, return_tensors="pt").to(cfg["device"])
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                                   output_hidden_states=True, return_dict_in_generate=True)
            # hidden_states[0] = prompt forward pass; 마지막 입력 토큰 위치
            h0 = g.hidden_states[0]
            vec = np.stack([h0[l][0, -1].float().cpu().numpy() for l in layers])  # (L, hidden)
            hid.append(vec)
            gen = g.sequences[0, ids["input_ids"].shape[1]:]
            n_tok += len(gen)
            action, valid = parse_action(tok.decode(gen, skip_special_tokens=True), adm)
            meta_f.write(json.dumps(dict(episode=ep, task_id=gamefile, step=step, action=action,
                                         valid=valid, hid_idx=len(hid) - 1, model=tag)) + "\n")
            meta_f.flush()
            trace.append((text, action))
            obs, _, dones, infos = env.step([action if valid else "look"])
            text = obs[0]
            if infos["won"][0]: won = True
            if dones[0]: break
        wins += won
        # 에피소드 최종 성공 라벨을 해당 episode의 모든 step에 부여하는 작업은 후처리(probes)에서 수행
        with open(os.path.join(out, "episodes.jsonl"), "a") as f:
            f.write(json.dumps(dict(episode=ep, task_id=gamefile, success=bool(won), steps=len(trace))) + "\n")
        print(f"[{tag}] ep {ep+1}/{cfg['num_episodes']} won={won} steps={len(trace)} total_wins={wins}", flush=True)

    np.save(os.path.join(out, "hidden.npy"), np.stack(hid).astype(np.float16))
    el = time.time() - t0
    print(json.dumps(dict(model=tag, episodes=cfg["num_episodes"], success_rate=wins / cfg["num_episodes"],
                          sec=el, tokens_per_sec=n_tok / el, mps_mem_gb=torch.mps.current_allocated_memory() / 1e9,
                          layers=layers)))


if __name__ == "__main__":
    main()
