"""ALFWorld ReAct rollout + hidden state 저장.

각 step에서 action 생성 직전 마지막 입력 토큰의 hidden state(25/50/75/final layer)를 저장한다.
사용: python collect/rollout.py --config configs/pilot.yaml [--model Qwen/Qwen2.5-1.5B-Instruct]
"""
import argparse, copy, difflib, json, os, re, time
import numpy as np
import torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = """You are an agent in a household text environment. Each turn you see the task, what happened so far, and the list of admissible actions.

Rules:
- Choose exactly ONE action from the admissible actions list and copy it exactly.
- A closed receptacle (cabinet, drawer, fridge...) must be opened before you can see or take what is inside.
- Only take the objects the task asks for. Ignore other objects.
- After taking the object, go to the target receptacle and move the object there. Do not move it back to where you found it.
- If an action got "Nothing happens", it did not work. Do not repeat it; pick a different one.
- Reply with one line only: Action: <action>

Example of a good episode:
Task: clean some mug and put it in countertop.
Observation: You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a countertop 1, a sinkbasin 1.
Action: go to cabinet 1
Observation: You arrive at cabinet 1. The cabinet 1 is closed.
Action: open cabinet 1
Observation: You open the cabinet 1. The cabinet 1 is open. In it, you see a mug 1.
Action: take mug 1 from cabinet 1
Observation: You pick up the mug 1 from the cabinet 1.
Action: go to sinkbasin 1
Observation: You arrive at sinkbasin 1. On the sinkbasin 1, you see nothing.
Action: clean mug 1 with sinkbasin 1
Observation: You clean the mug 1 using the sinkbasin 1.
Action: go to countertop 1
Observation: You arrive at countertop 1. On the countertop 1, you see nothing.
Action: move mug 1 to countertop 1"""


def peak_mem_gb(device):
    if device.startswith("cuda"): return torch.cuda.max_memory_allocated() / 1e9
    if device == "mps": return torch.mps.current_allocated_memory() / 1e9
    return 0.0


def layer_indices(n_layers, rel):
    # hidden_states 길이는 n_layers+1 (embedding 포함). 1.0 -> final layer.
    return [max(1, min(n_layers, round(r * n_layers))) for r in rel]


def parse_action(text, admissible):
    """(실행할 action, 정확히 일치 여부). 정확히 일치하지 않으면 가장 가까운 admissible로 맞춘다(cutoff 0.85)."""
    m = re.search(r"Action:\s*(.+)", text)
    cand = (m.group(1) if m else text).strip().splitlines()[0].strip().lower().rstrip(".")
    adm = {a.lower(): a for a in admissible}
    if cand in adm: return adm[cand], True
    close = difflib.get_close_matches(cand, list(adm), n=1, cutoff=0.85)
    return (adm[close[0]] if close else cand), False


def make_prompt(tok, task, trace, obs, admissible):
    # trace: (obs, action, ok) ; ok=False면 환경이 "Nothing happens"로 응답한 action
    admissible = [a for a in admissible if a != "help"]
    lines = [f"Task: {task}"]
    for o, a, _ in trace[-6:]:  # 최근 6 step만 유지해 길이 통제
        lines.append(f"Observation: {o}\nAction: {a}")
    lines.append(f"Observation: {obs}")
    failed = list(dict.fromkeys(a for _, a, ok in trace[-6:] if not ok))
    if failed: lines.append(f"Actions that did nothing recently (do not repeat): {', '.join(failed)}")
    lines.append(f"Admissible actions: {', '.join(admissible)}")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


SKIP_ACTIONS = {"help", "inventory", "look"}  # 후보에서 제외(정보 확인용 명령은 과제 진행에 쓰이지 않음)


@torch.no_grad()
def score_candidates(model, tok, prompt, candidates, layers, device, norm):
    """prompt를 1회 prefill해 hidden state를 뽑고, KV cache를 복제해 각 후보 'Action: <a><|im_end|>'의
    로그확률을 한 번에 계산한다. 반환: (hidden (L,H), 후보별 점수, 후보별 sum logprob,
    후보별 토큰 수, prompt 토큰 수)"""
    ids = tok(prompt, return_tensors="pt").to(device)
    pre = model(**ids, use_cache=True, output_hidden_states=True, logits_to_keep=1)
    vec = np.stack([pre.hidden_states[l][0, -1].float().cpu().numpy() for l in layers])
    first_lp = torch.log_softmax(pre.logits[0, -1].float(), dim=-1)          # 첫 후보 token의 분포

    enc = [tok(f"Action: {c}<|im_end|>", add_special_tokens=False)["input_ids"] for c in candidates]
    B, L = len(enc), max(len(e) for e in enc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    cid = torch.full((B, L), pad, dtype=torch.long, device=device)
    cmask = torch.zeros((B, L), dtype=torch.long, device=device)
    for i, e in enumerate(enc):
        cid[i, :len(e)] = torch.tensor(e, device=device); cmask[i, :len(e)] = 1
    cache = copy.deepcopy(pre.past_key_values); cache.batch_repeat_interleave(B)
    P = ids["input_ids"].shape[1]
    attn = torch.cat([torch.ones((B, P), dtype=torch.long, device=device), cmask], dim=1)
    lg = model(input_ids=cid, attention_mask=attn, past_key_values=cache, use_cache=True).logits.float()
    lp = torch.log_softmax(lg, dim=-1)                                       # (B, L, V)
    tok_lp = torch.zeros((B, L), device=device)
    tok_lp[:, 0] = first_lp[cid[:, 0]]
    if L > 1:
        tok_lp[:, 1:] = lp[:, :-1].gather(-1, cid[:, 1:, None]).squeeze(-1)
    tok_lp = tok_lp * cmask
    total = tok_lp.sum(1)
    ntok = cmask.sum(1)
    score = total / ntok if norm == "mean" else total
    return vec, score.cpu().numpy(), total.cpu().numpy(), ntok.cpu().numpy(), P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--model")
    ap.add_argument("--num_episodes", type=int)
    ap.add_argument("--out_dir", help="config의 out_dir을 덮어씀")
    ap.add_argument("--resume", action="store_true", help="out 폴더의 완료된 episode 뒤에서 이어서 수집")
    ap.add_argument("--overwrite", action="store_true", help="out 폴더에 결과가 있어도 지우고 새로 시작")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.model: cfg["model_name"] = a.model
    if a.num_episodes: cfg["num_episodes"] = a.num_episodes
    if a.out_dir: cfg["out_dir"] = a.out_dir
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
    ep_path, st_path = os.path.join(out, "episodes.jsonl"), os.path.join(out, "steps.jsonl")
    shard_dir = os.path.join(out, "hidden_shards"); os.makedirs(shard_dir, exist_ok=True)
    hid, t0, wins, done, eps_done = [], time.time(), 0, 0, []
    n_tok = 0

    if os.path.exists(ep_path) and os.path.getsize(ep_path) > 0:
        if a.overwrite:  # 명시적으로 요청했을 때만 기존 결과를 지운다
            for f in os.listdir(shard_dir): os.remove(os.path.join(shard_dir, f))
            open(ep_path, "w").close(); open(st_path, "w").close()
        elif a.resume:
            eps_done = [json.loads(l) for l in open(ep_path) if l.strip()]
            done = len(eps_done)
            for e in range(done):  # episode 단위 shard가 있어야 완료로 인정한다
                hid.extend(np.load(os.path.join(shard_dir, f"ep{e:05d}.npy")).astype(np.float32))
            rows = [r for r in map(json.loads, open(st_path)) if r["episode"] < done]
            assert not rows or max(r["hid_idx"] for r in rows) == len(hid) - 1, "steps/hidden shard 정합성 오류"
            with open(st_path, "w") as f:  # 완료되지 않은 episode의 step은 버린다
                for r in rows: f.write(json.dumps(r) + "\n")
            wins = sum(e["success"] for e in eps_done)
            print(f"[resume] {done} episode 완료, {len(hid)} step 복원", flush=True)
        else:
            raise SystemExit(f"{out}에 기존 결과가 있다. --resume 또는 --overwrite 중 하나를 지정하라.")
    meta_f = open(st_path, "a"); open(ep_path, "a").close()

    # 완료된 episode만큼 환경을 되감는다. 게임 순서가 달라졌으면 이어붙이면 안 되므로 중단한다.
    for e in range(done):
        _, infos0 = env.reset()
        if infos0["extra.gamefile"][0] != eps_done[e]["task_id"]:
            raise SystemExit(f"resume 불가: episode {e}의 게임 순서가 이전 실행과 다르다(환경 순서가 결정적이지 않음).")

    for ep in range(done, cfg["num_episodes"]):
        hid_start = len(hid)
        obs, infos = env.reset()
        text = obs[0]
        task = text.split("Your task is to:")[-1].strip()
        gamefile = infos["extra.gamefile"][0]
        trace, won = [], False
        for step in range(cfg["max_steps"]):
            adm = infos["admissible_commands"][0]
            prompt = make_prompt(tok, task, trace, text, adm)
            extra = {}
            if cfg.get("policy", "score") == "score":
                cands = [c for c in adm if c not in SKIP_ACTIONS] or list(adm)
                vec, score, total, ntok, n_prompt = score_candidates(model, tok, prompt, cands, layers,
                                                                     cfg["device"], cfg.get("score_norm", "sum"))
                best = int(score.argmax())
                p = np.exp(score - score.max()); p /= p.sum()
                action, valid, raw = cands[best], True, ""
                # prompt와 후보별 점수를 저장해 다른 크기 모델로 teacher-forced 재추출과 score_norm 분석을 가능하게 한다.
                # hidden은 prompt 마지막 토큰(위치 prompt_tokens-1)에서 뽑은 것이다.
                extra = dict(n_cand=len(cands), p_chosen=round(float(p[best]), 4),
                             prompt=prompt, prompt_tokens=int(n_prompt), cands=cands,
                             cand_logp=[round(float(x), 4) for x in total], cand_ntok=[int(x) for x in ntok])
                hid.append(vec)
            else:  # policy: generate (자유 생성 후 파싱)
                ids = tok(prompt, return_tensors="pt").to(cfg["device"])
                with torch.no_grad():
                    g = model.generate(**ids, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                                       output_hidden_states=True, return_dict_in_generate=True)
                h0 = g.hidden_states[0]  # prompt forward pass; 마지막 입력 토큰 위치
                hid.append(np.stack([h0[l][0, -1].float().cpu().numpy() for l in layers]))  # (L, hidden)
                gen = g.sequences[0, ids["input_ids"].shape[1]:]
                n_tok += len(gen)
                raw = tok.decode(gen, skip_special_tokens=True)
                action, valid = parse_action(raw, adm)   # valid=정확 일치, action=실행한 문자열(근접 매칭 포함)
                extra = dict(prompt=prompt, prompt_tokens=int(ids["input_ids"].shape[1]))
            ok = action in adm
            meta_f.write(json.dumps(dict(episode=ep, task_id=gamefile, step=step, action=action,
                                         valid=valid, ok=ok, raw=raw, hid_idx=len(hid) - 1, model=tag,
                                         **extra)) + "\n")
            meta_f.flush()
            trace.append((text, action, ok))
            obs, _, dones, infos = env.step([action])  # 무효 action도 그대로 보내 환경이 "Nothing happens"로 응답하게 함
            text = obs[0]
            if infos["won"][0]: won = True
            if dones[0]: break
        wins += won
        # episode가 끝나면 hidden shard를 먼저 저장하고 episodes.jsonl에 기록한다(기록이 '완료' 표시).
        np.save(os.path.join(shard_dir, f"ep{ep:05d}.npy"), np.stack(hid[hid_start:]).astype(np.float16))
        # 에피소드 최종 성공 라벨을 해당 episode의 모든 step에 부여하는 작업은 후처리(probes)에서 수행
        ttype = gamefile.split("/")[-3].split("-")[0]  # 예: pick_two_obj_and_place
        with open(os.path.join(out, "episodes.jsonl"), "a") as f:
            f.write(json.dumps(dict(episode=ep, task_id=gamefile, task_type=ttype, success=bool(won),
                                    steps=len(trace))) + "\n")
        print(f"[{tag}] ep {ep+1}/{cfg['num_episodes']} won={won} steps={len(trace)} total_wins={wins}", flush=True)

    np.save(os.path.join(out, "hidden.npy"), np.stack(hid).astype(np.float16))
    el = time.time() - t0
    print(json.dumps(dict(model=tag, episodes=cfg["num_episodes"], success_rate=wins / cfg["num_episodes"],
                          sec=el, tokens_per_sec=n_tok / el, mem_gb=peak_mem_gb(cfg["device"]),
                          layers=layers)))


if __name__ == "__main__":
    main()
