"""B6 (docs/review_b_plan_2026-10-21.md): hidden state at the last token of the chosen action, teacher-forced through the
3B actor. Input ids = tok(prompt) + tok("Action: <action><|im_end|>", add_special_tokens=False), the same concatenation
rollout.score_candidates scored; one forward pass over the whole sequence. Rows: first 250 episodes, steps 0-2 of main_3b.

    .venv/bin/python scripts/2026-10-21/extract_action_token.py --out data/main_3b/action_token/Qwen2.5-3B-Instruct
    (check: --limit 5 --out /tmp/at_check)
Writes hidden.npy (n, 2, 2048) float16 for layers 18 and 27 (50%/75% depth), rows.csv, meta.json; stops if hidden.npy exists.
Checks: prompt token count == prompt_tokens; hidden at the prompt's last token vs the stored hidden; the action's last
token is followed by <|im_end|> and the action span is not empty.
"""
import argparse, datetime, json, os, subprocess, sys
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "collect"))
from rollout import layer_indices  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="data/main_3b/Qwen2.5-3B-Instruct")
ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int)
a = ap.parse_args()
if os.path.exists(os.path.join(a.out, "hidden.npy")): raise SystemExit(f"output exists: {a.out}/hidden.npy")
print("start", datetime.datetime.now().isoformat(timespec="seconds"), "HEAD",
      subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(), flush=True)
steps = [json.loads(l) for l in open(os.path.join(a.run, "steps.jsonl"))]
steps = sorted([s for s in steps if s["episode"] < 250 and s["step"] <= 2], key=lambda s: s["hid_idx"])
assert len(steps) == 750, len(steps)
if a.limit: steps = steps[:a.limit]
ref = np.load(os.path.join(a.run, "hidden.npy"), mmap_mode="r")
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
all_layers = layer_indices(model.config.num_hidden_layers, [0.25, 0.5, 0.75, 1.0])
layers = all_layers[1:3]
assert layers == [18, 27], layers
end_id = tok.convert_tokens_to_ids("<|im_end|>")
H = np.zeros((len(steps), 2, model.config.hidden_size), np.float16)
rows, max_diff, rel_diff, mism = [], 0.0, 0.0, 0
with torch.no_grad():
    for n, s in enumerate(steps):
        assert s["action"] in s["cands"]
        p = tok(s["prompt"])["input_ids"]
        c = tok(f"Action: {s['action']}<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p) != s["prompt_tokens"]: mism += 1
        assert c[-1] == end_id and len(c) >= 3, c
        ids = torch.tensor([p + c], device="cuda")
        o = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1)
        pos = len(p) + len(c) - 2  # last action token (just before <|im_end|>)
        H[n] = np.stack([o.hidden_states[l][0, pos].float().cpu().numpy() for l in layers])
        # prompt-last-token hidden vs stored hidden (all 4 depths)
        hp = np.stack([o.hidden_states[l][0, len(p) - 1].float().cpu().numpy() for l in all_layers])
        r = ref[s["hid_idx"]].astype(np.float32)
        d = np.abs(hp.astype(np.float16).astype(np.float32) - r)
        max_diff = max(max_diff, float(d.max())); rel_diff = max(rel_diff, float(d.max() / np.abs(r).max()))
        rows.append(dict(episode=s["episode"], step=s["step"], hid_idx=s["hid_idx"], prompt_tokens=len(p), action_tokens=len(c),
                         last_action_token=tok.decode([c[-2]])))
        if n % 100 == 0: print(f"  {n}/{len(steps)} max|diff| {max_diff:.4g} rel {rel_diff:.4g}", flush=True)
os.makedirs(a.out, exist_ok=True)
np.save(os.path.join(a.out, "hidden.npy"), H)
pd.DataFrame(rows).to_csv(os.path.join(a.out, "rows.csv"), index=False)
meta = dict(model=a.model, revision=getattr(model.config, "_commit_hash", None), layers=layers, n=len(steps),
            prompt_token_mismatch=mism, max_abs_diff_vs_stored=max_diff, max_rel_diff_vs_stored=rel_diff)
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print("saved", H.shape, meta)
print("end", datetime.datetime.now().isoformat(timespec="seconds"), flush=True)
