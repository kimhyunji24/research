"""Position exploration (docs/position_plan_2026-10-24.md): hidden states at five prompt positions, teacher-forced through
the 3B actor on the stored prompts (no new episodes). Rows: main_3b + ext1000_3b, steps 0-2 (4,500 rows).

    .venv/bin/python scripts/2026-10-24/extract_positions.py --dry --limit 12            (tokenizer only, prints spans)
    .venv/bin/python scripts/2026-10-24/extract_positions.py --out /tmp/pos_check --limit 5   (GPU smoke)
    .venv/bin/python scripts/2026-10-24/extract_positions.py --out data/positions_3b          (GPU, user present)
Writes hidden.npy (n, 5, 2, 2048) float16: positions C1..C5 x layers 18, 27; rows.csv, meta.json. Stops if hidden.npy exists.
C1 'assistant' token; C2 last token of the latest observation (tokens overlapping the span, so a '.\n' token at its end counts); C3 mean over that observation; C4 mean over the history
(from the last 'Your task is to:' to 'Admissible actions:'); C5 mean over all prompt tokens.
Checks: token count == prompt_tokens, non-empty spans, prompt-last-token hidden vs stored hidden.
"""
import argparse, datetime, json, os, subprocess, sys
import numpy as np, pandas as pd
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["data/main_3b/Qwen2.5-3B-Instruct", "data/ext1000_3b/Qwen2.5-3B-Instruct"])
ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--out")
ap.add_argument("--limit", type=int)
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
if not a.dry:
    assert a.out, "--out required"
    if os.path.exists(os.path.join(a.out, "hidden.npy")): raise SystemExit(f"output exists: {a.out}/hidden.npy")
print("start", datetime.datetime.now().isoformat(timespec="seconds"), "HEAD",
      subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(), flush=True)

TASK, ADM, OBS = "Your task is to:", "\nAdmissible actions:", "Observation: "


def spans(prompt):
    """Character spans (obs, hist) of one prompt."""
    adm = prompt.rfind(ADM)
    task = prompt.rfind(TASK, 0, adm)
    assert adm > 0 and task > 0, (adm, task)
    o = prompt.rfind(OBS, task, adm)
    if o >= 0:
        obs = (o + len(OBS), adm)
    else:  # step 0: the room description paragraph before the task line
        end = task - 2 if prompt[task - 2:task] == "\n\n" else task
        start = prompt.rfind("\n\n", 0, end) + 2
        obs = (start, end)
    return obs, (task, adm)


def tok_range(offs, lo, hi):
    # tokens overlapping [lo, hi): BPE attaches a leading space to the word and a trailing '.' to the newline
    idx = [i for i, (s, e) in enumerate(offs) if e > s and e > lo and s < hi]
    assert idx, (lo, hi)
    return idx[0], idx[-1]


tok = AutoTokenizer.from_pretrained(a.model)
items = []
for r in a.runs:
    steps = [json.loads(l) for l in open(os.path.join(r, "steps.jsonl"))]
    steps = sorted([s for s in steps if s["step"] <= 2], key=lambda s: (s["episode"], s["step"]))
    for s in steps: s["_run"] = r
    items += steps
assert len(items) == 4500, len(items)
if a.limit:  # spread over both runs and steps 0-2
    items = items[: a.limit // 2] + items[1500:1500 + a.limit - a.limit // 2] if a.limit < 1500 else items[: a.limit]

if not a.dry:
    import torch
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "collect"))
    from rollout import layer_indices  # noqa: E402
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    all_layers = layer_indices(model.config.num_hidden_layers, [0.25, 0.5, 0.75, 1.0])
    layers = all_layers[1:3]
    assert layers == [18, 27], layers
    refs = {r: np.load(os.path.join(r, "hidden.npy"), mmap_mode="r") for r in a.runs}
    H = np.zeros((len(items), 5, 2, model.config.hidden_size), np.float16)
rows, mism, max_diff, rel_diff = [], 0, 0.0, 0.0
for n, s in enumerate(items):
    enc = tok(s["prompt"], return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) != s["prompt_tokens"]: mism += 1
    obs_c, hist_c = spans(s["prompt"])
    o0, o1 = tok_range(offs, *obs_c)
    h0, h1 = tok_range(offs, *hist_c)
    L = len(ids)
    pos = dict(c1=L - 2, c2=o1, o0=o0, o1=o1, h0=h0, h1=h1)
    rows.append(dict(run=s["_run"].split("/")[1], episode=s["episode"], step=s["step"], hid_idx=s["hid_idx"], prompt_tokens=L,
                     obs_tokens=o1 - o0 + 1, hist_tokens=h1 - h0 + 1, **pos))
    if a.dry:
        if n < 12 or n % 400 == 0:
            print(f"[{n}] ep{s['episode']} step{s['step']} L={L} assistant={tok.decode([ids[L-2]])!r} obs({o1-o0+1} tok)={tok.decode(ids[o0:o1+1])[:110]!r}"
                  f" last={tok.decode([ids[o1]])!r} hist({h1-h0+1} tok)={tok.decode(ids[h0:h1+1])[:60]!r}...")
        continue
    with torch.no_grad():
        o = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True, logits_to_keep=1)
        hs = [o.hidden_states[l][0].float() for l in layers]  # (L, d) per layer
        for j, h in enumerate(hs):
            H[n, 0, j] = h[L - 2].cpu().numpy()
            H[n, 1, j] = h[o1].cpu().numpy()
            H[n, 2, j] = h[o0:o1 + 1].mean(0).cpu().numpy()
            H[n, 3, j] = h[h0:h1 + 1].mean(0).cpu().numpy()
            H[n, 4, j] = h.mean(0).cpu().numpy()
        hp = np.stack([o.hidden_states[l][0, L - 1].float().cpu().numpy() for l in all_layers])
        ref = refs[s["_run"]][s["hid_idx"]].astype(np.float32)
        d = np.abs(hp.astype(np.float16).astype(np.float32) - ref)
        max_diff = max(max_diff, float(d.max())); rel_diff = max(rel_diff, float(d.max() / np.abs(ref).max()))
    if n % 250 == 0: print(f"  {n}/{len(items)} max|diff| {max_diff:.4g} rel {rel_diff:.4g}", flush=True)
df = pd.DataFrame(rows)
print("token count mismatches:", mism, "| obs tokens min/med/max", df.obs_tokens.min(), int(df.obs_tokens.median()), df.obs_tokens.max(),
      "| hist tokens", df.hist_tokens.min(), int(df.hist_tokens.median()), df.hist_tokens.max())
if a.dry: raise SystemExit("dry run done (no GPU)")
os.makedirs(a.out, exist_ok=True)
np.save(os.path.join(a.out, "hidden.npy"), H)
df.to_csv(os.path.join(a.out, "rows.csv"), index=False)
json.dump(dict(model=a.model, revision=getattr(model.config, "_commit_hash", None), layers=layers, n=len(items),
               positions=["c1_assistant", "c2_obs_last", "c3_obs_mean", "c4_hist_mean", "c5_prompt_mean"],
               prompt_token_mismatch=mism, max_abs_diff_vs_stored=max_diff, max_rel_diff_vs_stored=rel_diff),
          open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print("saved", H.shape, "end", datetime.datetime.now().isoformat(timespec="seconds"), flush=True)
