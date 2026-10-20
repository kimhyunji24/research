"""B7 (docs/review_b_plan_2026-10-21.md): neural text embeddings of the rung-5 text (user message) for the first 250
episodes, steps 0-2 of main_3b. Encoder BAAI/bge-base-en-v1.5 through the frozen .venv's transformers (no new packages).
Each text is split into 510-token chunks; the L2-normalised CLS embeddings of the chunks are averaged and re-normalised.

    .venv/bin/python scripts/2026-10-21/embed_text.py --out data/main_3b/text_embed/bge-base-en-v1.5
Writes emb.npy (n, 768) float32 and rows.csv (episode, step, hid_idx, n_chunks); stops if the folder already has emb.npy.
"""
import argparse, datetime, json, os, subprocess, sys
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "probes"))
from lib import load_run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--path", default="data/main_3b/Qwen2.5-3B-Instruct")
ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
ap.add_argument("--out", required=True)
ap.add_argument("--chunk", type=int, default=510)
a = ap.parse_args()
if os.path.exists(os.path.join(a.out, "emb.npy")): raise SystemExit(f"output exists: {a.out}/emb.npy")
print("start", datetime.datetime.now().isoformat(timespec="seconds"),
      "HEAD", subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(), flush=True)
df, _ = load_run(a.path)
df = df[(df["episode"] < 250) & (df["step"] <= 2)].reset_index(drop=True)
assert len(df) == 750, len(df)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModel.from_pretrained(a.model).to(dev).eval()
rev = getattr(model.config, "_commit_hash", None)
E, nch = np.zeros((len(df), model.config.hidden_size), np.float32), []
with torch.no_grad():
    for i, t in enumerate(df["text"]):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        chunks = [ids[k:k + a.chunk] for k in range(0, len(ids), a.chunk)] or [[]]
        vs = []
        for c in chunks:
            x = torch.tensor([[tok.cls_token_id] + c + [tok.sep_token_id]], device=dev)
            v = model(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state[0, 0]
            vs.append(torch.nn.functional.normalize(v, dim=0))
        v = torch.nn.functional.normalize(torch.stack(vs).mean(0), dim=0)
        E[i] = v.float().cpu().numpy(); nch.append(len(chunks))
os.makedirs(a.out, exist_ok=True)
np.save(os.path.join(a.out, "emb.npy"), E)
df.assign(n_chunks=nch)[["episode", "step", "hid_idx", "n_chunks"]].to_csv(os.path.join(a.out, "rows.csv"), index=False)
json.dump(dict(model=a.model, revision=rev, device=dev, chunk=a.chunk, n=len(df), max_chunks=int(max(nch))),
          open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print("saved", E.shape, "revision", rev, "chunks min/max", min(nch), max(nch), "device", dev)
print("end", datetime.datetime.now().isoformat(timespec="seconds"), flush=True)
