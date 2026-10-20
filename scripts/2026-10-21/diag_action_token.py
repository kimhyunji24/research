"""B6 diagnostic: why does the prompt-last-token hidden from the prompt+action forward differ from the stored hidden?
Compares, for 40 rows spread over the 750, A = prompt-only forward (as in rollout), B = prompt+action forward at the
prompt's last token, R = stored hidden. Prints relative L2 and cosine per depth. Writes nothing."""
import json, os, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "collect"))
from rollout import layer_indices  # noqa: E402

run = "data/main_3b/Qwen2.5-3B-Instruct"
steps = sorted([s for s in map(json.loads, open(f"{run}/steps.jsonl")) if s["episode"] < 250 and s["step"] <= 2],
               key=lambda s: s["hid_idx"])
ref = np.load(f"{run}/hidden.npy", mmap_mode="r")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype=torch.bfloat16).to("cuda").eval()
L = layer_indices(model.config.num_hidden_layers, [0.25, 0.5, 0.75, 1.0])


def rel(x, y): return np.linalg.norm(x - y, axis=-1) / np.linalg.norm(y, axis=-1)


def cos(x, y): return (x * y).sum(-1) / np.linalg.norm(x, axis=-1) / np.linalg.norm(y, axis=-1)


out = []
with torch.no_grad():
    for n in np.linspace(0, 749, 40).astype(int):
        s = steps[n]
        p = tok(s["prompt"])["input_ids"]; c = tok(f"Action: {s['action']}<|im_end|>", add_special_tokens=False)["input_ids"]
        oa = model(input_ids=torch.tensor([p], device="cuda"), output_hidden_states=True, logits_to_keep=1)
        ob = model(input_ids=torch.tensor([p + c], device="cuda"), output_hidden_states=True, logits_to_keep=1)
        A = np.stack([oa.hidden_states[l][0, -1].float().cpu().numpy() for l in L]).astype(np.float16).astype(np.float32)
        B = np.stack([ob.hidden_states[l][0, len(p) - 1].float().cpu().numpy() for l in L]).astype(np.float16).astype(np.float32)
        R = ref[s["hid_idx"]].astype(np.float32)
        out.append(np.stack([rel(A, R), rel(B, R), rel(B, A), cos(B, R)]))
o = np.array(out)  # (40, 4 quantities, 4 depths)
for qi, q in enumerate(["relL2(promptonly, stored)", "relL2(prompt+action, stored)", "relL2(prompt+action, promptonly)",
                        "cos(prompt+action, stored)"]):
    print(f"{q:36s}", " ".join(f"d{int(r*100)}: med {np.median(o[:, qi, j]):.2e} max {o[:, qi, j].max():.2e} min {o[:, qi, j].min():.4f}"
                                  for j, r in enumerate([0.25, 0.5, 0.75, 1.0])))
print("rows with promptonly exact match:", int((o[:, 0, :].max(1) == 0).sum()), "/ 40")
