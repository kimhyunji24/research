"""저장된 프롬프트를 다른 reader 모델에 통과시켜 마지막 토큰의 25/50/75/100% layer hidden을 뽑는다(teacher forcing).

사용: python collect/extract_reader.py --run data/main_3b/Qwen2.5-3B-Instruct --reader Qwen/Qwen2.5-0.5B-Instruct
      python collect/extract_reader.py --run ... --reader Qwen/Qwen2.5-7B-Instruct --load_4bit
출력: <run 상위>/readers/<reader 이름>/hidden.npy  (N, 4, H) float16, 행 순서는 run의 hid_idx와 같다.
검증: 각 step의 토큰 수가 steps.jsonl의 prompt_tokens와 같은지, reader가 actor와 같으면 저장된 hidden과 일치하는지.
"""
import argparse, json, os, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rollout import layer_indices  # alfworld는 main() 안에서만 import한다

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--reader", required=True)
ap.add_argument("--load_4bit", action="store_true", help="nf4 4-bit 양자화 로드(7B 이상, 12GB VRAM용)")
ap.add_argument("--relative_layers", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
ap.add_argument("--limit", type=int, help="앞쪽 N step만(점검용)")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

steps = [json.loads(l) for l in open(os.path.join(a.run, "steps.jsonl"))]
assert all("prompt" in s for s in steps), "steps.jsonl에 prompt가 없다(prompt 저장 이후 수집분만 가능)"
order = sorted(range(len(steps)), key=lambda i: steps[i]["hid_idx"])
if a.limit: order = order[:a.limit]

tok = AutoTokenizer.from_pretrained(a.reader)
kw = dict(dtype=torch.bfloat16)
if a.load_4bit:
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.bfloat16)
    kw["device_map"] = {"": 0}
model = AutoModelForCausalLM.from_pretrained(a.reader, **kw)
if not a.load_4bit: model = model.to(a.device)
model.eval()
layers = layer_indices(model.config.num_hidden_layers, a.relative_layers)
print(f"reader={a.reader} 4bit={a.load_4bit} layers={layers} hidden={model.config.hidden_size} steps={len(order)}", flush=True)

out_dir = os.path.join(os.path.dirname(a.run.rstrip("/\\")), "readers", a.reader.split("/")[-1] + ("-4bit" if a.load_4bit else ""))
os.makedirs(out_dir, exist_ok=True)
H = np.zeros((len(order), len(layers), model.config.hidden_size), dtype=np.float16)
mismatch, t0 = 0, time.time()
with torch.no_grad():
    for n, i in enumerate(order):
        s = steps[i]
        ids = tok(s["prompt"], return_tensors="pt").to(model.device)
        if ids["input_ids"].shape[1] != s["prompt_tokens"]: mismatch += 1
        o = model(**ids, output_hidden_states=True, logits_to_keep=1)
        H[n] = np.stack([o.hidden_states[l][0, -1].float().cpu().numpy() for l in layers])
        if n % 500 == 0: print(f"  {n}/{len(order)} {time.time() - t0:.0f}s", flush=True)
np.save(os.path.join(out_dir, "hidden.npy"), H)
print(f"saved {out_dir} shape={H.shape} token_mismatch={mismatch}")

# actor와 같은 모델이면 저장된 hidden과 비교한다(층 인덱스가 같을 때만)
ref_path = os.path.join(a.run, "hidden.npy")
if os.path.exists(ref_path) and H.shape[1:] == np.load(ref_path, mmap_mode="r").shape[1:]:
    ref = np.load(ref_path, mmap_mode="r")[:len(order)]
    print("max abs diff vs saved hidden:", float(np.abs(H.astype("float32") - ref.astype("float32")).max()))
json.dump(dict(reader=a.reader, load_4bit=a.load_4bit, layers=layers, n=len(order), token_mismatch=mismatch,
               rollout_run=a.run), open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
