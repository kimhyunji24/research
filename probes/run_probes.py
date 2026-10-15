"""사용: python probes/run_probes.py data/decision/Qwen2.5-3B-Instruct --layer 1 --rounds 1 2 3
출력: baseline 1~3단, 프로브(linear/MLP) x PCA k 스윕, 무작위 라벨 대조군의 라운드별 AUROC."""
import argparse
import pandas as pd
from lib import baseline_features, load_run, make_model, round_eval

ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("--layer", type=int, default=1, help="relative_layers 인덱스 (1=50%)")
ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--pca", type=int, nargs="*", default=[0, 8, 32, 64], help="0=PCA 없음(전체 차원)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

df, hid = load_run(a.path)
H = hid[df["hid_idx"].to_numpy(), a.layer].astype("float32")
rows = []
for r in a.rounds:
    for rung in (1, 2, 3):
        Xb = baseline_features(df, rung)
        res = round_eval(df, Xb, r, lambda: make_model("linear", C=1.0))
        if res: rows.append(dict(res, method=f"baseline{rung}"))
    for kind in ("linear", "mlp"):
        for k in a.pca:
            for shuf in (False, True):
                res = round_eval(df, H, r, lambda: make_model(kind, pca_k=k or None), shuffle_labels=shuf)
                if res: rows.append(dict(res, method=f"{kind}_pca{k or 'full'}" + ("_shuffled" if shuf else "")))
out = pd.DataFrame(rows)
print(out.round(3).to_string(index=False))
if a.out: out.to_csv(a.out, index=False)
