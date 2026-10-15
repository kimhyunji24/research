"""탐색적 확장 결과 요약. 완료된 reader의 모든 조합을 빠짐없이 출력한다(선별 없음)."""
import os, sys
import pandas as pd

readers = [r for r in ["0.5B", "1.5B", "3B", "7B-4bit"] if os.path.exists(f"results/explore_{r}_diff.csv")]
rows = []
for r in readers:
    a = pd.read_csv(f"results/explore_{r}.csv")
    d = pd.read_csv(f"results/explore_{r}_diff.csv").rename(columns={"ci_lo": "d_lo", "ci_hi": "d_hi"})
    a = a[a.method.str.startswith(("linear", "mlp"))].merge(d[["step", "method", "auroc_diff", "d_lo", "d_hi"]], on=["step", "method"])
    a["reader"] = r; rows.append(a)
t = pd.concat(rows)
t["kind"] = t.method.str.split("_").str[0]
t["layer"] = t.method.str.extract(r"_L(\d)_")[0].astype(int)
t["pct"] = t.layer.map({0: 25, 1: 50, 2: 75, 3: 100})
b5 = pd.concat([pd.read_csv(f"results/explore_{r}.csv").query('method=="baseline5"') for r in readers])
print("readers:", readers)
print("baseline5 AUROC (step별 reader 간 min/max):", b5.groupby("step").auroc.agg(["min", "max"]).round(3).to_dict("index"))
for kind in ["linear", "mlp"]:
    print(f"\n== {kind}: AUROC [baseline5 대비 차이 (95% CI)] p_perm")
    for step in [0, 1, 2]:
        print(f" step {step}")
        for r in readers:
            x = t[(t.kind == kind) & (t.step == step) & (t.reader == r)].sort_values("layer")
            print("  ", r.ljust(8), " | ".join(
                f"{int(p.pct)}%: {p.auroc:.3f} [{p.auroc_diff:+.2f} ({p.d_lo:+.2f},{p.d_hi:+.2f})] p={p.p_perm:.2f}"
                for p in x.itertuples()))
print(f"\n조합 수 {len(t)}, 차이 CI 상한이 0 이상인 조합 {int((t.d_hi >= 0).sum())}, 점추정이 양수인 조합 {int((t.auroc_diff > 0).sum())}")
i = t.auroc_diff.idxmax() if False else t.auroc_diff.reset_index(drop=True).idxmax()
m = t.reset_index(drop=True).loc[i]
print(f"차이 최댓값 {m.auroc_diff:+.3f} ({m.reader}, {m.method}, step {m.step}, CI {m.d_lo:+.2f}~{m.d_hi:+.2f})")
lin = t[t.kind == "linear"]
print("linear에서 C가 격자 끝(1.0 또는 1e-4)인 조합:", int(((lin.C_median >= 1.0) | (lin.C_median <= 1e-4)).sum()), "/", len(lin))
print("\nreader x layer 평균 AUROC (linear, step 0~2 평균)")
print(lin.groupby(["reader", "pct"]).auroc.mean().unstack().round(3).to_string())
mlp = t[t.kind == "mlp"]
print("\nreader x layer 평균 AUROC (mlp, step 0~2 평균)")
print(mlp.groupby(["reader", "pct"]).auroc.mean().unstack().round(3).to_string())
print("\nMLP - linear 평균 격차 (reader별, 같은 layer/step)")
mm = mlp.set_index(["reader", "pct", "step"]).auroc - lin.set_index(["reader", "pct", "step"]).auroc
print(mm.groupby("reader").mean().round(3).to_string())
