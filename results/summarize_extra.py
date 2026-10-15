import pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
A = pd.read_csv("results/extra_main_3b_A_reader_pairs.csv")
print("== A. reader 간 짝지은 AUROC 차이 (linear, 장면 cluster bootstrap 95% CI)")
for lp in (50, 75):
    print(f"-- layer {lp}%")
    x = A[(A.layer_pct == lp) & (A.b != "baseline5")]
    x = x.assign(pair=x.a + " - " + x.b, txt=x.apply(lambda r: f"{r['diff']:+.3f} ({r['lo']:+.3f},{r['hi']:+.3f})", axis=1))
    print(x.pivot(index="pair", columns="step", values="txt").to_string())
    y = A[(A.layer_pct == lp) & (A.b == "baseline5")]
    print("  reader AUROC:", {f"{r.a}@{r.step}": round(r.auroc_a, 3) for r in y.itertuples()})
B = pd.read_csv("results/extra_main_3b_B_late_rounds.csv")
print("\n== B. 후반 라운드 (3B reader, 75% layer, linear)")
p = B[B.method == "probe_3B_L75_linear"].set_index("step"); b5 = B[B.method == "baseline5"].set_index("step"); b4 = B[B.method == "baseline4"].set_index("step")
d = B[B.method == "probe_minus_baseline5"].set_index("step")
t = pd.DataFrame({"n": p.n, "n_fail": p.n_fail, "probe": p.auroc, "probe_lo": p.ci_lo, "probe_hi": p.ci_hi, "b4": b4.auroc, "b5": b5.auroc,
                  "diff_vs_b5": d.auroc, "d_lo": d.ci_lo, "d_hi": d.ci_hi, "probe_auprc": p.auprc, "b5_auprc": b5.auprc,
                  "probe_tpr@10fpr": p.tpr_at_10fpr, "b5_tpr@10fpr": b5.tpr_at_10fpr, "rem_steps_fail": p.mean_remaining_steps_of_failures})
print(t.round(3).to_string())
C = pd.read_csv("results/extra_main_3b_C_temporal.csv")
print("\n== C. 시간축 프로브 (3B reader, 75% layer, linear) AUROC [baseline5 대비 차이 (CI)]")
C["txt"] = C.apply(lambda r: f"{r.auroc:.3f} [{r['diff']:+.2f} ({r['lo']:+.2f},{r['hi']:+.2f})]", axis=1)
print(C.pivot(index="variant", columns="step", values="txt").to_string())
print("baseline5:", C.groupby("step").auroc_baseline5.first().round(3).to_dict())
