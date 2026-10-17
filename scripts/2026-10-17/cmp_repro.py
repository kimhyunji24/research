"""원래(10-15) vs 같은 환경 y30 재현 vs y100: AUROC와 차이를 비교해 환경 효과와 라벨 효과를 분리한다."""
import pandas as pd, numpy as np
pairs = [("primary", "results/probe_main_3b_primary", "results/repro_y30_primary", "results/sens_y100_primary"),
         ("confirm", "results/confirm_main_3b", "results/repro_y30_confirm", "results/sens_y100_confirm")]
for name, o, r, y in pairs:
    O, R, Y = (pd.read_csv(p + ".csv") for p in (o, r, y))
    k = ["step", "method"]; m = O.merge(R, on=k, suffixes=("_orig", "_y30")).merge(Y[k + ["auroc"]].rename(columns={"auroc": "auroc_y100"}), on=k)
    cols = ["auroc", "ci_lo", "ci_hi", "p_perm", "C_median"]
    exact = {c: bool(np.allclose(m[c + "_orig"], m[c + "_y30"], atol=0, rtol=0, equal_nan=True)) for c in cols if c + "_orig" in m}
    print(f"== {name}: rows={len(m)} exact match orig vs y30 per column: {exact}")
    m["env_effect"] = m["auroc_y30"] - m["auroc_orig"]; m["label_effect"] = m["auroc_y100"] - m["auroc_y30"]
    print(m[k + ["auroc_orig", "auroc_y30", "auroc_y100", "env_effect", "label_effect"]].round(4).to_string(index=False))
    Od, Rd, Yd = (pd.read_csv(p + "_diff.csv") for p in (o, r, y))
    d = Od.merge(Rd, on=["step", "method"], suffixes=("_orig", "_y30")).merge(Yd[["step", "auroc_diff", "ci_lo", "ci_hi"]].rename(columns={"auroc_diff": "d_y100", "ci_lo": "lo_y100", "ci_hi": "hi_y100"}), on="step")
    print(d[["step", "auroc_diff_orig", "ci_lo_orig", "ci_hi_orig", "auroc_diff_y30", "ci_lo_y30", "ci_hi_y30", "d_y100", "lo_y100", "hi_y100"]].round(4).to_string(index=False))
