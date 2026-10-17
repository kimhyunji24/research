"""null 보정 집계: 무작위 라벨 seed별 결과에서 p_perm 분포, CI가 0.5(또는 차이 0)를 포함하는 비율, 판정 문구 빈도."""
import glob, numpy as np, pandas as pd
from scipy import stats
R = pd.concat([pd.read_csv(f).assign(seed=f.split("/")[-1][1:-4]) for f in glob.glob("/tmp/null_calib/s*[0-9].csv")])
D = pd.concat([pd.read_csv(f).assign(seed=f.split("/")[-1].split("_")[0][1:]) for f in glob.glob("/tmp/null_calib/s*_diff.csv")])
print(f"seeds={R.seed.nunique()}  (기대: p_perm<=0.05 비율 0.05, CI가 0.5/0 포함 비율 0.95)\n")
R["cover"] = (R.ci_lo <= 0.5) & (R.ci_hi >= 0.5)
g = R.groupby(["pairs", "method"]).agg(n=("pair_auroc", "size"), mean_auc=("pair_auroc", "mean"), sd_auc=("pair_auroc", "std"),
                                      p05=("p_perm", lambda p: (p <= 0.05).mean()), p95=("p_perm", lambda p: (p >= 0.95).mean()),
                                      ci_cover=("cover", "mean"))
print("방법별 (step 1~3 합침)\n" + g.round(3).to_string() + "\n")
for pset in ["all", "nontied"]:
    p = R[R.pairs == pset].p_perm.to_numpy()
    print(f"[{pset}] p_perm 전체 n={len(p)}: <=0.05 {np.mean(p <= 0.05):.3f}, <=0.01 {np.mean(p <= 0.01):.3f}, "
          f">=0.95 {np.mean(p >= 0.95):.3f}, KS vs U(0,1) p={stats.kstest(p, 'uniform').pvalue:.3f}")
D["cover0"] = (D.ci_lo <= 0) & (D.ci_hi >= 0)
print("\n차이(프로브 - baseline) family별\n" + D.groupby("family").agg(
    n=("diff", "size"), mean_diff=("diff", "mean"), sd_diff=("diff", "std"), ci_cover0=("cover0", "mean"),
    holm_sig=("p_holm", lambda p: (p < 0.05).mean())).round(3).to_string())
print("\n판정 문구 빈도 (null에서)\n" + D.groupby("family").verdict.value_counts(normalize=True).unstack(fill_value=0).round(3).to_string())
# bootstrap CI 폭 vs seed 간 실제 변동: CI 반폭 평균과 seed 간 sd x 1.96 비교
D["half"] = (D.ci_hi - D.ci_lo) / 2
cmp = D.groupby(["family", "step", "b"]).agg(boot_half=("half", "mean"), seed_sd=("diff", "std"))
cmp["seed_1.96sd"] = 1.96 * cmp.seed_sd
print("\nCI 반폭(bootstrap 평균) vs 1.96 x seed 간 sd\n" + cmp.round(3).to_string())
