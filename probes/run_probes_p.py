"""run_probes.py를 그대로 실행하고, 프로브 - baseline 차이마다 bootstrap 양측 p와 Holm 보정 p를 덧붙인다
(docs/model_main_plan_2026-10-17.md: 과제 단위 3개 step에 Holm을 적용한 우수 판정 병기). 점추정·CI는 run_probes.py와 같다.
사용: python probes/run_probes_p.py <run_probes.py 인자들, --out 필수>"""
import os, runpy, sys
import numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import lib
from sklearn.metrics import roc_auc_score
P = []
_orig = lib.paired_diff


def paired_diff(y, s_a, s_b, idx_list):
    d = np.array([roc_auc_score(y[i], s_a[i]) - roc_auc_score(y[i], s_b[i]) for i in idx_list if len(np.unique(y[i])) == 2])
    P.append(float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
    return _orig(y, s_a, s_b, idx_list)


lib.paired_diff = paired_diff
out = sys.argv[sys.argv.index("--out") + 1]
sys.argv = [os.path.join(here, "run_probes.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
f = out.replace(".csv", "_diff.csv")
d = pd.read_csv(f)
assert len(d) == len(P), (len(d), len(P))
d["p_boot"] = P
for m, g in d.groupby("method"):  # Holm within each method family (3 steps)
    order = g["p_boot"].sort_values().index
    k, run = len(order), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (k - r) * d.loc[i, "p_boot"])); d.loc[i, "p_holm"] = run
d["superior_holm"] = (d["ci_lo"] > 0) & (d["p_holm"] < 0.05)
d.to_csv(f.replace("_diff.csv", "_diff_p.csv"), index=False)
print("\n프로브 - baseline, bootstrap 양측 p와 Holm(3 step)\n" + d.round(4).to_string(index=False))
