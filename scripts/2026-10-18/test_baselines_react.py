"""ReAct 텍스트 변환 점검: (1) v2·Qwen3 형식은 패치 전후 같음 (2) ReAct 형식 파싱 결과 확인. 결과 열(성공)은 보지 않는다."""
import sys, json, importlib.util, pandas as pd
sys.path.insert(0, "probes")
def load(path):
    spec = importlib.util.spec_from_file_location("b_" + str(abs(hash(path))), path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
old, new = load(sys.argv[1]), load("probes/baselines.py")
def steps(d, n=400):
    rows = []
    for l in open(f"{d}/steps.jsonl"):
        rows.append(json.loads(l))
        if len(rows) >= n: break
    return pd.DataFrame(rows)[["episode", "step", "action", "prompt"]]
for d in ["data/main_3b/Qwen2.5-3B-Instruct", "data/main_qwen3_4b/Qwen3-4B"]:
    df = steps(d); a = old.add_text_columns(df); b = new.add_text_columns(df)
    pa, pb = old.progress_features(a), new.progress_features(b)
    print(d, "text/task/obs same:", all((a[c] == b[c]).all() for c in ["text", "task", "obs"]), "progress same:", (pa == pb).all().all())
df = steps("data/react_qwen25_3b/Qwen2.5-3B-Instruct", 60); b = new.add_text_columns(df)
print("react: example text leaked:", b["text"].str.contains("Here are two examples").any(), "| empty task:", (b["task"] == "").sum(),
      "| target parsed:", b.groupby("episode")["task"].first().map(lambda t: new.target_object(t) is not None).mean())
for k in [0, 2]:
    r = b.iloc[k]; print(f"step {r['step']}: task={r['task']!r}\n  obs={r['obs'][:160]!r}\n  text tail={r['text'][-120:]!r}")
pf = new.progress_features(b); print(pf.head(4).to_string())
