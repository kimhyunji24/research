"""시간 한도 100 민감도 분석(docs/horizon100_plan_2026-10-17.md, 사전 지정): 라벨만 y100으로 바꿔 run_probes.py를 그대로 실행한다.
y100 = 100 step 안에 성공하지 못하면 1. main_3b에서 30 step 안 실패였지만 data/horizon100에서 31~100 step에 성공한 게임만 y가 1 -> 0으로 바뀐다.
--label y30이면 라벨을 바꾸지 않는다(같은 환경에서 원래 실행을 재현하는 점검용, 결과를 대체하지 않음).
사용: python probes/sens_y100.py [--max_episode 250] [--label y100|y30] -- <run_probes.py 인자들>"""
import json, os, runpy, sys
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
argv = sys.argv[1:]; sep = argv.index("--"); own, rest = argv[:sep], argv[sep + 1:]
max_ep = int(own[own.index("--max_episode") + 1]) if "--max_episode" in own else None
label = own[own.index("--label") + 1] if "--label" in own else "y100"
assert label in ("y100", "y30")
H = os.environ.get("H100_RUN", "data/horizon100/Qwen2.5-3B-Instruct")
late = {e["task_id"] for e in map(json.loads, open(os.path.join(H, "episodes.jsonl"))) if e["success"]}
import lib
_orig = lib.load_run
def load_run(path, hidden_path=None):
    df, hid = _orig(path, hidden_path)
    if max_ep is not None: df = df[df["episode"] < max_ep].reset_index(drop=True)
    ep = df.groupby("episode")["task_id"].first()
    flip = set(ep[ep.isin(late)].index) if label == "y100" else set()
    y30 = df.groupby("episode")["y"].first()
    assert all(y30[e] == 1 for e in flip), "late success game was not a 30-step failure"
    print(f"[{label}] episodes={df['episode'].nunique()} y30 successes={(y30 == 0).sum()} relabeled late successes={len(flip)} "
          f"(eval>=250: {sum(e >= 250 for e in flip)})", flush=True)
    df.loc[df["episode"].isin(flip), "y"] = 0
    return df, hid
lib.load_run = load_run
sys.argv = [os.path.join(here, "run_probes.py")] + rest
runpy.run_path(sys.argv[0], run_name="__main__")
