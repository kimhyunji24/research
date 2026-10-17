"""프롬프트가 같은 (실패, 성공) 평가 쌍에서: 프로브 입력(hidden layer 2)이 정확히 같은가, baseline 4단의 시점 이후 특징(repeat, p_chosen)이 다른가.
라벨은 쌍을 고르는 데만 쓰고 점수·AUROC는 계산하지 않는다."""
import collections, itertools, json, os, sys
import numpy as np
sys.path.insert(0, "probes")
from lib import load_run
runs = ["data/multi_rollout_pilot/Qwen2.5-3B-Instruct", "data/multi_rollout_main_r2/Qwen2.5-3B-Instruct"]
for s in (1, 2, 3):
    same_hid = diff_hid = post_diff = n = 0
    for d in runs:
        if not os.path.exists(f"{d}/hidden.npy"):  # 본 규모는 shard 저장 -> 행 순서대로 이어붙임
            pass
        df, hid = (load_run(d) if os.path.exists(f"{d}/hidden.npy") else (None, None))
        if df is None:
            eps = [json.loads(l) for l in open(f"{d}/episodes.jsonl")]
            hid = np.concatenate([np.load(f"{d}/hidden_shards/ep{e['episode']:05d}.npy") for e in eps])
            np.save("/tmp/r2_hidden.npy", hid)  # 임시(분석 폴더에 쓰지 않음)
            df, hid = load_run(d, "/tmp/r2_hidden.npy")
        r = df[df.step == s]
        for g, gg in r.groupby("task_id"):
            for (_, a), (_, b) in itertools.combinations(gg.iterrows(), 2):
                if a.y == b.y or a.prompt != b.prompt: continue
                n += 1
                if np.array_equal(hid[a.hid_idx, 2], hid[b.hid_idx, 2]): same_hid += 1
                else: diff_hid += 1
                post_diff += (a.repeat != b.repeat) or (a.p_chosen != b.p_chosen)
    print(f"step {s}: identical-prompt eval pairs={n}, hidden exactly equal={same_hid}, hidden differ={diff_hid}, "
          f"baseline post-decision features (repeat/p_chosen) differ={post_diff} ({post_diff/max(n,1):.1%})")
