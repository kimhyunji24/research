"""baseline 사다리 4~5단용 특징 추출 (docs/analysis_plan_2026-10-15.md 참고).

모든 진행도 특징은 현재 step 이전에 실행한 action에서만 계산한다. 관찰 텍스트는 프롬프트(user 메시지)에서 읽는다.
프롬프트는 최근 6 step만 담으므로 `obj_seen`은 그 창 안에서만 확인 가능하다.
"""
import re
import numpy as np
import pandas as pd

TRANSFORM = ("heat", "cool", "clean", "use")
PROGRESS_COLS = ["n_go_unique", "n_open", "n_take_obj", "took_obj", "n_transform", "n_move", "obj_seen", "obj_parsed"]

_PATTERNS = [
    r"look at (\w+) under",
    r"(?:heat|cool|clean) some (\w+) and put it",
    r"find two (\w+) and put them",
    r"put (?:a|an|some|two) (?:(?:clean|hot|cool|cold) )?(\w+) (?:in|on) ",
]


def user_content(prompt):
    i = prompt.index("<|im_start|>user\n") + len("<|im_start|>user\n")
    j = prompt.index("<|im_end|>", i)
    return prompt[i:j]


def target_object(task):
    t = task.lower().strip().rstrip(".")
    for p in _PATTERNS:
        m = re.match(p, t)
        if m: return m.group(1)
    return None


def observation_text(user):
    """user 메시지에서 관찰 텍스트만 남긴다(과제 문장, 행동 이력, admissible 목록 제외)."""
    keep = []
    for line in user.split("\n"):
        if line.startswith(("Task:", "Admissible actions:", "Actions that did nothing", "Action:")): continue
        line = re.sub(r"Observation:\s*", "", line)
        line = re.sub(r"Your task is to:.*", "", line)
        keep.append(line)
    return " ".join(keep).lower()


def add_text_columns(df):
    """prompt 열이 있으면 text(user 메시지), task(과제 문장), obs(관찰 텍스트)를 추가한다."""
    if "prompt" not in df.columns: return df
    u = df["prompt"].map(user_content)
    df = df.assign(text=u,
                   task=u.map(lambda s: re.match(r"Task: (.*)", s).group(1) if re.match(r"Task: (.*)", s) else ""),
                   obs=u.map(observation_text))
    return df


def progress_features(df):
    """df: episode, step 순으로 정렬된 steps 표(action 필요, task/obs는 있으면 사용). PROGRESS_COLS 열을 가진 표를 반환."""
    out = np.zeros((len(df), len(PROGRESS_COLS)))
    has_text = "task" in df.columns
    pos = {i: k for k, i in enumerate(df.index)}
    for _, g in df.groupby("episode", sort=False):
        acts = g["action"].tolist()
        obj = target_object(g["task"].iloc[0]) if has_text else None
        for n, i in enumerate(g.index):
            prev = acts[:n]
            go = {a for a in prev if a.startswith("go to ")}
            take_obj = [a for a in prev if obj and re.match(rf"take {obj} \d+", a)]
            seen = bool(obj and re.search(rf"\b{obj}\b", g["obs"].iloc[n])) if has_text else False
            out[pos[i]] = [len(go), sum(a.startswith("open ") for a in prev), len(take_obj), int(len(take_obj) > 0),
                           sum(a.split()[0] in TRANSFORM for a in prev), sum(a.startswith("move ") for a in prev),
                           int(seen), int(obj is not None)]
    return pd.DataFrame(out, columns=PROGRESS_COLS, index=df.index)
