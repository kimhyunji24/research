import pathlib
p = pathlib.Path("probes/baselines.py"); s = p.read_text()
assert "REACT_SEP" not in s
old = '''def add_text_columns(df):
    """prompt 열이 있으면 text(user 메시지), task(과제 문장), obs(관찰 텍스트)를 추가한다."""
    if "prompt" not in df.columns: return df
'''
new = '''REACT_SEP = "\\nHere is the task.\\n"


def react_parts(prompt):
    """ReAct 형식(chat template 없음, docs/prompt_robustness_plan_2026-10-17.md): 예시를 뺀 현재 게임 부분만 쓴다.
    text = 'Here is the task.' 뒤 전체, task = 'Your task is to:' 줄, obs = 현재(마지막) 관찰."""
    cur = prompt[prompt.index(REACT_SEP) + len(REACT_SEP):]
    m = re.search(r"Your task is to: (.*)", cur)
    lines = cur.split("\\n")
    if lines and lines[-1].strip() == ">": lines = lines[:-1]
    acts = [k for k, l in enumerate(lines) if l.startswith("> ")]
    obs = lines[acts[-1] + 1:] if acts else lines
    obs = " ".join(re.sub(r"Your task is to:.*", "", l) for l in obs).lower()
    return cur, (m.group(1) if m else ""), obs


def add_text_columns(df):
    """prompt 열이 있으면 text(user 메시지), task(과제 문장), obs(관찰 텍스트)를 추가한다."""
    if "prompt" not in df.columns: return df
    if not df["prompt"].iloc[0].startswith("<|im_start|>") and REACT_SEP in df["prompt"].iloc[0]:
        parts = df["prompt"].map(react_parts)
        return df.assign(text=parts.map(lambda t: t[0]), task=parts.map(lambda t: t[1]), obs=parts.map(lambda t: t[2]))
'''
assert s.count(old) == 1
p.write_text(s.replace(old, new)); print("patched")
