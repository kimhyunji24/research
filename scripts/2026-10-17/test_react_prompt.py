"""ReAct 형식 A 패치의 오프라인 점검(모델·환경 없이): 예시 수정, 과제 유형 매핑, 프롬프트 모양, 토큰 수."""
import importlib.util, json, re, sys
spec = importlib.util.spec_from_file_location("ro", "collect/rollout.py")
ro = importlib.util.module_from_spec(spec); spec.loader.exec_module(ro)
ex = json.load(open("prompts/react/alfworld_3prompts.json"))
eps = [json.loads(l) for l in open("/root/research/data/main_3b/Qwen2.5-3B-Instruct/episodes.jsonl")][:250]
seen, cnt = {}, {}
for e in eps:
    h = ro.react_header(ex, e["task_id"])
    assert not re.search(r"^> put ", h, flags=re.M) and "You put the" not in h, ("put action/obs left", e["task_type"])
    assert h.count("Here are two examples") == 1 and h.endswith("\nHere is the task.\n")
    seen.setdefault(e["task_type"], h); cnt[e["task_type"]] = cnt.get(e["task_type"], 0) + 1
print("all 250 games mapped; task types:", cnt)
h = seen["pick_and_place_simple"]
print("move lines:", re.findall(r"> move [^\n]+", h)[:2], "| move obs:", re.findall(r"You move the [^\n]+", h)[:1])
first = "You are in the middle of a room. Looking quickly around you, you see a cabinet 1.\nYour task is to: put some mug on countertop."
trace = [(first, "go to cabinet 1", True), ("You arrive at cabinet 1. The cabinet 1 is closed.", "open cabinet 1", True)]
p = ro.make_prompt_react(h, first, trace, "You open the cabinet 1. The cabinet 1 is open. In it, you see a mug 1.")
print("TAIL:", repr(p[-300:]))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", revision="aa8e72537993ba99e69dfaafa59ed015b17504d1")
ids = tok(p)["input_ids"]
print("tokens:", len(ids), "| first:", tok.convert_ids_to_tokens(ids[:2]), "| last:", tok.convert_ids_to_tokens(ids[-2:]))
c = tok(" take mug 1 from cabinet 1\n", add_special_tokens=False)["input_ids"]
print("cand tokens:", tok.convert_ids_to_tokens(c))
both = tok(p + " take mug 1 from cabinet 1\n")["input_ids"]
print("joint tokenization == prompt + cand:", both == ids + c)
