"""collect/rollout.py에 prompt_style: react(ReAct 원 ALFWorld 예시, 형식 A: chat template 없는 completion)를 추가한다.
기본값(prompt_style 없음 = v2)은 기존 동작과 바이트 단위로 같아야 한다. docs/prompt_robustness_plan_2026-10-17.md"""
import pathlib
p = pathlib.Path("collect/rollout.py"); s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


REACT_BLOCK = r'''# ---- ReAct 원 ALFWorld 예시 (Yao 등 2023, prompts/react/alfworld_3prompts.json), 형식 A
REACT_PREFIXES = {"pick_and_place": "put", "pick_clean_then_place": "clean", "pick_heat_then_place": "heat",
                  "pick_cool_then_place": "cool", "look_at_obj": "examine", "pick_two_obj": "puttwo"}


def react_adapt(text):
    """원 예시의 최소 수정(계획서 표의 1번): put A in/on B -> move A to B, 그 관찰 문장도 같은 방식으로."""
    text = re.sub(r"^> put (.+?) in/on (.+)$", r"> move \1 to \2", text, flags=re.M)
    return re.sub(r"You put the (.+?) in/on the (.+?)\.", r"You move the \1 to the \2.", text)


def react_header(examples, gamefile):
    """원 notebook과 같은 머리말 + 과제 유형별 예시 2개(react_{유형}_1, react_{유형}_0)."""
    name = gamefile.replace("\\", "/").split("/")[-3]
    v = next(v for k, v in REACT_PREFIXES.items() if name.startswith(k))
    return ("Interact with a household to solve a task. Here are two examples.\n"
            + react_adapt(examples[f"react_{v}_1"]) + react_adapt(examples[f"react_{v}_0"]) + "\nHere is the task.\n")


def make_prompt_react(header, first_obs, trace, cur_obs):
    """원 alfworld_run과 같은 completion 문자열: 머리말 + 첫 관찰 + '\n>' + 매 step ' {action}\n{observation}\n>'.
    trace[i] = (그 action 직전 관찰, action, ok). action i 다음 관찰은 trace[i+1]의 관찰, 마지막은 cur_obs."""
    out = header + first_obs + "\n>"
    for i, (_, a, _) in enumerate(trace):
        nxt = trace[i + 1][0] if i + 1 < len(trace) else cur_obs
        out += f" {a}\n{nxt}\n>"
    return out


SKIP_ACTIONS = {"help", "inventory", "look"}'''

rep('SKIP_ACTIONS = {"help", "inventory", "look"}', REACT_BLOCK)
rep('def score_candidates(model, tok, prompt, candidates, layers, device, norm, turn_end="<|im_end|>", cand_batch=0):',
    'def score_candidates(model, tok, prompt, candidates, layers, device, norm, turn_end="<|im_end|>", cand_batch=0,\n'
    '                     cand_fmt="Action: {c}{turn_end}"):')
rep('    enc_all = [tok(f"Action: {c}{turn_end}", add_special_tokens=False)["input_ids"] for c in candidates]',
    '    enc_all = [tok(cand_fmt.format(c=c, turn_end=turn_end), add_special_tokens=False)["input_ids"] for c in candidates]')
rep(r'''    probe = tok.apply_chat_template([{"role": "user", "content": "x"}, {"role": "assistant", "content": "Action: y"}],
                                    tokenize=False, **tmpl_kwargs)
    assert f"Action: y{turn_end}" in probe, f"turn_end {turn_end!r}가 chat template의 assistant 턴 끝과 다르다: {probe[-80:]!r}"''',
    r'''    style = cfg.get("prompt_style", "v2")
    if style == "v2":
        probe = tok.apply_chat_template([{"role": "user", "content": "x"}, {"role": "assistant", "content": "Action: y"}],
                                        tokenize=False, **tmpl_kwargs)
        assert f"Action: y{turn_end}" in probe, f"turn_end {turn_end!r}가 chat template의 assistant 턴 끝과 다르다: {probe[-80:]!r}"
        cand_fmt = "Action: {c}{turn_end}"
    elif style == "react":  # 형식 A: chat template 없음, 후보는 '>' 뒤의 ' {행동}\n'
        react_examples = json.load(open(cfg.get("react_prompts", "prompts/react/alfworld_3prompts.json")))
        cand_fmt = " {c}\n"
    else:
        raise SystemExit(f"알 수 없는 prompt_style: {style}")''')
rep('        trace, won = [], False',
    '        trace, won = [], False\n'
    '        if style == "react":\n'
    '            header = react_header(react_examples, gamefile)\n'
    '            first_obs = "\\n".join(text.split("\\n\\n")[1:])  # 원 notebook과 같이 환영 문구를 뺀 첫 관찰(과제 문장 포함)')
rep('            prompt = make_prompt(tok, task, trace, text, adm, tmpl_kwargs)',
    '            prompt = (make_prompt(tok, task, trace, text, adm, tmpl_kwargs) if style == "v2"\n'
    '                      else make_prompt_react(header, first_obs, trace, text))')
rep('                                                                     turn_end, cand_batch)',
    '                                                                     turn_end, cand_batch, cand_fmt)')
p.write_text(s)
print("patched")
