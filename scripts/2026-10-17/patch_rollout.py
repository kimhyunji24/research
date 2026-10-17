"""collect/rollout.py에 모델별 설정을 추가한다. 기본값은 기존 동작과 바이트 단위로 같아야 한다(Qwen2.5)."""
import pathlib
p = pathlib.Path("collect/rollout.py")
s = p.read_text()

def rep(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)

rep('''def make_prompt(tok, task, trace, obs, admissible):''',
    '''def make_prompt(tok, task, trace, obs, admissible, tmpl_kwargs=None):''')
rep('''    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)''',
    '''    # tmpl_kwargs: 모델별 chat template 인자(예: Qwen3 enable_thinking=False, Llama 3 date_string 고정). 기본은 없음.
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **(tmpl_kwargs or {}))''')
rep('''def score_candidates(model, tok, prompt, candidates, layers, device, norm):
    """prompt를 1회 prefill해 hidden state를 뽑고, KV cache를 복제해 각 후보 'Action: <a><|im_end|>'의
    로그확률을 한 번에 계산한다.''',
    '''def score_candidates(model, tok, prompt, candidates, layers, device, norm, turn_end="<|im_end|>", cand_batch=0):
    """prompt를 1회 prefill해 hidden state를 뽑고, KV cache를 복제해 각 후보 'Action: <a><turn_end>'의
    로그확률을 계산한다. turn_end는 모델의 assistant 턴 종료 토큰(Qwen <|im_end|>, Llama 3 <|eot_id|>).
    cand_batch > 0이면 후보를 그 크기로 나눠 계산한다(KV cache가 큰 모델의 메모리용, 0이면 한 번에).''')
rep('''    enc = [tok(f"Action: {c}<|im_end|>", add_special_tokens=False)["input_ids"] for c in candidates]
    B, L = len(enc), max(len(e) for e in enc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
''',
    '''    enc_all = [tok(f"Action: {c}{turn_end}", add_special_tokens=False)["input_ids"] for c in candidates]
    P = ids["input_ids"].shape[1]
    step = cand_batch if cand_batch and cand_batch > 0 else len(enc_all)
    totals, ntoks = [], []
    for s0 in range(0, len(enc_all), step):
        t, n = _score_chunk(model, tok, pre, first_lp, enc_all[s0:s0 + step], P, device)
        totals.append(t); ntoks.append(n)
    total, ntok = torch.cat(totals), torch.cat(ntoks)
    score = total / ntok if norm == "mean" else total
    return vec, score.cpu().numpy(), total.cpu().numpy(), ntok.cpu().numpy(), P


@torch.no_grad()
def _score_chunk(model, tok, pre, first_lp, enc, P, device):
    B, L = len(enc), max(len(e) for e in enc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
''')
rep('''    cache = copy.deepcopy(pre.past_key_values); cache.batch_repeat_interleave(B)
    P = ids["input_ids"].shape[1]
''',
    '''    cache = copy.deepcopy(pre.past_key_values); cache.batch_repeat_interleave(B)
''')
rep('''    tok_lp = tok_lp * cmask
    total = tok_lp.sum(1)
    ntok = cmask.sum(1)
    score = total / ntok if norm == "mean" else total
    return vec, score.cpu().numpy(), total.cpu().numpy(), ntok.cpu().numpy(), P
''',
    '''    tok_lp = tok_lp * cmask
    return tok_lp.sum(1), cmask.sum(1)
''')
rep('''    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=getattr(torch, cfg["dtype"])).to(cfg["device"]).eval()''',
    '''    rev = cfg.get("revision")  # HF commit 해시로 고정(없으면 기본 revision)
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], revision=rev)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], revision=rev, torch_dtype=getattr(torch, cfg["dtype"])).to(cfg["device"]).eval()
    tmpl_kwargs = cfg.get("chat_template_kwargs") or {}
    turn_end = cfg.get("turn_end", "<|im_end|>")
    cand_batch = int(cfg.get("cand_batch") or 0)
    # turn_end가 이 모델의 assistant 턴 끝과 같은지 chat template으로 확인한다.
    probe = tok.apply_chat_template([{"role": "user", "content": "x"}, {"role": "assistant", "content": "Action: y"}],
                                    tokenize=False, **tmpl_kwargs)
    assert f"Action: y{turn_end}" in probe, f"turn_end {turn_end!r}가 chat template의 assistant 턴 끝과 다르다: {probe[-80:]!r}"''')
rep('''            prompt = make_prompt(tok, task, trace, text, adm)''',
    '''            prompt = make_prompt(tok, task, trace, text, adm, tmpl_kwargs)''')
rep('''                vec, score, total, ntok, n_prompt = score_candidates(model, tok, prompt, cands, layers,
                                                                     cfg["device"], cfg.get("score_norm", "sum"))''',
    '''                vec, score, total, ntok, n_prompt = score_candidates(model, tok, prompt, cands, layers,
                                                                     cfg["device"], cfg.get("score_norm", "sum"),
                                                                     turn_end, cand_batch)''')
p.write_text(s)
print("patched")
