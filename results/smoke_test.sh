set -u
cd ~/research && . .venv/bin/activate && export ALFWORLD_DATA=/root/alfworld_data
C=configs/decision_rerun.yaml
rm -rf data/_smoke_A data/_smoke_B
R="python -W ignore collect/rollout.py --config $C"
echo "--- fresh 3"; $R --num_episodes 3 --out_dir data/_smoke_A 2>&1 | grep -E '^\[Qwen' | tail -3
echo "--- again without flag (should refuse)"; $R --num_episodes 3 --out_dir data/_smoke_A 2>&1 | tail -1
echo "--- resume to 5"; $R --num_episodes 5 --out_dir data/_smoke_A --resume 2>&1 | grep -E 'resume|^\[Qwen|Error|불가' | tail -5
echo "--- fresh 5 (reference)"; $R --num_episodes 5 --out_dir data/_smoke_B 2>&1 | grep -E '^\[Qwen' | tail -2
python -I - <<'PY'
import json,os,numpy as np
L=lambda p:[json.loads(l) for l in open(p)]
A='data/_smoke_A/Qwen2.5-3B-Instruct/'; B='data/_smoke_B/Qwen2.5-3B-Instruct/'
ea,eb,sa,sb=L(A+'episodes.jsonl'),L(B+'episodes.jsonl'),L(A+'steps.jsonl'),L(B+'steps.jsonl')
print('episodes',len(ea),len(eb),'same task order',[e['task_id'] for e in ea]==[e['task_id'] for e in eb])
print('steps',len(sa),len(sb),'same actions',[s['action'] for s in sa]==[s['action'] for s in sb])
ha,hb=np.load(A+'hidden.npy'),np.load(B+'hidden.npy')
print('hidden',ha.shape,hb.shape,'max abs diff',float(np.abs(ha.astype('f4')-hb.astype('f4')).max()) if ha.shape==hb.shape else 'shape differs')
s=sa[0]; print('keys',sorted(s))
print('prompt_tokens',s['prompt_tokens'],'n_cand',s['n_cand'],len(s['cands']),len(s['cand_logp']),len(s['cand_ntok']))
print('hid_idx ok',max(x['hid_idx'] for x in sa)==len(ha)-1,'shards',len(os.listdir(A+'hidden_shards')))
print('steps.jsonl MB',round(os.path.getsize(A+'steps.jsonl')/1e6,2),'for',len(sa),'steps')
PY
