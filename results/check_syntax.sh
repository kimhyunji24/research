cd ~/research && . .venv/bin/activate
for f in collect/rollout.py collect/make_game_list.py; do cp /mnt/c/research/$f /tmp/$(basename $f); python -m py_compile /tmp/$(basename $f) && echo "ok $f"; done
cp /mnt/c/research/probes/extra_analysis.py /tmp/ && python -m py_compile /tmp/extra_analysis.py && echo "ok extra_analysis.py"
