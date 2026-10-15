cd ~/research && . .venv/bin/activate && export ALFWORLD_DATA=/root/alfworld_data
cp /mnt/c/research/results/test_env_replay.py results/
python -W ignore results/test_env_replay.py 2>&1 | grep -v -E "it/s|Warning" | tail -6
