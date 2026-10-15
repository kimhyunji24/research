cd ~/research && . .venv/bin/activate
F=$(python -c "import alfworld,os;print(os.path.dirname(alfworld.__file__))" 2>/dev/null)
echo $F
sed -n 1,400p $F/agents/environment/alfred_tw_env.py | grep -n -E "def |game_files|shuffle|num_games|register_games|batch_size|seed|reset|train_eval|max_episode" | head -80
