cd ~/research && . .venv/bin/activate
D=$(python -c "import textworld,os;print(os.path.dirname(textworld.__file__))" 2>/dev/null)
grep -n -E "gamefiles|shuffle|np_random|random|seed|def reset|_next|iter" $D/gym/envs/batch.py 2>/dev/null | head -40
echo ---- 
grep -n -E "gamefiles|shuffle|np_random|random|seed|def reset|_next|iter" $D/gym/envs/text.py 2>/dev/null | head -40
echo ---- register
grep -n -E "def register_games|batch_size|asynchronous|shuffle|gamefiles" $D/gym/__init__.py | head -30
sed -n 262,300p $(python -c "import alfworld,os;print(os.path.dirname(alfworld.__file__))")/agents/environment/alfred_tw_env.py
