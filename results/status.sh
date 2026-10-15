cd ~/research; date
echo "--- probes chain"; tail -n 2 results/explore_probes_chain.log
echo "--- confirm collection"; wc -l data/main_3b/Qwen2.5-3B-Instruct/episodes.jsonl
tail -c 300 ~/main_3b_confirm.log | tr '\r' '\n' | tail -2
