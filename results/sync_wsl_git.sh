cd ~/research && git fetch -q origin && git reset -q --hard origin/probes-baseline && git log --oneline -1 && git status --short | head -3
echo "claude 언급(원격 브랜치): $(git log origin/probes-baseline --format='%an <%ae>%n%B' | grep -i -c claude)"
bash /mnt/c/research/results/status.sh | tail -4
