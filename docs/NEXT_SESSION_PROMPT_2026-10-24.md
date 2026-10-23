너는 "소형 Qwen 에이전트의 실패 예측(ALFWorld)" 연구를 이어서 한다. 저장소 kimhyunji24/research, 브랜치 probes-baseline, WSL 경로 ~/research(Windows에서 \\wsl.localhost\Ubuntu-22.04\root\research). 먼저 docs/experiment_report_2026-10-24.md(전체 실험 보고서)와 docs/position_plan_2026-10-24.md(최근 계획·변경 이력)를 읽고 시작해.

## 규칙(반드시 지킬 것)
- 과대 해석 금지. 항상 표본 수와 95% CI를 함께 적고, 불확실하면 "확인 필요"로 표시. 사전 등록·확증 결과는 "보인다", 탐색·사후는 "시사한다". "동등", "우수"는 미리 정한 판정 규칙에 따를 때만.
- 사전 등록 절차는 결과를 본 뒤 바꾸지 않는다. 바꾸면 계획서 변경 이력에 날짜와 이유를 적는다. 사전 등록 분석은 1회만(확증, 게임 내 주 분석, Qwen3-4B 분석은 다시 돌리지 않음). 새 분석은 계획 문서를 먼저 쓰고 commit한 뒤 1회 실행.
- 게임 내 독립 반복, 온도를 바꾼 재수집, 표본 확대 금지(유일한 예외는 docs/collect_ext1000_plan_2026-10-21.md). 기존 data/는 덮어쓰지 않는다.
- 실제 라벨 분석 전에는 라벨을 섞은 smoke로 파이프라인만 확인한다. 확인 명령 출력에 승패·성공률이 보이지 않게 한다.
- GPU 작업은 내가 있을 때만.
- commit은 한 세션만. 메시지는 요약줄 + 핵심 항목 + 검증/미검증, Claude·세션 표시 없음, 작성자 `git -c user.name=kimhyunji24 -c user.email=hyunji4560@gmail.com`, 메시지는 Write로 UTF-8 파일을 만든 뒤 -F. commit·push는 내 승인 뒤에만.
- push는 WSL에서 안 됨: Windows clone C:\Users\catholic\Desktop\research에서 `git fetch --no-tags //wsl.localhost/Ubuntu-22.04/root/research probes-baseline`, fast-forward 확인 후 `git push origin <sha>:refs/heads/probes-baseline`를 내 터미널 탭에서 실행(브라우저 로그인 필요할 수 있음). force push·이력 재작성·대용량 업로드는 먼저 물어봐.
- WSL 명령: C:\Users\catholic\AppData\Local\Temp\ws\에 스크립트를 쓰고 CRLF를 제거한 뒤 `wsl.exe -d Ubuntu-22.04 --cd /root -- bash /mnt/c/.../script.sh`. 모든 스크립트는 `export ALFWORLD_DATA=/root/alfworld_data`로 시작. Python 파일은 Write로. WSL에는 /mnt/d가 없다(USB 복사는 PowerShell로).
- 진행 단계마다 Notion 2026-10-24 페이지(id 4026d17a-2fa2-81a1-8022-e4d2fe3430f9) 또는 새 날짜 페이지에 표와 짧은 글로 기록.
- 원고: 영문 기본, 국문 동기화, 숫자는 results/*.csv에서, 동사 강도는 증거 수준에 맞춤. 순서: md → 전체 .tex → `python3 scripts/2026-10-21/extract_floats.py en|ko` → 단축본 .tex → `python3 scripts/2026-10-21/check_short_numbers.py en|ko` → latexmk(XeLaTeX).
- 판단이 필요한 질문은 YES/NO로, 내 의견과 이유를 붙여 물어봐.

## 현재 상태(2026-10-24 03:45 기준)
- GitHub probes-baseline = ed24ff0. 로컬에는 그 위에 1b40439(위치 계획), 394504a(추출 스크립트), af4681a(분석 스크립트·순열 10회·계획 보완)가 commit되어 있고 push 안 함.
- commit 전(작업 트리): results/positions_{lc,diff,check}.csv, results/positions.log, docs/experiment_report_2026-10-24.md, docs/NEXT_SESSION_PROMPT_2026-10-24.md, docs/position_plan_2026-10-24.md(실행·열람 기록), 원고 6개(docs/paper_draft_{en,ko}.md, paper/main_{en,ko}{,_short}.tex), floats, PDF 4개.
- data/positions_3b(184MB)는 git 밖. USB 백업: D:\research_backup_2026-10-24_af4681a_wip(bundle af4681a, 미commit patch, untracked tar, positions tar, sha256 7/7 일치). USB의 D:\wsl_ubuntu\ext4.vhdx가 실행 중인 WSL 디스크이므로 USB를 뽑기 전에 `wsl --shutdown`.
- 핵심 결과: 프로브는 어떤 설정에서도 baseline 5단을 넘지 못했다. 학습 약 1,200개까지 늘려도(주 대비 −0.008, −0.017 ~ +0.002), 읽는 위치를 5개로 바꿔도(위치 − P75 −0.014 ~ +0.008, Holm p 0.33–1.00) 결론은 같았다. 순열 귀무 SD가 장면 bootstrap SE의 약 1.3–2.6배여서 1,500 episode 사후 구간은 너무 좁을 수 있다.

## 할 일(순서대로, 각 단계마다 내 승인을 받을 것)
1. **commit 정리.** 위 commit 전 파일을 두 개로 나눠 commit할지 물어봐: (a) 위치 분석 결과와 계획서 열람 기록, (b) 원고 갱신과 실험 보고서·이 프롬프트. 각각 요약줄에 "사후 탐색"을 표기. push는 별도 승인.
2. **push.** 1b40439…최신 commit을 Windows clone에서 fetch → fast-forward 확인 → push → ls-remote로 검증. 이후 USB 백업을 최신 bundle로 갱신(기존 폴더는 덮어쓰지 말고 새 폴더).
3. **원고 PDF 시각 점검.** paper/main_en.pdf, main_ko.pdf에서 새 문단(4.8절 위치 비교·구간 보정), 표 9의 새 10행, 표 A8의 새 4행이 깨지지 않았는지, 표가 페이지를 넘지 않는지 확인. 3,053·108 같은 다른 숫자도 results/*.csv와 다시 대조.
4. **구간 보정이 기존 사후 주장에 주는 영향 점검(새 사후 분석, 계획 먼저).** 순열 귀무로 P75 − R4, R5 − R4, 대비 12개를 다시 점검하는 계획을 써서 보여줘(순열 횟수, 시간 추정, 해석 규칙). 승인 후 1회 실행. 50% 프로브 − 5단처럼 큰 차이도 같은 방식으로 확인할지 물어봐.
5. **(선택) 순열 99회 이상.** 위치 비교를 Holm과 함께 순열 p로 판정하려면 99회 이상 필요. CPU 약 3시간 추정(확인 필요). 할지 YES/NO로 물어봐.
6. **(선택, 새 계획 필요) 프롬프트 간 전이 실패의 원인 분리.** v2와 ReAct 프롬프트에서 같은 의미의 위치(예: 최근 관찰 끝, 이력 평균)로 hidden을 뽑아 전이를 다시 보는 실험. 재추출만이고 새 episode는 없음. GPU 필요.
7. **(선택, 새 계획 필요) 더 강한 텍스트 baseline.** 미세조정한 소형 텍스트 모델 등. 시간·GPU 비용을 먼저 추정해서 보여줘.
8. **기타 확인 필요.** disk_check의 커널 ext4 경고 3건의 내용, data/positions_3b의 외부 백업 위치.
