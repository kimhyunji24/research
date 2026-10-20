# 선행연구 인용 점검표 (2026-10-20)

원고(`docs/paper_draft_en.md`, `_ko.md`)에 들어간 선행연구 수치·서술을 한 줄씩 적었다. 투고 전에 PDF로 대조하고 "PDF" 열을 채운다.
확인 수준: **HTML** = 별도 작성 세션이 arXiv HTML 본문과 대조(2026-10-20, 원문 문장은 그 세션 기록에 있고 저장소에는 정정 결과만 있음: commit `20168e5`, `docs/literature_review_2026-10-17.md`),
**초록** = 주 세션이 arXiv 초록 페이지로 확인(2026-10-20).
**PDF 열** = 주 세션이 arXiv PDF(최신 판본, 2026-10-20 내려받음)를 `pdftotext`로 뽑아 검색·대조. 줄 번호는 `pdftotext -layout` 출력 기준(`/root/citation_pdfs_2026-10-20/<id>.txt`, 저장소 밖). ✓ = 일치, 부재 서술은 검색어로 찾지 못했다는 뜻이며 저자 확인은 아니다.

| 참고 | 원고에 쓴 내용 | 원고 위치 | 확인 수준 | 원문에서 볼 곳 | PDF |
|---|---|---|---|---|---|
| [1] AgentStop | 소비자 기기에서 전력·발열·배터리 상승, 에너지 상당량이 실패 실행에 쓰임 | 1절 첫 문단 | HTML | 서론·측정 절(실패 실행 에너지 비교) | ✓ 초록 "power draw, temperature, and battery drain"(42행), 서론 87·95행 |
| [1] | Qwen3-30B-A3B 낭비 에너지 15–20% 감소, 성능 하락 5% 미만 | 2.3 | HTML | 초록·주 결과 표 | ✓ 52–54행, 137–138행 |
| [1] | 1.7B 모델에서 실패·성공 로그확률이 더 겹침 | 2.3 | HTML | 모델별 분석 절 | ✓ 6.1절 412–414행 "runs show more logprobs overlap" |
| [2] Fail-Fast | 0.6B 모니터, 9–31B 공개 모델 세 개에 0.1–0.6% 계산량 | 1절, 2.3, 3.7 | HTML | 4.1절 비용 추정(2N FLOPs) | ✓ 259–261행(2N FLOPs, 9–31B, 0.1%–0.6%), 공개 모델 세 개는 81–82행 |
| [2] | 5% 오탐에서 토큰 20.4% 절감, AgentStop 변형 12.5%, step 수 대조 11.4%(Qwen3.6-27B) | 2.3, 5절 | HTML | 주 결과 표 | ✓ 초록 41–43행, 서론 76–79행("duration-only control’s 11.4%") |
| [2] | 학습 때 진행 정도를 맞춘 성공·실패 prefix 쌍, 평가는 100 인스턴스 × 5회 합침, 인스턴스 내 분석 없음 | 2.3 | HTML(부재는 검색 근거) | 학습 데이터 절(103행 부근), 평가 절 | ✓ 161–167행(step-fraction decile 안에서 짝, "controls for both instance difficulty and trajectory progress"), 706–707행(100 × 5 = 500 합침). 부재: "within-instance", "same instance" 검색 0건 |
| [3] Doomed | 라운드 1 AUC 0.86, 0.81(TextCraft, 세 모델 중 두 모델) | 1절 | HTML | 주 결과 표 | ✓ 318–320행(TextCraft Qwen-2.5-7B, Qwen3-1.7B) |
| [3] | 셀당 800 episode, 1.7B–7B 세 에이전트, 과제 단위 묶은 분할, 모델·환경마다 층 하나 | 2.2 | HTML | 실험 설정 절 | ✓ 244–247행(Llama-3.2-3B, Qwen-2.5-7B, Qwen3-1.7B, 800), 434–438행(grouped splits, pilot-fixed layer) |
| [3] | ALFWorld: Qwen-2.5-7B 477 rollout AUC 0.815, Qwen3-1.7B 0.764 | 2.2 | HTML | 부록 E | ✓ 본문 경계 시험 단락 401–403행, 0.764는 같은 단락 다음 문장(raw 추출 628행), 부록 E 표 5(둘 다 477 rollout) |
| [3] | 행동 scorer 특징(로그확률, 토큰 수, prefix 길이, 오류 키워드), 과제당 4–8 rollout, 과제 내 평가·난이도 통제 보고 없음, 분할이 분포 안에 머문다고 저자 언급 | 1절, 2.2, 4.5 | HTML(부재는 검색 근거) | baseline 절, 한계 절 | ✓ 141–149행(특징 다섯 개), 247–248행(TextCraft 8, WebShop 4), 436행 "remain in-distribution". 부재: "within-task", "same task" 검색 0건 |
| [4]–[6] | P(IK), 의미 엔트로피, semantic entropy probe 서술(수치 없음) | 1절, 2.1 | HTML | 초록 | ✓ 제목·초록 확인(2207.05221, 2302.09664, 2406.15927) |
| [7] | 질문만 읽은 활성의 프로브, 7B–70B, 수학에서 일반화 약함 | 2.1 | HTML | 초록·결과 | ✓ 335–336행(7B–70B 여섯 모델), 초록 37행·198행(GSM8K) |
| [8] EarlyEval | step 13–26% 감소 | 2.3 | HTML | 초록·주 결과 | ✓ 초록 38행 |
| [9] Mehta & Datta | 100문항 중 64개 항상 정답, 23개 한 번도 아님, 회차 단위 프로브 AUC 0.34–0.56 | 1절, 2.4 | HTML | 결과 절 | ✓ 404–405행("64 questions are 10/10 correct, 23 are 0/10") |
| [9] | 텍스트 유사도 r = −0.63 대 hidden state −0.35, step 4에서 r = −0.35(Llama-3.1-70B) | 1절, 2.2 | HTML | 상관 분석 표 | ✓ 207–208행, 353·361행, 254행(layer 40, step 4, r = −0.348) |
| [10] Bulut | 합친 AUROC = 문제 내·문제 간 가중합, 문제만 쓰는 예측기는 문제 안에서 0.5 | 1절, 2.4 | HTML | 분해식 절 | ✓ 식 (1) 330행, 332행("every question-only predictor has AUROC_within = 1/2 exactly") |
| [10] | 재구성 프로브 합친 0.849, 문제 내 0.496(0.466–0.527), 사전 등록 검정 이득 없음 | 1절, 2.4, 5절 | HTML | 재구성 절, 사전 등록 결과 | ✓ 377·384행, 4.1절 277행 |
| [10] | 궤적을 읽지 않는 난이도 대리 변수 AUROC 0.873(R1 생성 19만 건) | 2.4 | HTML | 공개 덤프 분석 절 | ✓ 초록 29행, 349행(192,315건) |
| [11] Lugoloobi | 생성 전 활성 선형 프로브로 수학·코딩 성공 예측, 질문 길이·TF-IDF보다 높음, 모델별 난이도 | 1절, 2.1, 5절 | 초록 | 초록, 결과 표(TF-IDF 대비 수치) | **정정** 초록은 "substantially outperforming"이지만 주 결과 표(346–367행) 15개 설정 중 선형 프로브가 TF-IDF보다 높은 것 14개(예외 DeepSeek-R1-Distill-Qwen-7B 코드 0.81 대 0.83), 길이보다 높은 것 14개(예외 Qwen2.5-1.5B Greedy 0.68 대 0.73). 원고 2.1·5절의 "clearly outperformed"를 "15개 중 14개"로 바꿈 |
| [14] Khan & Khan | Gemma 4 E4B 28층, 대리 라벨 35개, 측정된 추가 추론 비용 0, 1B–1.5B 개선 없음 | 2.2 | HTML + 초록("zero measured inference overhead") | 초록, 방법 절(예시 35개) | ✓ 30·142행(layer 28), 511행(35개), 31행, 36행(1B–1.5B) |
| [15] Pan 등 | 관찰 가능한 궤적 특징이 내부 신호 baseline과 견줄 만함, 과제 단위 교차검증 | 2.3, 5절 | HTML | 결과 표(내부 신호 baseline 종류 확인) | ✓ 초록 43행 "competitive with internal-signal baselines", 과제 단위 분할은 raw 추출 447행("by task") |
| [16] Poulis 등 | 진실성 프로브의 기하·일반화가 프롬프트 템플릿·지시문에 따라 달라짐 | 5절 | 초록 | 초록, 템플릿 실험 절 | ✓ 초록 42–52행, 5절 제목 460행 "Model instructions change the geometry" |
| [17] Marks & Tegmark | 충분한 규모에서 참·거짓 진술 표현의 선형 구조, 데이터셋 사이 전이, 방향의 인과적 관여 | 2.1 | PDF(2026-10-21, 리뷰 제안으로 추가) | 초록 | ✓ 초록(COLM 2024 판, 머리말 "Published as a conference paper at COLM 2024") |
| [18] Burns 등 | 라벨 없는 활성에서 진술과 부정의 논리적 일관성을 만족하는 방향을 찾아 예·아니오 지식 복원 | 2.1 | PDF(2026-10-21) | 초록 | ✓ 초록. 학회 수록 여부는 PDF에서 확인 못 해 arXiv로 인용 |
| [19] Ghasemabadi & Niu | 고정 모델(1.7B–20B)의 hidden state·attention 패턴으로 자기 답의 정답 여부 예측, 내부 baseline과 큰 외부 심사 모델보다 나음 | 2.1 | PDF(2026-10-21) | 초록 | ✓ 초록("frozen backbones ranging from 1.7B to 20B", "outperforms strong internal baselines and large external judges") |
| [20] Dubey | 관찰 가능한 단계 기록(출력 임베딩, 토큰 불확실성, 행동 메타데이터)만으로 정상 실행에서 학습한 모니터로 실패 감지 | 2.3 | PDF(2026-10-21) | 초록 | ✓ 초록("observable step telemetry", "train only on healthy runs") |
| [21] Advani | TF-IDF 감지기가 LLM 심사 모델보다 거짓 성공을 더 잘 찾음, 과제 분리 AUROC 0.83(tau2-bench)·0.95(AppWorld), 지연 3,300배 낮음 | 5절 | PDF(2026-10-21) | 초록, 서론 기여 | ✓ 초록 37–41행, 서론 88–95행. 워크숍(Failure Modes in Agentic AI) 수록 표기는 arXiv로 대신함 |

## 우리 계산 (선행연구 수치가 아님)
- 모니터 비용 13–16%, 약 2.4배: Qwen2.5 공개 파라미터 수와 2N FLOPs 근사(3.7절). 파라미터 수는 Qwen2.5 모델 카드에서 다시 확인한다.

## 남은 확인
- 2026년 프리프린트는 심사 판본(학회 수록본)이 있는지 투고 직전에 확인한다.
- PDF는 저장소에 넣지 않았다(저작권). 다시 받으려면 arXiv id로 내려받는다.
