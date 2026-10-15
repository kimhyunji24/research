# 선행연구 확인: 같은 과제 반복과 난이도 교락을 어떻게 다뤘나 (2026-10-15)

웹에서 원문(HTML) 또는 검색 요약으로 확인했다. 확인하지 못한 항목은 "확인 필요"로 둔다.

## 1. 에이전트 벤치마크 (성공률 측정)
- **ReAct (Yao 등, 2022, arXiv 2210.03629)**: ALFWorld unseen 134게임에서 평가(Shridhar 등 2020 설정을 따름). 과제 유형별 프롬프트 6개(주석 궤적 2개의 순열).
  디코딩 설정(탐욕/샘플링)은 검색 결과로 확인하지 못함(확인 필요).
- 이후 논문들은 게임 수(134/136), step 상한(30/50), 디코딩(예: temperature 0.4 샘플링)이 제각각이다.
- 의미: 고정 게임을 한 번씩 돌리는 것은 성공률 측정의 표준이다. 게임 재사용 자체는 문제가 아니다.

## 2. 질문 단위 지표를 일부러 쓰는 연구 (불확실성·정답 예측)
- **Kadavath 등 2022 (arXiv 2207.05221), P(IK)**: 질문마다 T=1로 답 30개를 샘플링해 채점하고, 정답 P(IK)를 "샘플 중 정답 비율"로 정의한다(부록).
  즉 목표 자체가 **질문 단위**(이 질문을 아는가)이고, 같은 질문의 여러 샘플로 그 값을 추정한다.
- **Kuhn 등 2023 (Semantic Uncertainty, ICLR 2023)**: 질문마다 여러 답을 샘플링해 의미 군집의 엔트로피를 계산한다. 질문 단위 불확실성.
- **Kossen 등 2024 (Semantic Entropy Probes, arXiv 2406.15927)**: 정답 여부가 아니라 여러 샘플로 계산한 semantic entropy를 hidden state 프로브의 라벨로 쓴다.
  생성 전 hidden state로도 예측된다고 보고. 역시 질문 단위 목표.
- 의미: "이 과제를 실패할지"(우리의 게임 간 예측)는 이 계열에서 정당한 목표로 다뤄진다. 단, 이 경우 주장은 질문(과제) 단위로 한정된다.

## 3. 에이전트 실패를 hidden state로 예측하는 연구
- **Ruan 등 2026, "Doomed from the Start" (arXiv 2607.06503)** (계획서의 "Ruan 등"으로 보임):
  - 환경: TextCraft, WebShop, ALFWorld(스트레스 테스트). 모델: Llama-3.2-3B, Qwen-2.5-7B, Qwen3-1.7B.
  - **과제당 여러 rollout**: TextCraft 100과제 x 8, WebShop 200과제 x 4, ALFWorld 120과제에서 477 rollout(Qwen-2.5-7B).
  - teacher-forced forward로 hidden state 추출, **과제 단위 stratified group k-fold 교차적합**, 라운드별로 살아 있는 episode의 AUC.
  - baseline: 행동만 보는 surface scorer(로그확률, 토큰 수, 길이, 오류 키워드), 활성+surface stacking.
  - **명시적 과제 난이도 baseline이나 과제 내(within-task) 평가는 본문에서 찾지 못함**(본문 약 95% 확인, 마지막 약 5천 자 미확인). 라운드별 AUC가 과제를 합친 것인지 명시되지 않음.
  - 샘플링 온도·디코딩 설정 언급 없음(확인 필요).
  - ALFWorld: Qwen-2.5-7B probe AUC 0.815, Qwen3-1.7B 0.764. 통계는 seed 20개 평균 ± 표준편차.
  - 우리 설계(teacher forcing, 라운드별 AUROC, 과제 단위 그룹 분할, 25/50/75% layer)는 이 논문 구조와 매우 비슷하다.
- **"When Agents Commit Too Soon" (arXiv 2606.22936)**:
  - Llama-3.1-70B ReAct, HotpotQA 100문항 x 10회(T=0.5).
  - **hidden state 선형 프로브는 run 단위 정답을 예측하지 못함(AUC 0.34~0.56)**. 이유를 정답이 "대체로 질문이 결정"하기 때문으로 설명(64문항은 10/10 정답, 23문항은 0/10).
  - 대신 같은 질문의 run 사이 hidden state 유사도("representational commitment")를 쓰고, 이를 질문별 행동 변동성과 상관(r = -0.35, 95% CI -0.51~-0.16)시킨다.
  - 우리 파일럿(탐욕 성공 게임 0.80, 실패 게임 0.025)과 같은 현상이다.
- **"Fail-Fast, Restart-Smart" (arXiv 2608.03222)**: SWE 에이전트의 조기 종료·재시작용 probing head. 상세는 확인하지 않음(확인 필요).

## 4. 우리 연구에 주는 시사점
1. 게임 재사용은 표준이며 문제가 아니다. 문제는 **목표 수준(과제 단위 vs 시도 단위)을 명시하지 않는 것**이다.
2. 과제 단위 목표(P(IK), semantic entropy 계열)는 정당하다. 우리 `main_3b` 결과는 이 수준의 결과로 보고한다: "과제 실패 예측에서 hidden state가 텍스트·진행도 baseline보다 낫다는 증거가 없다".
3. 가장 가까운 선행연구(Ruan 등)는 과제당 여러 rollout을 쓰면서도 과제 난이도를 분리한 평가는 보이지 않는다(확인 범위 내).
   **과제 내 쌍 AUROC와 난이도 오라클 baseline을 넣으면 차별점이 된다.**
4. "When Agents Commit Too Soon"은 run 단위 신호가 약할 수 있음을 보여 준다. 우리 게임 내 결과가 0.5 근처로 나와도 선행 결과와 일관된 부정 결과로 보고할 수 있다(사전 해석표의 세 번째 줄).

## 출처
- https://arxiv.org/pdf/2210.03629
- https://arxiv.org/abs/2207.05221 (부록의 P(IK) 라벨 구성은 검색 요약으로 확인)
- https://iclr.cc/virtual/2023/poster/11013
- https://arxiv.org/abs/2406.15927
- https://arxiv.org/html/2607.06503
- https://arxiv.org/html/2606.22936
- https://arxiv.org/pdf/2608.03222
