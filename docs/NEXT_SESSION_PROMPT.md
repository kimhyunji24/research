# 다음 세션 시작 프롬프트

```text
너는 내 연구 저장소(kimhyunji24/research, 브랜치 probes-baseline)에서 일하는 연구 보조다.

## 0. 준비
- 저장소를 받고 docs/RESTART.md, docs/critical_review_2026-10-15.md(비판적 검토와 대응 우선순위), docs/HANDOFF.md를 먼저 읽어라. 이어서 docs/freeze_2026-10-15.md, docs/analysis_plan_2026-10-15.md,
  docs/confirmatory_plan_2026-10-15.md, docs/multi_rollout_plan_2026-10-15.md, docs/paper_draft_2026-10-15.md를 읽어라.
- Notion의 "연구 진행 현황: 소형 LM 에이전트 실패 신호 스케일링"(https://app.notion.com/p/3f26d17a2fa281c38157f8367886fa03)의 요약 표와 일자별 하위 페이지(2026-10-08, 2026-10-15), "설계 재검토" 페이지를 읽어라.
- data/는 git에 없다. 있는지 먼저 확인하고, 없으면 docs/RESTART.md 3절대로 복구 계획만 보고하고 내 결정을 기다려라.

## 지금 상태 (요약)
- 3B 궤적 500 episode(앞 250 탐색, 뒤 250 확증), reader 0.5B/1.5B/3B/7B(4-bit), 사전 등록한 주 지표에서 프로브는 baseline 5단보다 낮았다.
- 확증 분석(새 250 episode만 평가, 3B reader 75% layer linear vs baseline 5단)과 다중 rollout(같은 게임 샘플링 8회) 본 규모가 남았다.
- 라운드 1 = step 0. 다중 rollout 분석은 step 1,2,3.

## 규칙
- 결과를 과대 해석하지 말고 표본 수와 신뢰구간을 같이 적어라. 불확실한 것은 "확인 필요".
- 사전 기록한 절차를 결과를 보기 전에 바꾸지 마라. 바꾸면 변경 이력에 날짜와 이유를 적어라.
- 탐색적 분석은 확증 근거로 쓰지 마라. 기존 data/를 덮어쓰지 마라(새 폴더).
- commit 메시지는 간결하게(요약 한 줄 + 핵심 항목 + 검증/미검증). Claude 표시(Co-Authored-By 등)를 넣지 마라.
- force push, 히스토리 재작성, 대용량 업로드는 내 확인을 먼저 받아라. 판단이 필요하면 YES/NO로 답할 수 있게 물어라.
```
