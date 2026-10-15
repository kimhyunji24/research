# 소형 LM 에이전트 실패 신호 스케일링

소형 언어모델(0.5B~3B)이 ALFWorld에서 에이전트로 일할 때, hidden state로 episode 실패를 예측할 수 있는지 본다.
설계와 진행 기록은 Notion 페이지 "연구 진행 현황: 소형 LM 에이전트 실패 신호 스케일링"과 하위 페이지 "설계 재검토"에 있다.
**다음 작업은 `docs/HANDOFF.md`부터 읽는다.**

## 폴더

| 경로 | 내용 | git |
|---|---|---|
| `collect/rollout.py` | ALFWorld rollout + hidden state 저장, episode 단위 중간 저장/`--resume` | 포함 |
| `configs/` | `alfworld_base.yaml`, `pilot.yaml`, `decision.yaml`, `decision_rerun.yaml`, `maxstep50.yaml`, `main_3b.yaml` | 포함 |
| `probes/` | 라운드별 AUROC 평가 코드 (**수정 필요 항목이 있다**, `docs/HANDOFF.md` 참고) | 포함 |
| `docs/` | 설정 동결 기록, 인수인계 | 포함 |
| `results/` | 예비 분석 결과 CSV (연구 결과가 아니라 코드 점검용) | 포함 |
| `data/` | 수집 결과 | **제외 (.gitignore)** |
| `$ALFWORLD_DATA` | ALFWorld 게임 데이터 (약 2.3GB) | **제외** |

## 데이터 (git에 없으므로 직접 옮겨야 한다)

수집 결과는 `<out_dir>/<모델 이름>/` 아래에 생긴다. 실제 확인한 구성이다(2026-10-15).

```
data/main_3b/Qwen2.5-3B-Instruct/          # configs/main_3b.yaml, 본 수집 250 episode
data/decision_rerun/Qwen2.5-3B-Instruct/   # configs/decision_rerun.yaml, 예비 100 episode (prompt 저장 이전 코드)
data/maxstep50/Qwen2.5-3B-Instruct/        # configs/maxstep50.yaml, 20 episode (prompt 저장 이전 코드)
    episodes.jsonl    # episode, task_id, task_type, success, steps  (episode 완료 표시)
    steps.jsonl       # step별 action, valid, ok, hid_idx, n_cand, p_chosen
                      # + (2026-10-15 이후 수집분) prompt, prompt_tokens, cands, cand_logp, cand_ntok
    hidden.npy        # (전체 step 수, 4 layer, hidden) float16. 25/50/75/100% 깊이. 종료 시 생성
    hidden_shards/    # episode별 ep#####.npy. 중간 저장/재개용 (2026-10-15 이후 수집분)
```

- `hid_idx`는 `hidden.npy`의 첫 번째 축 인덱스다. 최댓값이 행 수보다 작은지 확인했다.
- 원본 `data/decision`(결정용 실험 3B 100, 1.5B 100, 0.5B 20)은 이전 컴퓨터에만 있고 이 저장소 환경에는 없다.
- 예전 README는 `data/decision/<모델>/`을 설명했으나 해당 폴더는 이 환경에서 확인하지 못했다.
- ALFWorld 게임 데이터는 `alfworld-download`로 받고 `ALFWORLD_DATA`를 설정한다.

## 실행

```bash
pip install -r requirements.txt
export ALFWORLD_DATA=<게임 데이터 경로>
python collect/rollout.py --config configs/main_3b.yaml            # 새로 수집 (기존 결과가 있으면 거부)
python collect/rollout.py --config configs/main_3b.yaml --resume   # 중단된 수집을 이어서
python probes/run_probes.py data/decision_rerun/Qwen2.5-3B-Instruct --layer 1 --rounds 1 2 3 --pca 0 8 32 64
```

## probes/ 주의

- `lib.py`: 라운드 r에 살아 있는 episode당 1행, task(게임) 단위 GroupKFold, 표준화와 PCA는 Pipeline 안에서 fold별로 학습, 무작위 라벨 대조군, bootstrap CI.
- `--rounds`는 0-based step이다. 1이 두 번째 action이다.
- 알려진 문제(미수정): 셔플 대조군이 0.5가 아니라 0.33~0.41로 나온다(작은 표본의 cross-fitting 아티팩트로 보이나 원인 확인 필요),
  baseline `C=1.0`과 프로브 `C=0.01`이 달라 비교가 공정하지 않다, PCA k가 fold 표본 수를 넘으면 실패한다(n=100에서 k=128),
  전체 차원 대비 손실(explained variance)을 출력하지 않는다.
- `synth_check.py`는 합성 데이터 점검용이다. 연구 결과가 아니다.
- baseline 사다리는 1~3단만 있다. 4단(진행도), 5단(TF-IDF), teacher-forced 재추출 스크립트는 없다.
