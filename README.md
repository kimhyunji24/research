# 소형 LM 에이전트 실패 신호 스케일링

소형 언어모델(0.5B~3B)이 ALFWorld에서 에이전트로 일할 때, hidden state로 episode 실패를 예측할 수 있는지 본다.
설계와 진행 기록은 Notion 페이지 "연구 진행 현황: 소형 LM 에이전트 실패 신호 스케일링"과 그 하위 페이지 "설계 재검토"에 있다.

## 폴더

| 경로 | 내용 | git |
|---|---|---|
| `collect/rollout.py` | ALFWorld rollout + hidden state 저장 | 포함 |
| `configs/` | `alfworld_base.yaml`, `pilot.yaml`, `decision.yaml` | 포함 |
| `probes/` | 라운드별 AUROC 평가 코드 (아래 참고, **실제 데이터로 미검증**) | 포함 |
| `data/` | 수집 결과 (episodes/steps/hidden) | **제외 (.gitignore)** |
| `$ALFWORLD_DATA` | ALFWorld 게임 데이터 (약 2.3GB) | **제외** |

## 데이터 위치 (git에 없으므로 다른 컴퓨터로 직접 옮겨야 한다)

수집 결과는 `configs/*.yaml`의 `out_dir`/`<모델 이름>/` 아래에 생긴다.

```
data/decision/<Qwen2.5-0.5B|1.5B|3B-Instruct>/   # configs/decision.yaml (train 게임, 결정용 실험)
data/pilot/<모델>/                               # configs/pilot.yaml
    episodes.jsonl   # episode, task_id, task_type, success, steps
    steps.jsonl      # step별 action, p_chosen, n_cand, hid_idx 등
    hidden.npy       # (전체 step 수, 4 layer, hidden) float16. 25/50/75/100% 깊이
```

- 원본 위치는 결정용 실험을 돌린 **Windows PC의 WSL2 저장소 폴더 안 `data/`** 다. `.gitignore` 때문에 GitHub에는 올라가 있지 않다.
- 옮기는 방법 예: `rsync -av data/ <새 컴퓨터>:<저장소>/data/` 또는 `data/`를 압축해 외장 저장소/클라우드로 복사. (`*.zip`과 `*.npy`는 .gitignore 대상이라 git으로는 안 된다.)
- ALFWorld 게임 데이터는 새 컴퓨터에서 다시 받는다: `alfworld-download`로 받고 환경변수 `ALFWORLD_DATA`를 설정한다.
- 이 저장소를 만든 세션에서는 `data/`에 접근할 수 없었다. 위 파일 구성은 `collect/rollout.py` 코드 기준이며, 실제 폴더 이름과 파일 존재 여부는 옮기기 전에 직접 확인한다.

## 실행

```bash
pip install -r requirements.txt            # Mac 환경은 requirements_mac.txt
python collect/rollout.py --config configs/decision.yaml --model Qwen/Qwen2.5-3B-Instruct
python probes/run_probes.py data/decision/Qwen2.5-3B-Instruct --layer 1 --rounds 1 2 3
```

## probes/ 주의

- `lib.py`: 라운드 r에 살아 있는 episode당 1행으로 평가(길이 교락 차단), task 단위 GroupKFold, 표준화와 PCA는 fold 안에서만 학습, 무작위 라벨 대조군, bootstrap CI.
- `synth_check.py`는 합성 데이터로 코드와 PCA 가정을 점검하는 용도다. 연구 결과가 아니다.
- baseline 사다리는 1~3단만 구현했다. 4단(진행도), 5단(TF-IDF)은 아직 없다.
- 후보 점수화 정책(`policy: score`)은 계획서 IV-3에 아직 반영되지 않았다.
