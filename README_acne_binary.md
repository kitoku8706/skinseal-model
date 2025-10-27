# Acne 전용 이진 분류 모델 학습 가이드

이 문서는 여드름(Acne)만 판독하는 별도 모델(Non-Acne vs Acne)을 학습하는 방법을 안내합니다.

## 데이터 구조
- 학습 스크립트는 `skinseal-model/train` 디렉터리를 ImageFolder 형식으로 사용합니다.
- `train` 아래의 모든 클래스 폴더를 읽고, 폴더명이 `Acne`인 경우에만 양성(1), 나머지는 모두 음성(0)으로 매핑합니다.

예시 디렉터리:
```
skinseal-model/
  train/
    Acne/
    Eczema/
    Psoriasis/
    ...
```

## 빠른 시작 (Windows PowerShell)
다음 명령으로 학습을 시작하면, 최적 가중치가 `skinseal-pythonAI/models/best_acne_model.pth`로 저장됩니다.

```powershell
python .\skinseal-model\train_acne_binary.py -e 10 -b 32 -o .\skinseal-pythonAI\models\best_acne_model.pth
```

옵션:
- `-t, --train-dir` 학습 데이터 경로 (기본: `skinseal-model/train`)
- `-e, --epochs` 학습 epoch 수 (기본: 10)
- `-b, --batch-size` 배치 크기 (기본: 32)
- `--val-ratio` train에서 검증으로 분할할 비율 (기본: 0.15)
- `--pretrained` ImageNet 사전학습 가중치 사용 시 지정 (네트워크 제한 환경에서는 생략 권장)
- `-o, --out` 저장 경로 (기본: `skinseal-pythonAI/models/best_acne_model.pth`)

## 서버 연동
- Flask 앱(`skinseal-pythonAI/app.py`)이 `models/best_acne_model.pth`를 자동 로드하도록 설정되어 있습니다.
- 서버가 실행 중이라면, 모델 파일을 교체한 뒤 서버 재시작 또는 핫리로드 환경에서 자동 반영됩니다.

엔드포인트:
- 공용: `POST /api/diagnosis/acne_model`
- 별칭: `POST /api/predict-acne`

폼 데이터:
- `file` (이미지 파일), `userId` (필수)

응답 예시:
```json
[
  {"class": "Acne", "probability": "92.10%"},
  {"class": "Non-Acne", "probability": "7.90%"}
]
```

## 참고
- 이미지 입력 크기는 기본 224x224이며, 학습 및 추론 모두 동일한 정규화를 사용합니다.
- GPU가 있으면 자동 사용되며, 없으면 CPU로 동작합니다.
