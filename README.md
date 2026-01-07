제공해주신 [과제]표지판 찾기.ipynb 파일의 내용을 바탕으로 GitHub 리포지토리의 README.md 스타일로 정리해 드립니다.

🚸 Road Sign Detection with YOLOv8 (표지판 감지 프로젝트)
이 프로젝트는 YOLOv8 (You Only Look Once version 8) 모델을 사용하여 도로 표지판을 실시간으로 감지하고 분류하는 딥러닝 모델 학습 노트북입니다. Kaggle의 Road Sign Detection 데이터셋을 활용하여 데이터 전처리부터 모델 학습, 예측 시각화까지의 전 과정을 포함하고 있습니다.

📝 프로젝트 개요 (Overview)
목표: 도로 이미지에서 4가지 종류의 표지판(신호등, 정지, 속도제한, 횡단보도)을 식별합니다.

사용 모델: Ultralytics YOLOv8n (Nano) - 빠르고 가벼운 객체 탐지 모델

데이터셋: Kaggle Road Sign Detection Dataset

환경: Google Colab (GPU 가속 권장)

📂 데이터셋 구조 (Dataset)
이 프로젝트는 Pascal VOC 형식(XML)의 라벨을 YOLO 형식(TXT)으로 변환하여 사용합니다.

Classes (총 4개 클래스):

trafficlight (신호등)

stop (정지)

speedlimit (속도제한)

crosswalk (횡단보도)

데이터 분할: 전체 데이터의 80%를 학습(Train)용, 20%를 검증(Val)용으로 무작위 분할합니다.

🛠️ 설치 및 요구사항 (Prerequisites)
이 노트북을 실행하기 위해서는 다음 라이브러리들이 필요합니다.

Python

# 필수 라이브러리 설치
!pip install ultralytics
Python 3.x

Ultralytics (YOLOv8)

PyTorch

OpenCV

Matplotlib

Kaggle API (데이터셋 다운로드용)

🚀 사용 방법 (Usage)
1. 데이터셋 준비 (Data Preparation)
Kaggle API Token(kaggle.json)을 업로드하여 데이터셋을 다운로드하고 압축을 해제합니다.

Python

# Kaggle API 설정 및 데이터 다운로드
from google.colab import files
files.upload()  # kaggle.json 업로드
!kaggle datasets download -d andrewmvd/road-sign-detection
!unzip -q road-sign-detection.zip -d ./road_sign_dataset
2. 데이터 전처리 (Preprocessing)
XML 형식의 바운딩 박스 좌표를 YOLO 포맷(x_center, y_center, width, height)으로 변환하고 디렉토리 구조를 재배치합니다.

Plaintext

datasets/
  └── road_sign/
      ├── images/ (train/val)
      └── labels/ (train/val)
3. 학습 설정 (Configuration)
road_sign_data.yaml 파일을 생성하여 데이터 경로와 클래스 정보를 정의합니다.

4. 모델 학습 (Training)
YOLOv8n 모델을 로드하여 20 에폭(Epoch) 동안 학습을 진행합니다.

Python

from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='road_sign_data.yaml',
    epochs=20,
    imgsz=640,
    batch=16,
    name='road_sign_model'
)
5. 추론 및 시각화 (Inference)
검증 데이터셋 중 임의의 이미지를 선택하여 학습된 모델로 표지판을 탐지하고 결과를 시각화합니다.

Python

# 예측 실행 및 결과 이미지 출력
results = model.predict(source=test_image, conf=0.25, save=True)
📊 결과 (Results)
학습이 완료되면 runs/detect/road_sign_model 경로에 학습 결과(손실 그래프, 혼동 행렬 등)와 가중치 파일(best.pt)이 저장됩니다. 테스트 이미지를 통해 바운딩 박스와 클래스, 신뢰도(Confidence Score)가 표시된 결과를 확인할 수 있습니다.

작성자: [사용자 ID/이름] 라이선스: CC0-1.0 (데이터셋 라이선스 따름)
