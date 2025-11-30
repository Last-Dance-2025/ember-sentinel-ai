from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Load a pretrained model

# 주요 하이퍼파라미터 설정 추가
results = model.train(
    data="./FASDD_CV/fasdd_data.yaml",
    workers=2,
    epochs=30, 
    imgsz=640,
    patience=5,
    cache=True,
    # cos_lr=True,
    amp=True,
    compile=True,
    # dropout=0.1,
    batch=32,             # 배치 크기: 32로 늘려서 학습 속도 향상 시도 (GPU 메모리 허용 시)
    project='experiments', # 프로젝트 폴더 이름 지정
    name='yolov11s_30',      # 실험 이름 지정
    exist_ok=True,
)