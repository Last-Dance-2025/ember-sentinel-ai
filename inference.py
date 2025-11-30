from ultralytics import YOLO

# --- 설정 변수 ---
# 1. 학습 완료된 모델의 경로 (best.pt)
MODEL_PATH = "./experiments/yolov11n/weights/best_ncnn_model"

# 2. 데이터셋 정보가 담긴 YAML 파일 경로
DATA_YAML_PATH = "./FASDD_CV/fasdd_data.yaml"

# 3. 추론 컴파일 플래그 (GPU 환경에서 속도 향상)
COMPILE_FLAG = True 
# ------------------

# 1. 모델 로드
try:
    model = YOLO(MODEL_PATH)
except FileNotFoundError:
    print(f"오류: 모델 파일이 존재하지 않습니다. 경로를 확인하세요: {MODEL_PATH}")
    exit()

print("--- Test 폴더 전체 추론 및 성능 지표 계산 시작 ---")

# 2. model.val() 함수를 사용하여 성능 계산
# model.val() 함수는 검증/테스트 데이터셋에 대해 추론을 수행하고 mAP 등의 지표를 계산합니다.
results = model.val(
    data=DATA_YAML_PATH,  # 데이터셋 정보가 담긴 YAML 파일 경로
    split='test',         # 'test' 세트에 대해 실행하도록 명시
    imgsz=640,            # 추론 이미지 크기 (학습 시와 동일하게 설정)
    conf=0.1,           # confidence threshold (매우 낮게 설정하여 모든 감지 결과 포함)
    iou=0.5,              # NMS(Non-Maximum Suppression) IoU 임계값
    # compile=COMPILE_FLAG, # GPU 환경 최적화
    # save_json=True,       # JSON 파일로 결과 저장 (선택 사항)
    # save_txt=True,        # 예측된 바운딩 박스를 TXT 파일로 저장 (선택 사항)
)