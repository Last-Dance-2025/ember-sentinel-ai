import os
import shutil
import yaml
from tqdm import tqdm # ⭐️ tqdm 라이브러리 임포트

# --- 1. 경로 설정 (사용자 지정) ---
# 데이터셋의 루트 디렉토리
DATASET_ROOT = 'FASDD_CV'

# 원본 파일 경로
ORIGINAL_IMAGES_DIR = os.path.join(DATASET_ROOT, 'images')
ORIGINAL_LABELS_DIR = os.path.join(DATASET_ROOT, 'annotations', 'YOLO_CV', 'labels')
ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, 'annotations', 'YOLO_CV')

# 분할 목록 파일 (train.txt, val.txt, test.txt)
SPLIT_FILES = {
    'train': os.path.join(ANNOTATIONS_DIR, 'train.txt'),
    'val': os.path.join(ANNOTATIONS_DIR, 'val.txt'),
    'test': os.path.join(ANNOTATIONS_DIR, 'test.txt'),
}

# 최종 Ultralytics 형식의 데이터셋이 저장될 디렉토리 (DATASET_ROOT 내부에 생성)
OUTPUT_BASE_DIR = os.path.join(DATASET_ROOT, 'yolo_format')
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, 'images')
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_BASE_DIR, 'labels')

# 2. data.yaml 파일 생성 경로 및 내용
YAML_OUTPUT_PATH = os.path.join(DATASET_ROOT, 'fasdd_data.yaml')

# 새로운 데이터셋에 맞춰 수정된 YAML 내용 (기존 Roboflow 형식을 기반으로 함)
NEW_YAML_CONTENT = {
    'path': os.path.join(os.getcwd(), DATASET_ROOT, 'yolo_format'), 
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    
    # ⚠️ 클래스 정보 확인 필요
    'nc': 2,
    'names': ['fire', 'smoke'],
}
# --- 설정 끝 ---


def get_file_list_from_txt(txt_path):
    """txt 파일에서 파일 경로를 읽어 파일 이름만 추출합니다."""
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        file_names = []
        for line in lines:
            line = line.strip()
            if line:
                base_name = os.path.basename(line).split('.')[0]
                file_names.append(base_name)
        return file_names
    except FileNotFoundError:
        print(f"오류: 목록 파일이 없습니다. 경로를 확인하세요: {txt_path}")
        return []

def organize_dataset():
    """파일 목록에 따라 이미지와 레이블 파일을 train/val/test 폴더로 복사합니다."""
    
    print(f"--- 데이터셋 재구성 시작: {DATASET_ROOT} ---")
    
    # 출력 폴더 구조 생성
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_IMAGES_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_LABELS_DIR, split), exist_ok=True)
        print(f"디렉토리 생성 완료: {split}")

    total_files = 0
    
    for split_name, txt_path in SPLIT_FILES.items():
        
        file_names = get_file_list_from_txt(txt_path)
        
        if not file_names:
            continue
            
        count = 0
        
        # ⭐️ tqdm 적용: file_names 리스트를 tqdm으로 감싸 진행률 표시
        print(f"\n[{split_name.upper()} 분할] 총 {len(file_names)}개 파일 복사 중...")
        
        for base_name in tqdm(file_names, desc=f"{split_name} 복사"):
            
            img_found = False
            for ext in ['.jpg', '.png', '.jpeg']: # 흔한 확장자 목록
                src_img_path = os.path.join(ORIGINAL_IMAGES_DIR, base_name + ext)
                
                if os.path.exists(src_img_path):
                    src_label_path = os.path.join(ORIGINAL_LABELS_DIR, base_name + '.txt')
                    
                    if os.path.exists(src_label_path):
                        dst_img_path = os.path.join(OUTPUT_IMAGES_DIR, split_name, base_name + ext)
                        dst_label_path = os.path.join(OUTPUT_LABELS_DIR, split_name, base_name + '.txt')
                        
                        shutil.copy2(src_img_path, dst_img_path)
                        shutil.copy2(src_label_path, dst_label_path)
                        count += 1
                        img_found = True
                        break # 이미지 찾았으니 다음 파일로 이동

            if not img_found:
                 # 해당 파일이 존재하지 않는 경우 (txt 목록에는 있으나 파일이 없는 경우)
                 # tqdm 내에서는 print 대신 logging을 사용해야 하지만, 간단한 스크립트에서는 무시하거나 주석 처리합니다.
                 pass

        print(f"성공적으로 {split_name} 폴더로 복사된 파일 수: {count}쌍")
        total_files += count
        
    print(f"\n--- 재구성 완료. 총 복사된 파일: {total_files}쌍 ---")


def create_yaml_file():
    """Ultralytics 학습을 위한 data.yaml 파일을 생성합니다."""
    
    print("\n--- data.yaml 파일 생성 시작 ---")
    
    # YAML 파일 생성
    with open(YAML_OUTPUT_PATH, 'w') as f:
        yaml.dump(NEW_YAML_CONTENT, f, sort_keys=False)
        
    print(f"✅ data.yaml 파일 생성 완료: {YAML_OUTPUT_PATH}")
    print("⚠️ 'nc'와 'names'가 FASDD_CV 데이터셋의 실제 클래스 정보와 일치하는지 반드시 확인하세요.")

# --- 메인 실행 ---
if __name__ == "__main__":
    
    # 1단계: 데이터셋 구조 변경 및 파일 이동
    organize_dataset()
    
    # 2단계: data.yaml 파일 생성
    create_yaml_file()

    # 3단계: 다음 학습 명령어 안내
    print("\n\n--- 다음 학습 명령어 ---")
    print("python train.py \\")
    print(f"    data=./{os.path.basename(DATASET_ROOT)}/fasdd_data.yaml \\")
    print("    epochs=... \\")
    print("    ...")