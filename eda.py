import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. 설정 변수 ---
# ⚠️ 데이터셋의 루트 경로를 확인해주세요.
DATASET_ROOT = 'FASDD_CV/yolo_format'
LABELS_DIR = os.path.join(DATASET_ROOT, 'labels')
IMAGE_DIR = os.path.join(DATASET_ROOT, 'images')

# data.yaml의 클래스 이름과 순서에 맞춥니다.
CLASS_NAMES = ['fire', 'smoke']
NUM_CLASSES = len(CLASS_NAMES)
# --------------------

def collect_all_files():
    """모든 레이블 파일 경로와 전체 이미지 수를 수집합니다."""
    
    # ⚠️ test 분할 제외
    splits_to_analyze = ['train', 'val']
    
    all_label_paths = []
    all_image_count = 0
    
    print("--- 파일 목록 수집 중 ---")
    
    for split_dir in splits_to_analyze:
        labels_path = os.path.join(LABELS_DIR, split_dir)
        images_path = os.path.join(IMAGE_DIR, split_dir)
        
        # 레이블 파일 수집
        if os.path.exists(labels_path):
            label_files = [os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.txt')]
            all_label_paths.extend(label_files)
        
        # 이미지 파일 수집 (총 이미지 수 계산)
        if os.path.exists(images_path):
            all_image_count += len([f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
    return all_label_paths, all_image_count

def analyze_yolo_labels_combined(all_label_paths, total_image_count):
    """
    수집된 모든 레이블 파일들을 분석하여 통합된 정보를 추출합니다.
    """
    
    all_class_counts = np.zeros(NUM_CLASSES, dtype=int)
    all_box_widths = []
    all_box_heights = []
    all_box_areas = []
    all_aspect_ratios = []
    
    total_annotations = 0
    total_labeled_images = 0
    
    # tqdm을 사용하여 전체 레이블 파일 처리 진행률 표시
    for file_path in tqdm(all_label_paths, desc=f"레이블 분석 및 데이터 추출 중"):
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                continue
                
            total_labeled_images += 1
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    # YOLO 포맷: [class_id, center_x, center_y, width, height] (모두 정규화된 값 0~1)
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    if 0 <= class_id < NUM_CLASSES:
                        all_class_counts[class_id] += 1
                        total_annotations += 1
                        
                        all_box_widths.append(width)
                        all_box_heights.append(height)
                        all_box_areas.append(width * height)
                        # 0으로 나누는 오류 방지를 위해 작은 값 추가
                        all_aspect_ratios.append(width / (height + 1e-6)) 
                        
        except Exception as e:
            # print(f"경고: {file_path} 파일 처리 중 오류 발생: {e}")
            pass
            
    return {
        'total_image_count': total_image_count,
        'total_annotations': total_annotations,
        'labeled_images': total_labeled_images,
        'class_counts': all_class_counts,
        'widths': all_box_widths,
        'heights': all_box_heights,
        'areas': all_box_areas,
        'aspect_ratios': all_aspect_ratios
    }

def print_combined_summary(analysis_data):
    """통합 분석 결과를 요약하여 출력합니다."""
    
    labeled_images = analysis_data['labeled_images']
    image_count = analysis_data['total_image_count']
    
    print("\n" + "=" * 60)
    print("## 📋 Training Dataset Summary")
    print("=" * 60)
    print(f"Total Images: {image_count}")
    print(f"Labeled Images: {labeled_images} ({labeled_images / image_count * 100:.2f}%)")
    print(f"Total Annotations: {analysis_data['total_annotations']}")

    # 클래스 분포 테이블
    df_class = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Count': analysis_data['class_counts'],
        'Percentage': analysis_data['class_counts'] / analysis_data['total_annotations'] * 100
    })
    print("\n### Class Distribution:")
    print(df_class.to_markdown(index=False, floatfmt=".2f"))

# ⭐️ plot_distribution 함수 수정: 제목/레이블이 모두 영어로 변경됨
def plot_distribution(data, title, xlabel, ylabel, bins=50, log_scale=False, filename=None):
    """히스토그램을 그려 파일로 저장합니다."""
    
    plt.figure(figsize=(10, 6))
    
    # 종횡비 데이터 클리핑
    if 'Aspect Ratio' in title:
        data = np.clip(data, 0, 10) 
        bins = 100
        
    plt.hist(data, bins=bins, log=log_scale, color='indianred', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) # ⭐️ ylabel도 동적으로 설정
    plt.grid(axis='y', alpha=0.75)
    
    # 그림 저장
    plot_dir = 'eda_results'
    os.makedirs(plot_dir, exist_ok=True)
    if filename:
        plt.savefig(os.path.join(plot_dir, filename))
        print(f"✅ Plot saved: {os.path.join(plot_dir, filename)}")
    plt.close()

def perform_eda_combined():
    """메인 통합 EDA 실행 함수."""
    
    print(f"Dataset analysis path: {DATASET_ROOT}")
    
    # 1. 모든 레이블 파일 경로와 총 이미지 수 수집
    all_label_paths, total_image_count = collect_all_files()
    
    if not all_label_paths:
        print("Error: No label files found in train or val splits.")
        return

    # 2. 통합 분석 실행
    analysis_data = analyze_yolo_labels_combined(all_label_paths, total_image_count)
    
    # 3. 요약 출력
    print_combined_summary(analysis_data)
        
    print("\n" + "=" * 50)
    print("## 📈 Training Data Distribution Analysis (Plots)")
    print("=" * 50)

    # 4. 시각화 분석 및 파일 저장 (모두 영문으로 변경)
    
    # A. 객체 크기 분포 (정규화된 Area)
    plot_distribution(
        analysis_data['areas'], 
        "Object Area Distribution (Normalized, Log Scale)", 
        "Normalized Area (W * H)", 
        "Frequency (Log Scale)", # ⭐️ Y축 레이블 변경
        bins=100, 
        log_scale=True,
        filename="1_area_distribution.png"
    )
    
    # B. 종횡비 분포 (Aspect Ratio)
    plot_distribution(
        analysis_data['aspect_ratios'], 
        "Object Aspect Ratio Distribution", 
        "Aspect Ratio (W / H)", 
        "Frequency",
        bins=100,
        filename="2_aspect_ratio_distribution.png"
    )

    # C. 객체 Width/Height 분포
    plot_distribution(
        analysis_data['widths'], 
        "Normalized Object Width Distribution", 
        "Normalized Width", 
        "Frequency",
        bins=50,
        filename="3_normalized_width_distribution.png"
    )
    plot_distribution(
        analysis_data['heights'], 
        "Normalized Object Height Distribution", 
        "Normalized Height", 
        "Frequency",
        bins=50,
        filename="4_normalized_height_distribution.png"
    )
    
    print("\nAnalysis complete. Check the 'eda_results' folder for plots.")
    print("--- EDA.py finished ---")


if __name__ == "__main__":
    perform_eda_combined()