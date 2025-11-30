# 사용법
## 프로젝트 설정
```
git clone https://github.com/Last-Dance-2025/ember-sentinel-ai.git
cd ember-sentinel-ai
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```
## 데이터셋 다운로드
- 데이터셋(FASDD_CV.zip) 다운로드 후 ember-sentinel-ai/FASDD_CV.zip 경로에 위치시킬 것 </br>
(링크: https://drive.google.com/file/d/1TdJvs7Q0_ylIxBQvLI_U8plnoSItS-SX/view?usp=sharing)
```
unzip ./FASDD_CV.zip
```

## 데이터 전처리
```
python preprocessing.py
```

## 모델 다운로드
- 모델(yolov11n.zip) 다운로드 후 ember-sentinel-ai/experiments/yolov11n.zip 경로에 위치시킬 것 </br>
(링크: https://drive.google.com/file/d/1RBWNbcXIIVwbMlQNk9-QOH2hBbzR9vog/view?usp=sharing)

```
unzip ./experiments/yolov11n.zip
```

## 추론
```
python inference.py
```
