from ultralytics import YOLO

# Load a model
model = YOLO("./experiments/yolov11n/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="ncnn",
    device=0,
    half=True)