from ultralytics import YOLO

# YOLOv8 모델 로드 (필요에 따라 "yolov8n", "yolov8s", "yolov8m" 등의 다양한 크기가 있음)
model = YOLO("yolov8n.pt")  # 사전 학습된 YOLOv8 모델 사용

# 모델 훈련
model.train(data="data.yaml", epochs=50, imgsz=640)
