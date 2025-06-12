from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Start from pretrained weights

model.train(
    data='data.yaml',
    epochs=50,              # Fewer epochs to avoid overfitting
    imgsz=1024,
    batch=2,
    augment=True,
    hsv_h=0.015,           # Hue
    hsv_s=0.7,             # Saturation
    hsv_v=0.4,             # Brightness
    flipud=0.5,            # Vertical flip
    fliplr=0.5,            # Horizontal flip
    optimizer='Adam',      # Adam may help tiny datasets
    lr0=0.001,             # Lower LR for Adam
    freeze=20,             # Freeze more layers
    device='cpu',
    verbose=True
)