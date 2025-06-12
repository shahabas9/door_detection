import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import numpy as np
CLASS_NAMES = ['background', 'door', 'str']
def predict(image_path, threshold=0.5):
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(DEVICE)  # Send to GPU

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)

    # Convert to OpenCV format (BGR)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Process detections
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            cls_name = CLASS_NAMES[label]
            color = (0, 255, 0)  # Green
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_cv, f'{cls_name} {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display in Colab (no cv2.imshow needed)
    from google.colab.patches import cv2_imshow
    cv2_imshow(img_cv)

# After training, test on the same 3 images
model.eval()
for img_name in train_dataset.imgs:
    img_path = os.path.join(IMG_DIR, img_name)
    predict(img_path, threshold=0.9)  # Use your existing predict function