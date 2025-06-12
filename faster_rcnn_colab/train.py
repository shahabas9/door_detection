import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import cv2
import numpy as np


from google.colab import drive
drive.mount('/content/drive')


import zipfile

# Path to your zip file in Google Drive
zip_path = '/content/drive/MyDrive/dataset.zip'  # Update with your path

# Extract to Colab's temporary storage (faster access)
#unzip -q "{zip_path}" -d "/content/dataset"

# === Configuration ===
DATA_DIR = '/content/dataset'
CHECKPOINT_DIR = '/content/drive/MyDrive/checkpoints'
IMG_DIR = os.path.join(DATA_DIR, 'images')
ANN_DIR = os.path.join(DATA_DIR, 'annotations')
NUM_CLASSES = 3  # background + 2 classes
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Dataset Class ===
class VOCDataset(Dataset):
    def __init__(self, img_dir, ann_dir):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        ann_path = os.path.join(self.ann_dir, self.imgs[idx].replace('.jpg', '.xml').replace('.png', '.xml'))
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Load annotations
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            box = [
                float(bndbox.find('xmin').text),
                float(bndbox.find('ymin').text),
                float(bndbox.find('xmax').text),
                float(bndbox.find('ymax').text)
            ]
            boxes.append(box)
            labels.append(1 if obj.find('name').text == 'door' else 2)
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': (torch.tensor(boxes)[:, 2] - torch.tensor(boxes)[:, 0]) * 
                    (torch.tensor(boxes)[:, 3] - torch.tensor(boxes)[:, 1]),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }
        return img, target

# Initialize dataset
train_dataset = VOCDataset(IMG_DIR, ANN_DIR)

# === Custom Model ===
class OverfitBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.out_channels = 32
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return {'0': x}

model = FasterRCNN(
    backbone=OverfitBackbone(),
    num_classes=NUM_CLASSES,
    rpn_anchor_generator=AnchorGenerator(
        sizes=((8, 16, 32),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    ),
    box_roi_pool=torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    ),
    min_size=512,
    max_size=512
).to(DEVICE)

# === Training ===
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

for epoch in range(300):
    model.train()
    total_loss = 0
    
    # Train on all 3 images
    for i in range(3):
        img, target = train_dataset[i]
        img = img.unsqueeze(0).to(DEVICE)
        target = [{k: v.to(DEVICE) for k, v in target.items()}]
        
        optimizer.zero_grad()
        loss_dict = model(img, target)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/3:.6f}")
        if total_loss/3 < 0.0001:
            break
# Save final model (both formats)
final_save_path = os.path.join(CHECKPOINT_DIR, 'final_model.pth')
best_save_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')

# Option 1: Save entire model
torch.save(model, final_save_path)

# Option 2: Save state dict (recommended)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': total_loss/3,
    'class_names': ['background', 'door', 'str']  # Save class info
}, best_save_path)

print(f"Models saved to:\n{final_save_path}\n{best_save_path}")
# === Evaluation ===
model.eval()
with torch.no_grad():
    for i in range(3):
        img, _ = train_dataset[i]
        prediction = model([img.to(DEVICE)])[0]
        print(f"\nImage {i+1} Predictions:")
        print("Boxes:", prediction['boxes'].cpu().numpy().round(2))
        print("Labels:", prediction['labels'].cpu().numpy())
        print("Scores:", prediction['scores'].cpu().numpy().round(4))