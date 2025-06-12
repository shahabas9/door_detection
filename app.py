import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Set up page
st.set_page_config(layout="wide")
st.title("🚪 Door Detection System")
st.markdown("Upload a floor plan image to detect and count doors")

# Constants
NUM_CLASSES = 3  # background + 2 classes

# Model architecture definition
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

# Load model function with caching
@st.cache_resource
def load_model():
    # Create model architecture
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
    )
    
    # Load saved weights
    checkpoint = torch.load('./faster_rcnn_colab/checkpoints/best_model.pth', 
                          map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Detection function
def detect_doors(model, image, confidence_threshold=0.5):
    # Convert to tensor
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(next(model.parameters()).device)
    
    # Predict
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    
    # Filter predictions
    door_indices = (predictions['labels'] == 1) & (predictions['scores'] > confidence_threshold)
    door_boxes = predictions['boxes'][door_indices].cpu().numpy()
    door_scores = predictions['scores'][door_indices].cpu().numpy()
    
    return door_boxes, door_scores

# Visualization function
def visualize_detections(image, boxes, scores):
    vis_image = image.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_image, f"Door: {score:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return vis_image

# Main app
def main():
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        # show_confidence = st.checkbox("Show Confidence Scores", True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Floor Plan", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Load model
            model = load_model()
            
            # Process image
            image = np.array(Image.open(uploaded_file).convert("RGB"))
            
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # Detect doors
            door_boxes, door_scores = detect_doors(model, image, confidence_threshold)
            vis_image = visualize_detections(image, door_boxes, door_scores)
            
            with col2:
                st.subheader(f"Detected Doors: {len(door_boxes)}")
                st.image(vis_image, use_container_width=True)
                
                # Display results table
                if len(door_boxes) > 0:
                    st.markdown("### Detection Details")
                    results = []
                    for i, (box, score) in enumerate(zip(door_boxes, door_scores), 1):
                        results.append({
                            "Door #": i,
                            "Location": f"({int(box[0])}, {int(box[1])}) to ({int(box[2])}, {int(box[3])})",
                            "Confidence": f"{score:.2%}"
                        })
                    st.table(results)
                
                # Download button for results
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    cv2.imwrite(tmp.name, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    with open(tmp.name, "rb") as file:
                        st.download_button(
                            label="Download Annotated Image",
                            data=file,
                            file_name="door_detections.jpg",
                            mime="image/jpeg"
                        )
                    os.unlink(tmp.name)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()