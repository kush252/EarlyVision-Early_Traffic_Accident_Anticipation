import torch
import time
import json
import os
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.preprocessor import (
    CRASH_IMG_FOLDER,
    image_resize_toTensor,
    get_images_labelled,
    FrameDataset,
    read_crash_labels,
    get_image_filenames
)
from src.data.video_to_frames import video_to_frames_converter
from src.models.vision_models.cnn_extractor import CNNExtractor

video_pth = input("Enter video path: ")
output_pth = os.path.join(os.getcwd(), "sample data", "frames")
video_to_frames_converter(video_pth, output_pth)

print("Initializing Feature Extractor...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "accident_detection_weighted_cnn_model.pth")
if not os.path.exists(WEIGHTS_PATH):
    WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "accident_detection_cnn_model.pth")

extractor = CNNExtractor(WEIGHTS_PATH).to(device)
extractor.eval()

print(f"Extracting features from frames in: {output_pth}...")
feature_list = []
frame_files = sorted([f for f in os.listdir(output_pth) if f.endswith('.jpg')])

if not frame_files:
    print("No frames found to process!")
else:
    with torch.no_grad():
        for i, frame_file in enumerate(frame_files):
            img_tensor = image_resize_toTensor(output_pth, frame_file)
            img_tensor = img_tensor.unsqueeze(0).to(device)
        
            features = extractor(img_tensor)
            features_np = features.cpu().numpy().flatten()
            feature_list.append(features_np)
            
