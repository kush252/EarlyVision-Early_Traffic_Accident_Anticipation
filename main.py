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
from src.utils.lstm_utils import frames_to_sequence
from src.models.temporal_model.LSTM_model import RiskLSTM


def get_accident_risk_prediction(video_pth):
    # Define Root and Output Path consistently
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    output_pth = os.path.join(ROOT_DIR, "sample data", "frames")

    # Cleanup previous frames to avoid mixing videos
    if os.path.exists(output_pth):
        for f in os.listdir(output_pth):
            fp = os.path.join(output_pth, f)
            if os.path.isfile(fp):
                os.remove(fp)

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
    frame_files = [f for f in os.listdir(output_pth) if f.endswith('.jpg')]

    frame_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or -1))

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
                
    if len(feature_list) >= 15:
        # Set frames_per_video to current length to treat it as one continuous sequence
        sequences = frames_to_sequence(feature_list, 10, 5, frames_per_video=len(feature_list))
    else:
        print("Video too short for sequence generation (needs >14 frames)")
        sequences = []

    print("Initializing LSTM Model...")
    LSTM_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "risk_lstm_model.pth")
        
    lstm_model = RiskLSTM(
        feature_dim=512,
        hidden_dim=256, 
        num_layers=2, 
        bidirectional=False
    ).to(device)

    if os.path.exists(LSTM_WEIGHTS_PATH):
        print(f"Loading LSTM weights from {LSTM_WEIGHTS_PATH}...")
        lstm_model.load_state_dict(torch.load(LSTM_WEIGHTS_PATH, map_location=device))
    else:
        print(f"Note: {LSTM_WEIGHTS_PATH} not found. Using random weights as placeholder.")

    lstm_model.eval()

    print("Predicting accident risk...")
    prediction_list = []

    if len(sequences) > 0:
        sequences_np = np.array(sequences, dtype=np.float32)
        sequences_tensor = torch.tensor(sequences_np).to(device)
        
        with torch.no_grad():
            logits = lstm_model(sequences_tensor)
            probs = torch.sigmoid(logits)
            prediction_list = probs.cpu().numpy().flatten().tolist()

    return prediction_list

if __name__ == "__main__":
    video_pth = input("Enter video path: ")
    predictions = get_accident_risk_prediction(video_pth)
    print(f"Predictions ({len(predictions)}):", predictions)

