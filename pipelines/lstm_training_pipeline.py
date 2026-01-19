import torch
import os
import sys
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessor import image_resize_toTensor
from src.data.preprocessor import read_crash_labels
from src.models.vision_models.cnn_extractor import CNNExtractor
from src.utils.lstm_utils import frames_to_sequence,framelabel_to_sequencelabel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "accident_detection_weighted_cnn_model.pth")

output_pth = os.path.join(PROJECT_ROOT, "dataset", "frames","Crash-1500")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L=10
K=5

crash_labels_path=os.path.join(PROJECT_ROOT, "dataset", "videos", "Crash-1500.txt")
print(f"Extracting features from frames in: {output_pth}...")
print(f"Using device: {device}")

extractor = CNNExtractor(WEIGHTS_PATH).to(device)
extractor.eval()

feature_list = []
frame_files = sorted([f for f in os.listdir(output_pth) if f.endswith('.jpg')])
total_frames = len(frame_files)

if not frame_files:
    print("No frames found to process!")
else:
    print(f"Total frames to process: {total_frames}")
    start_time = time.time()
    
    with torch.no_grad():
        for i, frame_file in enumerate(frame_files):
            img_tensor = image_resize_toTensor(output_pth, frame_file)
            img_tensor = img_tensor.unsqueeze(0).to(device)
        
            features = extractor(img_tensor)
            features_np = features.cpu().numpy().flatten()
            feature_list.append(features_np)
            if i == 999:
                break
            # Progress logging every 1000 frames
            if (i + 1) % 1000 == 0 or (i + 1) == total_frames:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                remaining = (total_frames - i - 1) / speed if speed > 0 else 0
                print(f"Progress: {i + 1}/{total_frames} ({100*(i+1)/total_frames:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | Speed: {speed:.1f} frames/s | ETA: {remaining:.1f}s")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nFeature extraction complete!")
    print(f"Total features extracted: {len(feature_list)}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    
    # Convert to numpy array and show dimensions/size
    features_array = np.array(feature_list, dtype=np.float32)
    print(f"\nFeature Array Info:")
    print(f"  Shape: {features_array.shape}")
    print(f"  Dtype: {features_array.dtype}")
   


sequences=frame_to_sequence(feature_list,L,K)
labels=read_crash_labels(crash_labels_path)
y=framelabel_to_sequencelabel(labels,L,K)


print(len(sequences))
print(len(y))

print("labels:",labels)
print("y:",y)