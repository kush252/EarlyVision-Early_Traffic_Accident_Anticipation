import torch
import os
import numpy as np
import sys
import shutil
import uuid

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessor import image_resize_toTensor
from src.data.video_to_frames import video_to_frames_converter
from src.utils.lstm_utils import frames_to_sequence

def predict_risk(video_path, cnn_extractor, lstm_model, device):
    """
    Predict accident risk from a video using pre-loaded models.
    """
    # Create a unique temporary directory for frames to allow concurrent requests
    run_id = str(uuid.uuid4())
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Use a temp directory for frames
    temp_frame_dir = os.path.join(ROOT_DIR, "temp_frames", run_id)
    
    os.makedirs(temp_frame_dir, exist_ok=True)

    try:
        # Convert video to frames
        yield {"status": "Preprocessing video...", "progress": 5}
        # print(f"[{run_id}] Converting video to frames...") 
        video_to_frames_converter(video_path, temp_frame_dir)
        yield {"status": "Extracting features...", "progress": 15}

        # Extract features
        # print(f"[{run_id}] Extracting features...")
        feature_list = []
        frame_files = [f for f in os.listdir(temp_frame_dir) if f.endswith('.jpg')]
        
        # Sort frames numerically
        frame_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or -1))

        if not frame_files:
            yield {"error": "No frames could be extracted from the video"}
            return

        total_frames = len(frame_files)
        with torch.no_grad():
            for i, frame_file in enumerate(frame_files):
                img_tensor = image_resize_toTensor(temp_frame_dir, frame_file)
                img_tensor = img_tensor.unsqueeze(0).to(device)
            
                features = cnn_extractor(img_tensor)
                features_np = features.cpu().numpy().flatten()
                feature_list.append(features_np)
                
                # Update progress every 5 frames or so to reduce traffic
                if i % 5 == 0 or i == total_frames - 1:
                    # Scale progress from 15 to 85
                    current_progress = 15 + int((i / total_frames) * 70)
                    yield {"status": f"Analyzing frame {i+1}/{total_frames}...", "progress": current_progress}

        # process sequences
        yield {"status": "Generating sequences...", "progress": 90}
        if len(feature_list) >= 15:
            # Using parameters from main.py: win_len=10, hop_len=5
            sequences = frames_to_sequence(feature_list, 10, 5, frames_per_video=len(feature_list))
        else:
             # Fallback
             sequences = []

        prediction_list = []
        if len(sequences) > 0:
            # print(f"[{run_id}] predicting with LSTM...")
            yield {"status": "Predicting risks...", "progress": 95}
            sequences_np = np.array(sequences, dtype=np.float32)
            sequences_tensor = torch.tensor(sequences_np).to(device)
            
            with torch.no_grad():
                logits = lstm_model(sequences_tensor)
                probs = torch.sigmoid(logits)
                prediction_list = probs.cpu().numpy().flatten().tolist()
        
        yield {"status": "Complete", "progress": 100, "predictions": prediction_list}

    except Exception as e:
        print(f"Error processing video {run_id}: {e}")
        yield {"error": str(e)}
    finally:
        # Cleanup
        if os.path.exists(temp_frame_dir):
            shutil.rmtree(temp_frame_dir)
