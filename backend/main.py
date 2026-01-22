from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import os
import sys
import shutil
import uuid
from contextlib import asynccontextmanager

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vision_models.cnn_extractor import CNNExtractor
from src.models.temporal_model.LSTM_model import RiskLSTM
from src.utils.scene_validator import DashcamValidator
from backend.inference import predict_risk

# Global variables to hold models
models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("Loading CNN Extractor...")
    cnn_weights = os.path.join(PROJECT_ROOT, "models", "accident_detection_weighted_cnn_model.pth")
    if not os.path.exists(cnn_weights):
        # Fallback as per main.py
        cnn_weights = os.path.join(PROJECT_ROOT, "models", "accident_detection_cnn_model.pth")
    
    if os.path.exists(cnn_weights):
        models["cnn"] = CNNExtractor(cnn_weights).to(device)
        models["cnn"].eval()
    else:
        print(f"WARNING: CNN weights not found at {cnn_weights}")
        # Depending on requirements, might want to raise error or initialize without weights
        models["cnn"] = CNNExtractor(None).to(device) # Assuming it handles None or we might need to fix this if CNNExtractor requires path
        models["cnn"].eval()

    print("Loading LSTM Model...")
    lstm_weights = os.path.join(PROJECT_ROOT, "models", "risk_lstm_model.pth")
    models["lstm"] = RiskLSTM(
        feature_dim=512,
        hidden_dim=256, 
        num_layers=2, 
        bidirectional=False
    ).to(device)
    
    if os.path.exists(lstm_weights):
        models["lstm"].load_state_dict(torch.load(lstm_weights, map_location=device))
        models["lstm"].eval()
    else:
        print(f"WARNING: LSTM weights not found at {lstm_weights}")
        # Proceed with random weights (or handle error)
        models["lstm"].eval()
        
    print("Loading Scene Validator...")
    models["validator"] = DashcamValidator()
    
    yield
    
    # Shutdown: Clean up resources if needed
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    # Save uploaded file to a temp location
    run_id = str(uuid.uuid4())
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(PROJECT_ROOT, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_video_path = os.path.join(temp_dir, f"{run_id}_{file.filename}")
    
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run inference
        # Run inference
        if "cnn" not in models or "lstm" not in models or "validator" not in models:
            raise HTTPException(status_code=500, detail="Models not loaded")
            
        # 1. Validate Scene Context
        is_dashcam, message = models["validator"].is_dashcam_footage(temp_video_path)
        if not is_dashcam:
             return {"error": f"Validation Failed: {message}"}
             
        # 2. Predict Risk
        result = predict_risk(temp_video_path, models["cnn"], models["lstm"], device)
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
             
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up video file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
