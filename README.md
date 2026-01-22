# EarlyVision: Early Traffic Accident Anticipation

This project is a video-based traffic accident anticipation system designed to predict traffic accidents from dashcam footage before they occur. It leverages a two-stage deep learning pipeline combining spatial feature extraction (CNN) and temporal reasoning (LSTM) to forecast accident risks in diverse weather and road conditions.

## Project Overview

Traffic accident anticipation is crucial for autonomous driving safety systems. EarlyVision analyzes video sequences to calculate a frame-by-frame risk probability. The system processes video input, validates the scene content to ensure relevance (e.g., presence of roads/vehicles), and then applies a trained AI model to generate a risk score. If the risk exceeds a safety threshold, the system flags the potential accident.

To ensure robustness, the solution is designed with a decoupled architecture: a backend API serving the PyTorch models and a frontend interface for user interaction.

## Architecture & Methodology

The system follows a modular architecture consisting of the following key components:

### 1. Deep Learning Pipeline (Spatial-Temporal)
The core intelligence is split into two specialized stages:
- **Spatial Feature Extractor (CNN)**:
  - Uses a **Custom CNN Architecture** (`SimpleCNN`) designed for accident detection.
  - Trained from scratch on crash/non-crash datasets to learn accident-relevant visual cues.
  - Extracts a flattened 512-dimensional feature vector from each frame.
- **Temporal Sequence Modeler (LSTM)**:
  - Taking the sequence of 512-dim features from the CNN, the **Long Short-Term Memory (LSTM)** network analyzes the progression of events over time.
  - It maintains a memory of past frames to detect developing patterns that precede a crash.
  - Outputs a risk score (converted to probability) for the video sequence.

### 2. Application Interface
- **Backend**: A **FastAPI** server that exposes the inference pipeline. It manages video processing, model loading, and streaming of real-time progress updates.
- **Frontend**: A **Streamlit** dashboard providing a user-friendly interface. It allows users to upload videos or select samples, visualizing the accident risk curve synchronized with video playback.

### 3. Scene Validation Module
Before any risk analysis, the system first verifies that the input video contains valid driving footage.
- **Tools**: Pre-trained MobileNetV2 (ImageNet weights).
- **Function**: Samples frames from the video and checks for the presence of traffic-related classes (e.g., vehicles, roads, traffic signs) using a pre-trained classifier. If the footage/scene is irrelevant, it is rejected to ensure reliable predictions.

## Directory Structure

```
root/
├── backend/
│   ├── main.py                # FastAPI app entry point (endpoints)
│   └── inference.py           # Core logic connecting models & video processing
├── frontend/
│   └── app.py                 # Streamlit dashboard application
├── src/
│   ├── models/
│   │   ├── vision_models/     # CNN architectures (e.g., MobileNetV2 extractor)
│   │   └── temporal_model/    # LSTM architectures
│   ├── data/                  # Data loaders and preprocessing scripts
│   └── utils/
│       ├── scene_validator.py # Logic for checking video content relevance
│       └── lstm_utils.py      # Helpers for sequence generation
├── pipelines/
│   ├── cnn_modeltrain_pipeline.py          # Script for training CNN
│   ├── cnn_weighted_modeltrain_pipeline.py # Training with weighted loss
│   └── lstm_training_pipeline.py           # Script for training LSTM
├── models/                    # Saved PyTorch model weights (.pth files)
└── requirements.txt           # Python dependencies
```

## Setup and Usage

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA support recommended for performance)

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
The project requires both the backend API and the frontend dashboard to be running simultaneously.

1. **Start the Backend Server**:
   Open a terminal and run the FastAPI server.
   ```bash
   python backend/main.py
   ```
   The API will start at `http://localhost:8000`.

2. **Start the Frontend Dashboard**:
   Open a new terminal window and launch the Streamlit app.
   ```bash
   streamlit run frontend/app.py
   ```
   The dashboard will open in your default web browser (usually at `http://localhost:8501`).

### How to Use
1. Open the Streamlit dashboard in your browser.
2. Use the sidebar to choose:
   - **Upload Video**: Upload your own dashcam .mp4 file.
   - **Use Sample**: Select from pre-loaded test scenarios.
3. Click **"Analyze Risk"**.
4. Observe the real-time progress bar. Once complete, view the results:
   - **Video Player**: Shows the analyzed footage.
   - **Risk Graph**: Displays the fluctuating accident probability over time, highlighting dangerous moments in red.

## Author
**Kush**  

