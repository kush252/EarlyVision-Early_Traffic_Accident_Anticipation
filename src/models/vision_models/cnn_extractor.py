import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.models.vision_models.cnn_model import SimpleCNN


class CNNExtractor(nn.Module):
    """
    Feature extractor that uses the trained SimpleCNN backbone.
    Outputs 512-dimensional feature vectors (from the last conv block).
    """
    def __init__(self, weights_path=None):
        super(CNNExtractor, self).__init__()
        
        self.base_model = SimpleCNN(num_classes=2)
        
        if weights_path:
            if os.path.exists(weights_path):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Loading weights from {weights_path}...")
                
                state_dict = torch.load(weights_path, map_location=device)
                self.base_model.load_state_dict(state_dict)
                print("Weights loaded successfully.")
            else:
                print(f"Warning: Weights path {weights_path} not found. Using random initialization.")
        
        self.base_model.fc = nn.Sequential(
            nn.Flatten()
        )

    def forward(self, x):
        return self.base_model(x)


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "accident_detection_weighted_cnn_model.pth")
    
    if not os.path.exists(WEIGHTS_PATH):
         WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "accident_detection_cnn_model.pth")

    extractor = CNNExtractor(WEIGHTS_PATH)
    extractor.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    features = extractor(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output features shape: {features.shape}")
    print("Extraction successful.")
