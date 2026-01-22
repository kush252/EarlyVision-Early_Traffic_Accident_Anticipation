import torch
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class DashcamValidator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Scene Validator on {self.device}...")
        
        # Load a lightweight pre-trained model (MobileNetV2 is fast and good enough)
        # We use the default weights (IMAGENET1K_V1)
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(self.device)
        self.model.eval()
        
        # Define standard transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Set of ImageNet class indices relevant to roads/traffic
        # These correspond to various vehicles, road structures, etc.
        self.valid_indices = {
            407, # ambulance
            436, # beach wagon
            444, # bicycle
            468, # cab
            511, # convertible
            569, # garbage truck
            609, # jeep
            627, # limousine
            654, # minibus
            656, # minivan
            661, # Model T
            665, # moped
            670, # motor scooter
            671, # mountain bike
            675, # moving van
            705, # passenger car
            717, # pickup
            734, # police van
            751, # racer
            757, # recreational vehicle
            779, # school bus
            817, # sports car
            829, # streetcar
            864, # tow truck
            867, # trailer truck
            913, # traffic light
            914, # street sign
            919, # street
        }

    def is_dashcam_footage(self, video_path, num_checks=3):
        """
        Checks a few frames from the video to see if they contain road/traffic elements.
        Returns (is_valid: bool, confidence: str)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < 10:
            cap.release()
            return False, "Video too short"

        frame_points = [
            int(total_frames * 0.2), # 20% mark
            int(total_frames * 0.5), # 50% mark
            int(total_frames * 0.8)  # 80% mark
        ]
        
        valid_frames = 0
        
        try:
            for idx in frame_points:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Preprocess
                input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    
                # Get top 5 predictions for this frame
                probs, indices = torch.topk(output, 5)
                indices = indices.cpu().numpy()[0]
                
                # Check if any top prediction is in our valid set
                # We relax the condition: if ANY of the top 5 is a vehicle/road item, we count it.
                if any(idx in self.valid_indices for idx in indices):
                    valid_frames += 1
                    
        except Exception as e:
            print(f"Validation Error: {e}")
            return False, str(e)
        finally:
            cap.release()
            
        # Decision logic: strictness can be adjusted. 
        # If at least 1 checked frame clearly looks like a road scene, we verify it.
        # This prevents rejecting a video just because one frame was blurry or empty road.
        if valid_frames >= 1:
            return True, "Valid Dashcam Footage"
        else:
            return False, "No traffic context detected (e.g., no cars, roads, or signs found)"
