import torch
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class DashcamValidator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Scene Validator on {self.device}...")
        
 
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(self.device)
        self.model.eval()
        

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.valid_indices = {
            407, 
            436, 
            444, 
            468, 
            511, 
            569, 
            609, 
            627, 
            654, 
            656, 
            661, 
            665, 
            670, 
            671, 
            675,
            705, 
            717, 
            734, 
            751, 
            757, 
            779, 
            817, 
            829, 
            864, 
            867, 
            913, 
            914, 
            919, 
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
            int(total_frames * 0.2),    
            int(total_frames * 0.5), 
            int(total_frames * 0.8) 
        ]
        
        valid_frames = 0
        
        try:
            for idx in frame_points:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    
                
                probs, indices = torch.topk(output, 5)
                indices = indices.cpu().numpy()[0]
                
                
                if any(idx in self.valid_indices for idx in indices):
                    valid_frames += 1
                    
        except Exception as e:
            print(f"Validation Error: {e}")
            return False, str(e)
        finally:
            cap.release()
            
        
        if valid_frames >= 1:
            return True, "Valid Dashcam Footage"
        else:
            return False, "No traffic context detected (e.g., no cars, roads, or signs found)"
