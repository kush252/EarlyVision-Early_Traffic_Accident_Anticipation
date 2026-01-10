import cv2
import os

# === CONFIGURATION ===
VIDEO_FOLDER = r"e:\Kush\2nd_year\projects\accident_pred\data\videos\Crash-1500"
OUTPUT_FOLDER = r"e:\Kush\2nd_year\projects\accident_pred\data\frames\Crash-1500"

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get all video files sorted
videos = sorted([f for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')])

print(f"Found {len(videos)} videos")

for video_num, video_name in enumerate(videos, start=1):
    video_path = os.path.join(VIDEO_FOLDER, video_name)
    cap = cv2.VideoCapture(video_path)
    
    frame_num = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Naming: video001_frame01.jpg, video001_frame02.jpg, etc.
        filename = f"video{video_num:03d}_frame{frame_num:02d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), frame)
        frame_num += 1
    
    cap.release()
    
    if video_num % 100 == 0:
        print(f"Processed {video_num} videos...")

print("Done!")
