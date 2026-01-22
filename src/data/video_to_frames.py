import cv2
import os

VIDEO_FOLDER = r"e:\Kush\2nd_year\projects\accident_pred\dataset\videos\Crash-1500"
OUTPUT_FOLDER = r"e:\Kush\2nd_year\projects\accident_pred\dataset\frames\Crash-1500"

def video_to_frames_converter(video_source, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    if os.path.isdir(video_source):
        video_files = sorted([f for f in os.listdir(video_source) if f.endswith('.mp4')])
        base_folder = video_source
        print(f"Batch Mode: Found {len(video_files)} videos in folder.")
    elif os.path.isfile(video_source):
        video_files = [os.path.basename(video_source)]
        base_folder = os.path.dirname(video_source)
        print(f"Single Mode: Processing {video_files[0]}")
    else:
        print(f"Error: {video_source} is not a valid file or directory.")
        return

    for video_num, video_name in enumerate(video_files, start=1):
        full_video_path = os.path.join(base_folder, video_name)
        cap = cv2.VideoCapture(full_video_path)
        
        frame_num = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            

            filename = f"video{video_num:03d}_frame{frame_num:02d}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), frame)
            frame_num += 1
        
        cap.release()
        
        if video_num % 100 == 0:
            print(f"Processed {video_num} videos...")

    print("Done!")



if __name__ == "__main__":
    video_to_frames_converter(VIDEO_FOLDER, OUTPUT_FOLDER)