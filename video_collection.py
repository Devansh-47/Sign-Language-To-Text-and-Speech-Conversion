import os
import cv2
import numpy as np

# Directory containing video files
video_data_dir = 'hindi_video'
# Directory to save extracted frames
output_data_dir = 'save_frame.jpg'

# Function to extract frames from a video and save them as images
def extract_and_save_frames(video_path, label, output_dir, frame_size=(64, 64)):
    # Create a directory for the label if it doesn't exist
    label_dir = os.path.join(output_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize and normalize the frame
        resized_frame = cv2.resize(frame, frame_size)
        resized_frame = resized_frame / 255.0  # Normalize
        
        # Construct the filename for the frame
        frame_filename = os.path.join(label_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, (resized_frame * 255).astype(np.uint8))  # Convert back to [0, 255] for saving
        
        frame_count += 1
    
    cap.release()
    print(f"Saved {frame_count} frames to {label_dir}")

# Extract frames from all videos and save them
for label_dir in os.listdir(video_data_dir):
    label_path = os.path.join(video_data_dir, label_dir)
    if os.path.isdir(label_path):
        for video_file in os.listdir(label_path):
            video_path = os.path.join(label_path, video_file)
            print(f"Processing video: {video_path}")
            extract_and_save_frames(video_path, label_dir, output_data_dir)

print("Finished extracting and saving frames.")
