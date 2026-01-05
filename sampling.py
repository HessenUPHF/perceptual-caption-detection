import os
import cv2
from tqdm import tqdm

# Set input directory that contains '0_real' and '1_fake' subdirectories
input_video_dir = 'Celeb-df-v2'

# Define the base output directory
output_base_dir = 'sampled_data'

# Global directories for real and fake videos
output_real_dir = os.path.join(output_base_dir, '0_real')
output_fake_dir = os.path.join(output_base_dir, '1_fake')

# Automatically create the necessary directories
os.makedirs(output_real_dir, exist_ok=True)
os.makedirs(output_fake_dir, exist_ok=True)

def sample_frames_from_videos(frames_per_video=5):
    # Loop through both '0_real' and '1_fake' video folders
    for label in ['0_real', '1_fake']:
        input_path = os.path.join(input_video_dir, label)
        output_path = os.path.join(output_base_dir, label)
        
        # Go through all the video files in each folder
        video_files = os.listdir(input_path)
        for file_name in tqdm(video_files, desc=f"Sampling {label} videos", unit="video"):
            video_path = os.path.join(input_path, file_name)
            cap = cv2.VideoCapture(video_path)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Generate indices to extract the frames at equal intervals
            frame_indices = list(range(0, total_frames, max(total_frames // frames_per_video, 1)))

            # Add a progress bar for frames being processed
            for idx, frame_index in tqdm(enumerate(frame_indices), total=len(frame_indices), desc=f"Processing frames for {file_name}", unit="frame", leave=False):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                
                if not ret:
                    break  # Break if the frame could not be read
                
                # Save the frame in the appropriate directory
                output_frame_path = os.path.join(output_path, f"{file_name.split('.')[0]}_frame{idx}.jpg")
                cv2.imwrite(output_frame_path, frame)

            cap.release()

# Run the sampling process
sample_frames_from_videos(frames_per_video=5)

print(f"Frames are saved in: {output_base_dir}")
