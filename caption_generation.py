import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BLIP-2 model and processor from local path
blip_model_path = "./BLIP2"  # Update this path as needed
blip_processor = Blip2Processor.from_pretrained(blip_model_path)
blip_model = Blip2ForConditionalGeneration.from_pretrained(blip_model_path).to(device)

# Paths
input_dir = "sampled_data"  # Folder with 0_real and 1_fake
metadata_file = "metadata.csv"  # Path to metadata file
output_frames_csv = "frames_captions.csv"  # Frame-level captions
output_videos_csv = "videos_captions.csv"  # Video-level captions
output_combined_csv = "frames_videos_metadata.csv"  # Combined CSV

# Custom prompts based on label
REAL_PROMPT = "Describe this face naturally and realistically, as if describing a real person."
FAKE_PROMPT = "Describe this face, but something about it seems slightly off or artificial."

# Generate captions for a single image based on its label
def generate_caption(image_path, label):
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Select the appropriate prompt based on real or fake
        custom_prompt = REAL_PROMPT if label == "0_real" else FAKE_PROMPT
        
        inputs = blip_processor(images=image, text=custom_prompt, return_tensors="pt").to(device)
        outputs = blip_model.generate(**inputs)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Process all frames in sampled_data
def process_frames(input_dir):
    frame_results = []
    video_captions = {}

    for label in ["0_real", "1_fake"]:
        folder_path = os.path.join(input_dir, label)

        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist. Skipping.")
            continue

        for frame_name in tqdm(os.listdir(folder_path), desc=f"Processing {label}"):
            frame_path = os.path.join(folder_path, frame_name)
            video_id = "_".join(frame_name.split("_")[:2])  # Extract video ID (e.g., id0_0000)

            # Generate caption for the frame with label-based prompt
            caption = generate_caption(frame_path, label)
            if caption:
                frame_results.append({
                    "Video": video_id,
                    "Frame": frame_name,
                    "Label": label,
                    "Frame_Caption": caption
                })

                # Collect captions for video-level summary
                if video_id not in video_captions:
                    video_captions[video_id] = []
                video_captions[video_id].append(caption)

    return frame_results, video_captions

# Save video-level captions to CSV
def save_video_captions(video_captions, label_mapping, output_videos_csv):
    video_results = []
    for video_id, captions in video_captions.items():
        video_label = label_mapping[video_id]  # Retrieve label
        video_caption = " ".join(captions)  # Concatenate frame captions
        video_results.append({
            "Video": video_id,
            "Label": video_label,
            "Video_Caption": video_caption
        })
    video_df = pd.DataFrame(video_results)
    video_df.to_csv(output_videos_csv, index=False)
    print(f"Video-level captions saved to {output_videos_csv}")

# Combine frame-level, video-level, and metadata
def combine_with_metadata(metadata_file, frames_csv, videos_csv, output_csv):
    metadata_df = pd.read_csv(metadata_file)
    frames_df = pd.read_csv(frames_csv)
    videos_df = pd.read_csv(videos_csv)

    # Merge video-level captions with metadata
    combined_df = pd.merge(metadata_df, videos_df, on=["Video", "Label"], how="inner")

    # Add frame-level captions grouped by video
    frame_grouped = frames_df.groupby(["Video", "Label"])["Frame_Caption"].apply(list).reset_index()
    combined_df = pd.merge(combined_df, frame_grouped, on=["Video", "Label"], how="inner")

    # Save the combined CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined CSV saved to {output_csv}")

# Main function
def main():
    # Map video IDs to their labels
    label_mapping = {}
    for label in ["0_real", "1_fake"]:
        folder_path = os.path.join(input_dir, label)
        for frame_name in os.listdir(folder_path):
            video_id = "_".join(frame_name.split("_")[:2])  # Extract video ID
            label_mapping[video_id] = label

    # Process all frames and generate captions
    frame_results, video_captions = process_frames(input_dir)

    # Save frame-level captions
    frame_df = pd.DataFrame(frame_results)
    frame_df.to_csv(output_frames_csv, index=False)
    print(f"Frame-level captions saved to {output_frames_csv}")

    # Save video-level captions
    save_video_captions(video_captions, label_mapping, output_videos_csv)

    # Combine frames, videos, and metadata
    combine_with_metadata(metadata_file, output_frames_csv, output_videos_csv, output_combined_csv)

if __name__ == "__main__":
    main()
