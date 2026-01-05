Environment Setup
Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate

Install dependencies
pip install -r requirements.txt

 Dataset Structure

Place your videos in the following structure:

data/
└── videos/
    ├── 0_real/
    │   ├── video1.mp4
    │   └── video2.mp4
    └── 1_fake/
        ├── video3.mp4
        └── video4.mp4

 Step 1 — Sample Frames from Videos

This step extracts frames from each video.

Run:
python sampling.py

Output:
data/
└── frames/
    ├── 0_real/
    └── 1_fake/


Each video is converted into a fixed number of frames.

 Step 2 — Compute BRISQUE Artifact Scores

This step computes perceptual artifact scores using the BRISQUE model.

Make sure the following files exist:

brisque_model_live.yml
brisque_range_live.yml

Run:
python Brisque.py

Output:
artifact_scores_brisque.csv


This file contains:

image path

label (real/fake)

BRISQUE artifact score

Step 3 — Generate Metadata & Perceptual Metrics

This step computes additional quality metrics such as:

sharpness

contrast

brightness

Run:
python generate_metadata.py

Output:
metadata.csv

Step 4 — Caption Generation with BLIP2

This step generates frame-level and video-level captions using BLIP2.

 BLIP2 must be available locally (or loaded via HuggingFace).

Run:
python caption_generation.py

Outputs:
frames_captions.csv
videos_captions.csv
frames_videos_metadata.csv


These files combine:

perceptual metrics

BRISQUE scores

generated captions

7️⃣ Step 5 — Run Full Pipeline (Optional)

If main.py is provided to execute all steps automatically:

python main.py


This will run:

frame sampling

BRISQUE scoring

metadata generation

caption generation
