## Usage (Pipeline)

### 0) Installation
Create and activate a virtual environment, then install dependencies.

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate

pip install -r requirements.txt
```

---

### 1) Prepare the input videos
Place your videos in the following structure:

```
data/videos/
├── 0_real/   # real videos
└── 1_fake/   # fake videos
```

---

### 2) Sample frames from videos
This step extracts a fixed number of frames from each video.

```bash
python sampling.py
```

Output:
```
data/frames/
├── 0_real/
└── 1_fake/
```

---

### 3) Compute BRISQUE artifact scores
Make sure the following files are available in the repository root:

- `brisque_model_live.yml`
- `brisque_range_live.yml`

Run:
```bash
python Brisque.py
```

Output:
- `artifact_scores_brisque.csv`

This file contains the image path, label (real/fake), and BRISQUE artifact score.

---

### 4) Generate metadata and perceptual metrics
This step computes additional quality metrics such as sharpness, contrast, and brightness.

```bash
python generate_metadata.py
```

Output:
- `metadata.csv`

---

### 5) Generate captions with BLIP2
This step generates frame-level and video-level captions using BLIP2.

```bash
python caption_generation.py
```

Outputs:
- `frames_captions.csv`
- `videos_captions.csv`
- `frames_videos_metadata.csv`

These files combine perceptual metrics, BRISQUE scores, and generated captions.

---

### 6) Run the full pipeline (optional)
If you want to execute all steps sequentially using a single command:

```bash
python main.py
```

