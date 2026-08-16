# Facial Expression Analysis

Real-time and batch facial expression analysis with person detection, emotion classification, beard detection, grooming analysis, and clothing detection.

## Project Structure

```
facial-expression-analysis/
├── assets/            # Model weights and static images
├── inputs/            # Input video files for post-processing
├── postprocess/       # Batch video analysis (input video → annotated output)
├── realtime/          # Live webcam analysis
├── src/               # Shared model-loading utilities
├── tests/             # Individual component test scripts
├── pyproject.toml
└── uv.lock
```

## Setup

```bash
# Install dependencies
pip install ultralytics transformers Pillow opencv-python mediapipe torch

# Download MediaPipe face landmark model (required for beard/grooming)
python -c "from mediapipe.tasks.python.vision import FaceLandmarker; \
  from mediapipe.tasks.python.core.base_options import BaseOptions; \
  from pathlib import Path; \
  Path.home().joinpath('.cache/mediapipe/models').mkdir(parents=True, exist_ok=True); \
  print('MediaPipe cache ready')"

# Download YOLO model (auto-downloaded on first run if missing)
```

## Usage

### Realtime (Webcam)

```bash
python realtime/main.py
# or
python -m realtime.main
```

- Press **'q'** to quit.
- Press **'s'** to save the current annotated frame.

### Post-Processing (Batch Video)

```bash
# Place your input video at inputs/input.mp4
python postprocess/main.py
# Output saved as output.mp4 in the repo root
```

## Models & Device Selection

Both scripts automatically select the best available device:

| Priority | Device | How it's selected |
|----------|--------|-------------------|
| 1 | NVIDIA GPU (CUDA) | `torch.cuda.is_available()` |
| 2 | Apple Silicon GPU (MPS) | `torch.backends.mps.is_available()` |
| 3 | CPU | fallback |

YOLO, CLIP, and the HuggingFace emotion classifier all respect this device selection. MediaPipe face landmarks is CPU-only (no GPU path available).

## Detections

| Task | Method | GPU-First |
|------|--------|-----------|
| Person detection | YOLO11n | ✅ |
| Emotion classification | `dima806/facial_emotions_image_detection` (HuggingFace) | ✅ |
| Beard detection | MediaPipe face landmarks (chin/cheek geometry) | ❌ CPU only |
| Grooming analysis | Skin uniformity + brightness + facial symmetry | ❌ CPU only |
| Clothing detection | OpenAI CLIP (`clip-vit-base-patch32`) | ✅ |

## Testing Individual Components

```bash
python tests/test_yolo.py        # Test YOLO person detection
python tests/test_emotion.py    # Test emotion classifier
python tests/test_expression.py # Test expression pipeline end-to-end
```
