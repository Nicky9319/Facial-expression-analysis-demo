# Post-Processing

Batch-processes a video file through the full analysis pipeline and writes an annotated output video.

## Usage

```bash
# From the repo root:
python postprocess/main.py

# Or as a module:
python -m postprocess.main
```

## Input / Output

| File | Location | Description |
|------|----------|-------------|
| Input | `inputs/input.mp4` | Source video to analyse |
| Output | `output.mp4` (repo root) | Annotated output video |

## Pipeline

1. **YOLO** — detects the most confident person in each frame (class 0, confidence ≥ 0.40)
2. **Emotion** — classifies expression from the upper 40% of the person bounding box
3. **Beard** — MediaPipe face landmarks; chin extension beyond cheek line → beard/stubble/shaven
4. **Grooming** — skin saturation uniformity, face brightness, facial symmetry from landmarks
5. **Clothing** — CLIP zero-shot classification of body region (lower 60% of person box)

Results are smoothed over a 7-frame sliding window. Only every 3rd frame is fully processed to improve throughput.

## GPU Acceleration

The script automatically selects the best device:

```
CUDA (NVIDIA) → MPS (Apple Silicon) → CPU
```

No manual configuration needed. Pass `--device cpu` to force CPU if desired (not currently exposed as a flag — edit the model loader in `src/models.py` to override).

## FFmpeg Requirement

FFmpeg must be installed for video encoding:

```bash
sudo apt install ffmpeg   # Linux
brew install ffmpeg        # macOS
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT` | `inputs/input.mp4` | Input video path |
| `OUTPUT` | `output.mp4` | Output video path |
| `FRAME_SKIP` | `3` | Process every Nth frame |
| `YOLO_CONFIDENCE` | `0.40` | Minimum YOLO detection confidence |
| `SMOOTHING_WINDOW` | `7` | Frames used for result smoothing |
