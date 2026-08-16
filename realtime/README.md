# Realtime

Live webcam analysis — processes your camera feed in real time and displays annotated output on-screen.

## Usage

```bash
# From the repo root:
python realtime/main.py

# Or as a module:
python -m realtime.main
```

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save current annotated frame to `realtime_frame_<timestamp>.jpg` |

## Pipeline

Each frame (up to 10 FPS by default):

1. **YOLO** — person detection (upper half of frame)
2. **Emotion** — expression from face crop (upper 40% of person box)
3. **Beard** — MediaPipe face landmark geometry
4. **Grooming** — skin + brightness + symmetry analysis
5. **Clothing** — CLIP classification of body crop (lower 60% of person box)

Results are smoothed over a 7-frame sliding window.

## GPU Acceleration

Automatically selects the best available device:

```
CUDA (NVIDIA) → MPS (Apple Silicon) → CPU
```

## Performance

| Device | Expected FPS |
|--------|-------------|
| NVIDIA GPU (CUDA) | 10–30+ |
| Apple Silicon (MPS) | 10–20 |
| CPU | ~5–10 |

`TARGET_FPS` is set to 10 by default. The script skips frames when behind schedule to maintain real-time display.
