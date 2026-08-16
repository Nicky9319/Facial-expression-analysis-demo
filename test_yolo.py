import cv2
import time
from ultralytics import YOLO

print("[1] Loading YOLO...", flush=True)

model = YOLO("yolo11n.pt")

print("[2] YOLO loaded", flush=True)

cap = cv2.VideoCapture("input.mp4")

if not cap.isOpened():
    raise RuntimeError("Could not open video")

print("[3] Video opened", flush=True)

ret, frame = cap.read()

if not ret:
    raise RuntimeError("Could not read frame")

print(
    f"[4] Frame loaded: {frame.shape}",
    flush=True
)

print("[5] Starting YOLO inference...", flush=True)

start = time.time()

results = model(
    frame,
    device="cpu",
    verbose=True,
)

print(
    f"[6] YOLO completed in {time.time() - start:.2f}s",
    flush=True
)

print(
    f"[7] Results: {len(results)}",
    flush=True
)

if results:
    print(
        f"[8] Detections: {len(results[0].boxes)}",
        flush=True
    )

cap.release()

print("[9] SUCCESS", flush=True)