import cv2
import time

from ultralytics import YOLO
from deepface import DeepFace


print("[1] Loading YOLO...", flush=True)

yolo = YOLO("yolo11n.pt")

print("[2] Opening video...", flush=True)

cap = cv2.VideoCapture("input.mp4")

ret, frame = cap.read()

if not ret:
    raise RuntimeError("Could not read frame")

print(
    f"[3] Frame loaded: {frame.shape}",
    flush=True
)

print("[4] Running YOLO...", flush=True)

results = yolo(
    frame,
    classes=[0],
    conf=0.4,
    device="cpu",
    verbose=False,
)

print("[5] YOLO finished", flush=True)

if not results or results[0].boxes is None:
    print("[ERROR] No people detected")
    exit()

boxes = results[0].boxes

print(
    f"[6] People detected: {len(boxes)}",
    flush=True
)

# Take the largest detected person
best_box = None
best_area = 0

for i in range(len(boxes)):

    x1, y1, x2, y2 = map(
        int,
        boxes.xyxy[i].tolist()
    )

    area = (x2 - x1) * (y2 - y1)

    if area > best_area:
        best_area = area
        best_box = (x1, y1, x2, y2)


x1, y1, x2, y2 = best_box

person = frame[y1:y2, x1:x2]

print(
    f"[7] Person crop: {person.shape}",
    flush=True
)

print(
    "[8] Starting DeepFace...",
    flush=True
)

start = time.time()

analysis = DeepFace.analyze(
    person,
    actions=["emotion"],
    detector_backend="opencv",
    enforce_detection=False,
    silent=True,
)

print(
    f"[9] DeepFace finished in "
    f"{time.time() - start:.2f}s",
    flush=True
)

if isinstance(analysis, list):
    analysis = analysis[0]

print(
    "[10] Dominant expression:",
    analysis["dominant_emotion"],
    flush=True
)

print(
    "[11] Scores:",
    analysis["emotion"],
    flush=True
)

cap.release()

print("[12] SUCCESS", flush=True)