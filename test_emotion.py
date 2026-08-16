from transformers import pipeline
from PIL import Image
import cv2

print("[1] Loading emotion model...", flush=True)

classifier = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    device=-1,
)

print("[2] Emotion model loaded", flush=True)

image = cv2.imread("face.jpg")

if image is None:
    raise RuntimeError("Could not load face.jpg")

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

image = Image.fromarray(image)

print("[3] Running emotion classification...", flush=True)

results = classifier(image)

print("[4] Results:", flush=True)

for result in results:
    print(
        f"    {result['label']}: "
        f"{result['score']:.4f}",
        flush=True
    )

print("[5] SUCCESS", flush=True)
