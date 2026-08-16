"""
Realtime Facial Expression Analysis — Webcam Version

Processes live webcam feed with:
  - YOLO person detection
  - Expression classification (HuggingFace)
  - Beard detection (MediaPipe face landmarks)
  - Grooming analysis (skin/brightness/symmetry)
  - Clothing detection (CLIP)

Press 'q' to quit.
Press 's' to save current annotated frame.
"""

import cv2
import time
from pathlib import Path
from collections import Counter, deque

from ultralytics import YOLO
from transformers import pipeline, CLIPModel, CLIPProcessor
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe import Image as MpImage, ImageFormat


# YOLO confidence
YOLO_CONFIDENCE = 0.40

# Number of recent predictions used for smoothing
SMOOTHING_WINDOW = 7

# Target FPS — skip frames if we're behind
TARGET_FPS = 10  # realistic for CPU


def log(message):
    print(f"[INFO] {message}", flush=True)


def init_models():
    """Load all models once at startup."""
    models = {}

    log("Loading YOLO...")
    start = time.time()
    models["yolo"] = YOLO("yolo11n.pt")
    log(f"YOLO loaded in {time.time() - start:.2f}s")

    log("Loading emotion classifier...")
    start = time.time()
    models["emotion"] = pipeline(
        "image-classification",
        model="dima806/facial_emotions_image_detection",
        device=-1,  # CPU
    )
    log(f"Emotion classifier loaded in {time.time() - start:.2f}s")

    log("Loading CLIP model...")
    start = time.time()
    models["clip_model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    models["clip_processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    models["clip_model"].eval()
    log(f"CLIP loaded in {time.time() - start:.2f}s")

    log("Loading MediaPipe face mesh...")
    start = time.time()
    model_path = str(Path.home() / ".cache/mediapipe/models/face_landmarker.task")
    base_options = BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    models["face_mesh"] = FaceLandmarker.create_from_options(options)
    log(f"MediaPipe loaded in {time.time() - start:.2f}s")

    return models


def detect_clothing(clip_model, clip_processor, body_crop):
    """Use CLIP to detect clothing type from body crop."""
    if clip_model is None or clip_processor is None:
        return "unknown", 0.0

    try:
        rgb = cv2.cvtColor(body_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        texts = [
            "person wearing a company branded t-shirt or uniform",
            "person wearing formal office attire with collar",
            "person wearing casual clothing without branding",
            "person wearing a polo shirt with logo",
            "person wearing casual t-shirt",
        ]

        inputs = clip_processor(
            text=texts,
            images=pil_img,
            return_tensors="pt",
            padding=True,
        )

        outputs = clip_model(**inputs)
        text_emb = outputs.text_embeds
        img_emb = outputs.image_embeds

        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sim = (img_emb @ text_emb.T).softmax(dim=-1)

        probs = sim[0].tolist()
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        best_prob = probs[best_idx]

        labels = [
            "company_tshirt", "formal_attire", "casual_no_branding",
            "polo_logo", "casual_tshirt"
        ]
        return labels[best_idx], best_prob

    except Exception:
        return "unknown", 0.0


def detect_beard(face_mesh, face_crop):
    """Detect beard using MediaPipe face mesh landmarks."""
    if face_mesh is None:
        return "unknown", 0.0

    try:
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        h = rgb.shape[0]
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
        results = face_mesh.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return "clean_shaven", 0.0

        landmarks = results.face_landmarks[0]

        chin_y = left_cheek_y = right_cheek_y = None
        for i, lm in enumerate(landmarks):
            if i == 152:
                chin_y = lm.y * h
            elif i == 234:
                left_cheek_y = lm.y * h
            elif i == 454:
                right_cheek_y = lm.y * h

        if chin_y is not None and left_cheek_y is not None and right_cheek_y is not None:
            cheek_avg = (left_cheek_y + right_cheek_y) / 2
            chin_lower_ratio = (chin_y - cheek_avg) / h
            if chin_lower_ratio > 0.06:
                return "beard_full", 0.75
            elif chin_lower_ratio > 0.03:
                return "beard_stubble", 0.60
            return "clean_shaven", 0.80
        return "clean_shaven", 0.70

    except Exception:
        return "clean_shaven", 0.0


def analyze_grooming(face_crop, face_mesh):
    """Analyze sanitation/grooming from face and skin analysis."""
    try:
        if face_crop.size == 0:
            return "unknown", 0.0

        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
        s_std = hsv[:, :, 1].std()
        skin_uniformity = 1.0 - min(1.0, (s_std / 50.0))

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        brightness_score = 1.0 - abs(brightness - 128) / 128

        symmetry_score = 0.7
        if face_mesh is not None:
            try:
                rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
                results = face_mesh.detect(mp_image)
                if results.face_landmarks and len(results.face_landmarks) > 0:
                    landmarks = results.face_landmarks[0]
                    left_eye_x = landmarks[33].x
                    right_eye_x = landmarks[263].x
                    eye_distance = abs(left_eye_x - right_eye_x)
                    nose_x = landmarks[1].x
                    left_dist = abs(nose_x - left_eye_x)
                    right_dist = abs(right_eye_x - nose_x)
                    symmetry = 1.0 - min(1.0, abs(left_dist - right_dist) / eye_distance)
                    symmetry_score = symmetry
            except Exception:
                pass

        grooming_score = skin_uniformity * 0.35 + brightness_score * 0.25 + symmetry_score * 0.4

        if grooming_score > 0.75:
            return "well_groomed", grooming_score
        elif grooming_score > 0.5:
            return "acceptable", grooming_score
        return "poor_grooming", grooming_score

    except Exception:
        return "unknown", 0.0


def process_frame(frame, models, state):
    """Process a single frame and update state. Returns annotated frame."""
    height, width = frame.shape[:2]

    # ---- YOLO DETECTION ----
    try:
        results = models["yolo"](
            frame,
            classes=[0],
            conf=YOLO_CONFIDENCE,
            verbose=False,
            device="cpu",
        )
    except Exception:
        results = []

    # ---- FIND BEST PERSON ----
    best_person = None
    best_confidence = 0
    for result in results:
        if result.boxes is None:
            continue
        for i in range(len(result.boxes)):
            conf = float(result.boxes.conf[i])
            if conf > best_confidence:
                best_confidence = conf
                best_person = result.boxes.xyxy[i].tolist()

    if best_person is None:
        # No person — just show frame with current state
        cv2.putText(
            frame, "No person detected",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )
        return frame

    x1, y1, x2, y2 = map(int, best_person)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    h, w = y2 - y1, x2 - x1
    person_box = frame[y1:y2, x1:x2]

    if person_box.size == 0:
        return frame

    # ---- CROPS ----
    face_h = int(h * 0.4)
    face_crop = frame[y1:y1 + face_h, x1:x2]
    body_y1 = y1 + int(h * 0.4)
    body_y2 = min(height, y2)
    body_crop = frame[body_y1:body_y2, x1:x2]

    # ---- EXPRESSION ----
    expr_label = state["expression"]
    expr_conf = 0.0
    if face_crop.size > 0:
        try:
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            expr_results = models["emotion"](pil_img)
            if expr_results:
                expr_label = expr_results[0]["label"]
                expr_conf = expr_results[0]["score"]
                state["expr_history"].append(expr_label)
                counts = Counter(state["expr_history"])
                state["expression"] = counts.most_common(1)[0][0]
        except Exception:
            pass

    # ---- BEARD ----
    beard_label = state["beard"]
    beard_conf = 0.0
    if face_crop.size > 0:
        beard_label, beard_conf = detect_beard(models["face_mesh"], face_crop)
        state["beard_history"].append(beard_label)
        counts = Counter(state["beard_history"])
        state["beard"] = counts.most_common(1)[0][0]

    # ---- GROOMING ----
    grooming_label = state["grooming"]
    grooming_conf = 0.0
    if face_crop.size > 0:
        grooming_label, grooming_conf = analyze_grooming(face_crop, models["face_mesh"])
        state["grooming_history"].append(grooming_label)
        counts = Counter(state["grooming_history"])
        state["grooming"] = counts.most_common(1)[0][0]

    # ---- CLOTHING ----
    clothing_label = state["clothing"]
    clothing_conf = 0.0
    if body_crop.size > 0:
        clothing_label, clothing_conf = detect_clothing(
            models["clip_model"], models["clip_processor"], body_crop
        )
        state["clothing_history"].append(clothing_label)
        counts = Counter(state["clothing_history"])
        state["clothing"] = counts.most_common(1)[0][0]

    # =========================================================
    # DRAW ANNOTATIONS
    # =========================================================

    # Person box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Info panel
    panel_x1 = max(0, x1)
    panel_y1 = max(0, y1 - 120)
    panel_x2 = min(width, x2)
    panel_y2 = y1
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 255, 0), 1)

    line_y = panel_y1 + 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    cv2.putText(frame, f"Expr: {state['expression']} ({expr_conf:.0%})",
                 (panel_x1 + 5, line_y), font, font_scale, (0, 255, 0), thickness); line_y += 22
    cv2.putText(frame, f"Clothing: {state['clothing']} ({clothing_conf:.0%})",
                 (panel_x1 + 5, line_y), font, font_scale, (0, 255, 0), thickness); line_y += 22
    cv2.putText(frame, f"Beard: {state['beard']} ({beard_conf:.0%})",
                 (panel_x1 + 5, line_y), font, font_scale, (0, 255, 0), thickness); line_y += 22
    cv2.putText(frame, f"Groom: {state['grooming']} ({grooming_conf:.0%})",
                 (panel_x1 + 5, line_y), font, font_scale, (0, 255, 0), thickness)

    # Top status bar
    status = (f"Expr: {state['expression']} | Clothing: {state['clothing']} "
              f"| Beard: {state['beard']} | Groom: {state['grooming']}")
    cv2.putText(frame, status, (10, 30),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def main():
    print("=" * 60, flush=True)
    print("REALTIME FACIAL EXPRESSION ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    # ---- LOAD MODELS ----
    models = init_models()

    # ---- STATE ----
    state = {
        "expression": "unknown",
        "clothing": "unknown",
        "beard": "unknown",
        "grooming": "unknown",
        "expr_history": deque(maxlen=SMOOTHING_WINDOW),
        "clothing_history": deque(maxlen=SMOOTHING_WINDOW),
        "beard_history": deque(maxlen=SMOOTHING_WINDOW),
        "grooming_history": deque(maxlen=SMOOTHING_WINDOW),
    }

    # ---- WEBCAM ----
    log("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam", flush=True)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    log(f"Webcam: {actual_w:.0f}x{actual_h:.0f} @ {actual_fps:.0f} FPS")

    log("Press 'q' to quit | Press 's' to save frame")

    frame_interval = 1.0 / TARGET_FPS
    last_process_time = time.time()
    frame_count = 0

    window_name = "Realtime Facial Expression Analysis"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            log("Webcam disconnected")
            break

        now = time.time()
        elapsed = now - last_process_time

        # Skip if we're running behind target FPS
        if elapsed < frame_interval:
            continue

        # ---- PROCESS ----
        frame = process_frame(frame, models, state)
        last_process_time = now
        frame_count += 1

        # ---- DISPLAY ----
        cv2.imshow(window_name, frame)

        # ---- CONTROLS ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            log("Quit requested")
            break
        elif key == ord('s'):
            filename = f"realtime_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            log(f"Saved: {filename}")

        # ---- STATUS ----
        if frame_count % 30 == 0:
            log(f"Running: {state['expression']} | {state['clothing']} | "
                f"{state['beard']} | {state['grooming']}")

    cap.release()
    cv2.destroyAllWindows()
    log("Camera closed. Done.")


if __name__ == "__main__":
    main()
