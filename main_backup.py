import cv2
import subprocess
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


INPUT = "input.mp4"
OUTPUT = "output.mp4"

# Process every Nth frame
FRAME_SKIP = 3

# YOLO confidence
YOLO_CONFIDENCE = 0.40

# Number of recent predictions used for smoothing
SMOOTHING_WINDOW = 7

# Beard classification confidence threshold
BEARD_THRESHOLD = 0.5


def log(message):
    print(f"[INFO] {message}", flush=True)


def init_clip():
    """Initialize CLIP for clothing detection."""
    log("Loading CLIP model...")
    try:
        start = time.time()
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        log(f"CLIP loaded in {time.time() - start:.2f}s")
        return clip_model, clip_processor
    except Exception as e:
        log(f"CLIP load failed: {e}, clothing detection disabled")
        return None, None


def init_mediapipe():
    """Initialize MediaPipe face mesh for landmark analysis."""
    log("Loading MediaPipe face mesh...")
    try:
        model_path = str(Path.home() / ".cache/mediapipe/models/face_landmarker.task")
        base_options = BaseOptions(model_asset_path=model_path)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        face_mesh = FaceLandmarker.create_from_options(options)
        log("MediaPipe face mesh loaded")
        return face_mesh
    except Exception as e:
        log(f"MediaPipe load failed: {e}")
        return None


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

        # Normalize and compute similarity
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sim = (img_emb @ text_emb.T).softmax(dim=-1)

        probs = sim[0].tolist()
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        best_prob = probs[best_idx]

        # Map to simplified labels
        if best_idx == 0:
            label = "company_tshirt"
        elif best_idx == 1:
            label = "formal_attire"
        elif best_idx == 2:
            label = "casual_no_branding"
        elif best_idx == 3:
            label = "polo_logo"
        else:
            label = "casual_tshirt"

        return label, best_prob

    except Exception as e:
        return "unknown", 0.0


def detect_beard_mediapipe(face_mesh, face_crop):
    """Detect beard using MediaPipe face mesh landmarks."""
    if face_mesh is None:
        return "unknown", 0.0

    try:
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        h = rgb.shape[0]

        # Create MediaPipe Image - pass numpy array directly, not bytes
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)

        results = face_mesh.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return "clean_shaven", 0.0

        landmarks = results.face_landmarks[0]

        # Landmark indices for chin (152), left cheek (234), right cheek (454)
        chin_y = None
        left_cheek_y = None
        right_cheek_y = None

        for i, lm in enumerate(landmarks):
            if i == 152:  # chin
                chin_y = lm.y * h
            if i == 234:  # left cheek
                left_cheek_y = lm.y * h
            if i == 454:  # right cheek
                right_cheek_y = lm.y * h

        # If chin is significantly below cheeks relative to face height, suggests beard
        if chin_y is not None and left_cheek_y is not None and right_cheek_y is not None:
            cheek_avg = (left_cheek_y + right_cheek_y) / 2
            chin_lower_ratio = (chin_y - cheek_avg) / h  # normalize by face height

            if chin_lower_ratio > 0.06:
                return "beard_full", 0.75
            elif chin_lower_ratio > 0.03:
                return "beard_stubble", 0.60
            else:
                return "clean_shaven", 0.80

        return "clean_shaven", 0.70

    except Exception as e:
        return "clean_shaven", 0.0


def analyze_grooming(face_crop, face_mesh):
    """Analyze sanitation/grooming from face and skin analysis."""
    try:
        if face_crop.size == 0:
            return "unknown", 0.0

        # 1. Skin color uniformity (low variance = clean skin)
        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
        s_std = hsv[:, :, 1].std()

        # High saturation or high variance in hue → possibly dirty/oily
        skin_uniformity = 1.0 - min(1.0, (s_std / 50.0))

        # 2. Face brightness analysis
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        brightness_score = 1.0 - abs(brightness - 128) / 128  # prefer mid brightness

        # 3. Use MediaPipe face mesh to check symmetry (asymmetry = poor grooming)
        symmetry_score = 0.7  # default
        if face_mesh is not None:
            try:
                rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
                results = face_mesh.detect(mp_image)
                if results.face_landmarks and len(results.face_landmarks) > 0:
                    landmarks = results.face_landmarks[0]
                    # Compare left-right landmark positions for symmetry
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

        # Combine scores
        grooming_score = (skin_uniformity * 0.35 + brightness_score * 0.25 + symmetry_score * 0.4)

        if grooming_score > 0.75:
            return "well_groomed", grooming_score
        elif grooming_score > 0.5:
            return "acceptable", grooming_score
        else:
            return "poor_grooming", grooming_score

    except Exception as e:
        return "unknown", 0.0


def main():

    print("=" * 70, flush=True)
    print("FACIAL EXPRESSION VIDEO ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # =========================================================
    # INPUT
    # =========================================================

    input_path = Path(INPUT)

    if not input_path.exists():
        print(f"[ERROR] Input does not exist: {input_path}", flush=True)
        return

    log(f"Input: {input_path.absolute()}")
    log(f"Input size: {input_path.stat().st_size / (1024 * 1024):.2f} MB")

    # =========================================================
    # MODELS
    # =========================================================

    log("Loading YOLO...")
    try:
        start = time.time()
        model = YOLO("yolo11n.pt")
        log(f"YOLO loaded in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"[ERROR] Could not load YOLO: {e}", flush=True)
        return

    log("Loading emotion classifier...")
    try:
        start = time.time()
        emotion_classifier = pipeline(
            "image-classification",
            model="dima806/facial_emotions_image_detection",
            device=-1,
        )
        log(f"Emotion classifier loaded in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"[ERROR] Could not load emotion classifier: {e}", flush=True)
        return

    clip_model, clip_processor = init_clip()
    face_mesh = init_mediapipe()

    # =========================================================
    # VIDEO
    # =========================================================

    log("Opening input video...")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print("[ERROR] Could not open video", flush=True)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    log(f"FPS: {fps:.2f}, Frames: {frame_count}, Resolution: {width}x{height}, Duration: {duration:.2f}s")
    log(f"Processing every {FRAME_SKIP} frames")

    # =========================================================
    # FFMPEG
    # =========================================================

    command = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        OUTPUT,
    ]

    log("Starting FFmpeg...")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print("[ERROR] FFmpeg not installed. Run: sudo apt install ffmpeg", flush=True)
        cap.release()
        return

    # =========================================================
    # STATE
    # =========================================================

    expression_history = deque(maxlen=SMOOTHING_WINDOW)
    clothing_history = deque(maxlen=SMOOTHING_WINDOW)
    beard_history = deque(maxlen=SMOOTHING_WINDOW)
    grooming_history = deque(maxlen=SMOOTHING_WINDOW)

    current_expression = "unknown"
    current_clothing = "unknown"
    current_beard = "unknown"
    current_grooming = "unknown"

    # =========================================================
    # COUNTERS
    # =========================================================

    frame_number = 0
    processed_frames = 0
    detections = 0

    processing_start = time.time()

    # =========================================================
    # PROCESS VIDEO
    # =========================================================

    log("=" * 70)
    log("STARTING ANALYSIS")
    log("=" * 70)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log("End of video.")
                break

            frame_number += 1

            if frame_number % FRAME_SKIP == 0:
                processed_frames += 1

                # ---- YOLO DETECTION ----
                try:
                    results = model(frame, classes=[0], conf=YOLO_CONFIDENCE, verbose=False, device="cpu")
                except Exception as e:
                    log(f"YOLO ERROR at frame {frame_number}: {e}")
                    results = []

                # ---- FIND BEST PERSON ----
                best_person = None
                best_confidence = 0

                for result in results:
                    if result.boxes is None:
                        continue
                    for i in range(len(result.boxes)):
                        confidence = float(result.boxes.conf[i])
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_person = result.boxes.xyxy[i].tolist()

                if best_person is not None:
                    detections += 1

                    x1, y1, x2, y2 = map(int, best_person)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    person_box = frame[y1:y2, x1:x2]
                    h, w = y2 - y1, x2 - x1

                    if person_box.size > 0:
                        # ---- FACE REGION (upper 40% of person box) ----
                        face_h = int(h * 0.4)
                        face_crop = frame[y1:y1 + face_h, x1:x2]

                        # ---- BODY REGION (lower 60%) ----
                        body_y1 = y1 + int(h * 0.4)
                        body_y2 = min(height, y2)
                        body_crop = frame[body_y1:body_y2, x1:x2]

                        # ---- EXPRESSION ----
                        expression_label = "unknown"
                        expression_conf = 0.0
                        try:
                            if face_crop.size > 0:
                                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(face_rgb)
                                expr_results = emotion_classifier(pil_img)
                                if expr_results:
                                    expression_label = expr_results[0]["label"]
                                    expression_conf = expr_results[0]["score"]
                                    expression_history.append(expression_label)
                                    counts = Counter(expression_history)
                                    current_expression = counts.most_common(1)[0][0]
                        except Exception as e:
                            log(f"Expression ERROR at frame {frame_number}: {e}")

                        # ---- CLOTHING (CLIP) ----
                        clothing_label = "unknown"
                        clothing_conf = 0.0
                        try:
                            if body_crop.size > 0:
                                clothing_label, clothing_conf = detect_clothing(
                                    clip_model, clip_processor, body_crop
                                )
                                clothing_history.append(clothing_label)
                                counts = Counter(clothing_history)
                                current_clothing = counts.most_common(1)[0][0]
                        except Exception as e:
                            log(f"Clothing ERROR at frame {frame_number}: {e}")

                        # ---- BEARD (MediaPipe) ----
                        beard_label = "unknown"
                        beard_conf = 0.0
                        try:
                            if face_crop.size > 0:
                                beard_label, beard_conf = detect_beard_mediapipe(face_mesh, face_crop)
                                beard_history.append(beard_label)
                                counts = Counter(beard_history)
                                current_beard = counts.most_common(1)[0][0]
                        except Exception as e:
                            log(f"Beard ERROR at frame {frame_number}: {e}")

                        # ---- GROOMING ----
                        grooming_label = "unknown"
                        grooming_conf = 0.0
                        try:
                            if face_crop.size > 0:
                                grooming_label, grooming_conf = analyze_grooming(face_crop, face_mesh)
                                grooming_history.append(grooming_label)
                                counts = Counter(grooming_history)
                                current_grooming = counts.most_common(1)[0][0]
                        except Exception as e:
                            log(f"Grooming ERROR at frame {frame_number}: {e}")

                        log(
                            f"Frame={frame_number} | "
                            f"Expr={current_expression}({expression_conf:.1f}) | "
                            f"Clothing={current_clothing}({clothing_conf:.1f}) | "
                            f"Beard={current_beard}({beard_conf:.1f}) | "
                            f"Groom={current_grooming}({grooming_conf:.1f})"
                        )

                        # =========================================
                        # DRAW ANNOTATIONS
                        # =========================================

                        # Person bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Info panel background
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

                        def put_label(txt, y_offset):
                            y = y_offset
                            cv2.putText(frame, txt, (panel_x1 + 5, y), font, font_scale, (0, 255, 0), thickness)

                        put_label(f"Expr: {current_expression} ({expression_conf:.0%})", line_y); line_y += 22
                        put_label(f"Clothing: {current_clothing} ({clothing_conf:.0%})", line_y); line_y += 22
                        put_label(f"Beard: {current_beard} ({beard_conf:.0%})", line_y); line_y += 22
                        put_label(f"Grooming: {current_grooming} ({grooming_conf:.0%})", line_y); line_y += 22

            # ---- STATUS OVERLAY (top-left) ----
            status = f"Expr: {current_expression} | Clothing: {current_clothing} | Beard: {current_beard} | Groom: {current_grooming}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # ---- WRITE FRAME ----
            try:
                process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                log("[ERROR] FFmpeg pipe closed.")
                break

            if frame_number % 30 == 0:
                elapsed = time.time() - processing_start
                speed = frame_number / elapsed if elapsed > 0 else 0
                progress = frame_number / frame_count * 100 if frame_count > 0 else 0
                log(f"Progress={progress:.1f}% | Frame={frame_number}/{frame_count} | Speed={speed:.2f} FPS")

    except Exception as e:
        print(f"[ERROR] Main processing error: {e}", flush=True)

    finally:
        cap.release()
        try:
            process.stdin.close()
        except Exception:
            pass

    # =========================================================
    # FFMPEG COMPLETION
    # =========================================================

    log("Waiting for FFmpeg...")
    stderr = process.stderr.read().decode(errors="replace")
    return_code = process.wait()
    log(f"FFmpeg exit code: {return_code}")

    if return_code != 0:
        print("[ERROR] FFmpeg failed:", flush=True)
        print(stderr[-5000:], flush=True)
        return

    output_path = Path(OUTPUT)
    if not output_path.exists():
        print("[ERROR] Output file was not created.", flush=True)
        return

    output_size = output_path.stat().st_size / (1024 * 1024)

    # =========================================================
    # SUMMARY
    # =========================================================

    print("=" * 70, flush=True)
    print("ANALYSIS COMPLETE", flush=True)
    print("=" * 70, flush=True)

    log(f"Frames processed: {processed_frames}")
    log(f"Person detections: {detections}")
    log(f"Final expression: {current_expression}")
    log(f"Final clothing: {current_clothing}")
    log(f"Final beard: {current_beard}")
    log(f"Final grooming: {current_grooming}")
    log(f"Output: {output_path.absolute()}")
    log(f"Output size: {output_size:.2f} MB")
    log("DONE")


if __name__ == "__main__":
    main()
