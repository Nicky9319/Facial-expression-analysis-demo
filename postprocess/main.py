"""
Post-Processing: Facial Expression Video Analysis

Batch-processes a video file (input.mp4 → output.mp4) with:
  - YOLO person detection  (GPU → CPU fallback)
  - Expression classification (HuggingFace, GPU → CPU fallback)
  - Beard detection (MediaPipe face landmarks)
  - Grooming analysis (skin/brightness/symmetry)
  - Clothing detection (CLIP, GPU → CPU fallback)

Usage:
    python -m postprocess.main
    # or from repo root:
    python postprocess/main.py
"""

import sys
import os

# Add project root to path so 'src' is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import subprocess
import time
from pathlib import Path
from collections import Counter, deque

from PIL import Image

from src.models import load_yolo, load_emotion_classifier, load_clip, load_face_mesh


INPUT = "inputs/input.mp4"
OUTPUT = "output.mp4"

# Process every Nth frame
FRAME_SKIP = 3

# YOLO confidence
YOLO_CONFIDENCE = 0.40

# Number of recent predictions used for smoothing
SMOOTHING_WINDOW = 7


def log(message):
    print(f"[INFO] {message}", flush=True)


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
            text=texts, images=pil_img, return_tensors="pt", padding=True
        )
        outputs = clip_model(**inputs)

        text_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
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


def detect_beard_mediapipe(face_mesh, face_crop):
    """Detect beard using MediaPipe face mesh landmarks."""
    from mediapipe import Image as MpImage, ImageFormat

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
    from mediapipe import Image as MpImage, ImageFormat

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


def main():
    print("=" * 70, flush=True)
    print("FACIAL EXPRESSION VIDEO ANALYSIS  (post-process)", flush=True)
    print("=" * 70, flush=True)

    # ---- INPUT ----
    input_path = Path(INPUT)
    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}", flush=True)
        return

    log(f"Input: {input_path.absolute()}")
    log(f"Input size: {input_path.stat().st_size / (1024 * 1024):.2f} MB")

    # ---- LOAD MODELS (GPU-first, CPU fallback) ----
    model, device = load_yolo()
    emotion_classifier, _ = load_emotion_classifier()
    clip_model, clip_processor = load_clip()
    face_mesh = load_face_mesh()

    # ---- VIDEO ----
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

    # ---- FFMPEG ----
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

    # ---- STATE ----
    expression_history = deque(maxlen=SMOOTHING_WINDOW)
    clothing_history = deque(maxlen=SMOOTHING_WINDOW)
    beard_history = deque(maxlen=SMOOTHING_WINDOW)
    grooming_history = deque(maxlen=SMOOTHING_WINDOW)

    current_expression = "unknown"
    current_clothing = "unknown"
    current_beard = "unknown"
    current_grooming = "unknown"
    prev_status_text = ""

    frame_number = 0
    processed_frames = 0
    detections = 0
    processing_start = time.time()

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
                    results = model(
                        frame, classes=[0], conf=YOLO_CONFIDENCE,
                        verbose=False, device=device,
                    )
                except Exception as e:
                    log(f"YOLO ERROR frame {frame_number}: {e}")
                    results = []

                # ---- BEST PERSON ----
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

                if best_person is not None:
                    detections += 1
                    x1, y1, x2, y2 = map(int, best_person)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    h = y2 - y1

                    if h > 0:
                        face_h = int(h * 0.4)
                        face_crop = frame[y1:y1 + face_h, x1:x2]
                        body_y1 = y1 + face_h
                        body_y2 = min(height, y2)
                        body_crop = frame[body_y1:body_y2, x1:x2]

                        # ---- EXPRESSION ----
                        expression_label = "unknown"
                        expression_conf = 0.0
                        if face_crop.size > 0:
                            try:
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
                                log(f"Expression ERROR frame {frame_number}: {e}")

                        # ---- CLOTHING ----
                        clothing_label = "unknown"
                        clothing_conf = 0.0
                        if body_crop.size > 0:
                            try:
                                clothing_label, clothing_conf = detect_clothing(
                                    clip_model, clip_processor, body_crop
                                )
                                clothing_history.append(clothing_label)
                                counts = Counter(clothing_history)
                                current_clothing = counts.most_common(1)[0][0]
                            except Exception as e:
                                log(f"Clothing ERROR frame {frame_number}: {e}")

                        # ---- BEARD ----
                        beard_label = "unknown"
                        beard_conf = 0.0
                        if face_crop.size > 0:
                            try:
                                beard_label, beard_conf = detect_beard_mediapipe(face_mesh, face_crop)
                                beard_history.append(beard_label)
                                counts = Counter(beard_history)
                                current_beard = counts.most_common(1)[0][0]
                            except Exception as e:
                                log(f"Beard ERROR frame {frame_number}: {e}")

                        # ---- GROOMING ----
                        grooming_label = "unknown"
                        grooming_conf = 0.0
                        if face_crop.size > 0:
                            try:
                                grooming_label, grooming_conf = analyze_grooming(face_crop, face_mesh)
                                grooming_history.append(grooming_label)
                                counts = Counter(grooming_history)
                                current_grooming = counts.most_common(1)[0][0]
                            except Exception as e:
                                log(f"Grooming ERROR frame {frame_number}: {e}")

                        log(
                            f"Frame={frame_number} | "
                            f"Expr={current_expression}({expression_conf:.1f}) | "
                            f"Clothing={current_clothing}({clothing_conf:.1f}) | "
                            f"Beard={current_beard}({beard_conf:.1f}) | "
                            f"Groom={current_grooming}({grooming_conf:.1f})"
                        )

                        # ---- ANNOTATIONS ----
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
                            cv2.putText(frame, txt, (panel_x1 + 5, y_offset),
                                        font, font_scale, (0, 255, 0), thickness)

                        put_label(f"Expr: {current_expression} ({expression_conf:.0%})", line_y); line_y += 22
                        put_label(f"Clothing: {current_clothing} ({clothing_conf:.0%})", line_y); line_y += 22
                        put_label(f"Beard: {current_beard} ({beard_conf:.0%})", line_y); line_y += 22
                        put_label(f"Grooming: {current_grooming} ({grooming_conf:.0%})", line_y); line_y += 22

            # ---- STATUS OVERLAY ----
            status = (f"Expr: {current_expression} | Clothing: {current_clothing} "
                      f"| Beard: {current_beard} | Groom: {current_grooming}")
            if status != prev_status_text:
                cv2.rectangle(frame, (5, 5), (800, 40), (0, 0, 0), -1)
                cv2.putText(frame, status, (10, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                prev_status_text = status

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

    # ---- FFMPEG COMPLETION ----
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

    # ---- SUMMARY ----
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
