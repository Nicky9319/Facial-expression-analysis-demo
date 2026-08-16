"""
Shared model loading utilities.

GPU-first: tries CUDA/MPS, falls back to CPU.
All functions return (model, device) tuples.
"""

import time
import torch
from pathlib import Path

from ultralytics import YOLO
from transformers import pipeline, CLIPModel, CLIPProcessor
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe import Image as MpImage, ImageFormat


def _best_device():
    """Return 'cuda', 'mps', or 'cpu', with availability check."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_yolo():
    """Load YOLO person-detection model. GPU-first.

    Downloads automatically on first run (cached at ~/.cache/ultralytics/).
    """
    device = _best_device()
    start = time.time()
    model = YOLO("yolo11n.pt")
    model.to(device)
    print(f"[models] YOLO loaded on {device} in {time.time() - start:.2f}s")
    return model, device


def load_emotion_classifier():
    """Load HuggingFace emotion classifier. GPU-first."""
    device = _best_device()
    start = time.time()
    # dima806/facial_emotions_image_detection supports both GPU and CPU
    clf = pipeline(
        "image-classification",
        model="dima806/facial_emotions_image_detection",
        device=0 if device == "cuda" else -1,
    )
    print(f"[models] Emotion classifier loaded on {device} in {time.time() - start:.2f}s")
    return clf, device


def load_clip():
    """Load OpenAI CLIP for clothing detection. GPU-first."""
    device = _best_device()
    start = time.time()
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        clip_model.to(device)
        print(f"[models] CLIP loaded on {device} in {time.time() - start:.2f}s")
        return clip_model, clip_processor
    except Exception as e:
        print(f"[models] CLIP load failed: {e} — clothing detection disabled")
        return None, None


def load_face_mesh():
    """Load MediaPipe face-landmarker. CPU only (no GPU path)."""
    start = time.time()
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
        print(f"[models] MediaPipe face mesh loaded in {time.time() - start:.2f}s")
        return face_mesh
    except Exception as e:
        print(f"[models] MediaPipe load failed: {e}")
        return None
