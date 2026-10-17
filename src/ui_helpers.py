"""
ui_helpers.py
---------------------------------
Logic used by the Streamlit front end (app.py), kept free of any Streamlit
code so that it can be tested on its own (see tests/test_ui_helpers.py).

  - quality_report()     : "is this photo usable?" check (blur / dark / small)
  - save_upload()        : validates and stores an uploaded photo
  - confirm_prediction() : logs the worker's final decision AND syncs to BPA
  - read_feedback_df() / read_bpa_df() : tables for the Records tab
"""

import hashlib
import io
import json
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import config
from feedback_loop import log_feedback, summarize_feedback
from bpa_sync_stub import sync_breed_to_bpa, MOCK_BPA_DB_PATH

UPLOAD_DIR = os.path.join(config.LOGS_DIR, "uploads")

# Advisory thresholds for the capture-quality check (section 4.2, "Image Capture")
BLUR_THRESHOLD = 60.0       # variance of Laplacian below this -> likely blurred
DARK_THRESHOLD = 50.0       # mean grey level below this      -> too dark
BRIGHT_THRESHOLD = 215.0    # mean grey level above this      -> over-exposed
MIN_SIDE_PX = 160           # smaller than the model input     -> too small


def quality_report(image_bytes):
    """Returns {"ok": bool, "warnings": [str], "blur": float, "brightness": float, "size": (w, h)}.
    Purely advisory: the worker can still proceed with a flagged photo."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("The file is not a readable image.")
    h, w = img.shape[:2]
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    brightness = float(grey.mean())

    warnings = []
    if blur < BLUR_THRESHOLD:
        warnings.append("The photo looks blurred - hold the phone steady and retake if possible.")
    if brightness < DARK_THRESHOLD:
        warnings.append("The photo is too dark - move to better light.")
    elif brightness > BRIGHT_THRESHOLD:
        warnings.append("The photo is over-exposed - avoid pointing the camera at the sun.")
    if min(h, w) < MIN_SIDE_PX:
        warnings.append(f"The photo is small ({w}x{h}px) - step closer so the animal fills the frame.")
    return {"ok": not warnings, "warnings": warnings, "blur": round(blur, 1),
            "brightness": round(brightness, 1), "size": (w, h)}


def save_upload(image_bytes):
    """Validates the bytes as an image, stores it as RGB JPEG and returns (path, sha1_prefix)."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise ValueError("The file is not a readable image.") from e
    digest = hashlib.sha1(image_bytes).hexdigest()[:12]
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, f"{digest}.jpg")
    img.save(path, quality=92)
    return path, digest


def top1_label(result):
    """Top-1 breed from a decision-layer result, whichever shape it has."""
    if result.get("status") == "auto_accepted":
        return result["breed"]
    return result["shortlist"][0]["breed"]


def decide_source(result, final_label, manual=False):
    """Maps the worker's action to the 'source' values documented in feedback_loop.py."""
    if manual:
        return "manual_entry"
    top1 = top1_label(result)
    if result.get("status") == "auto_accepted":
        return "auto_accepted" if final_label == top1 else "worker_overrode"
    return "worker_confirmed" if final_label == top1 else "worker_overrode"


def confirm_prediction(image_path, result, prob_map, final_label, animal_id, worker_id,
                       village=None, manual=False):
    """Logs the final human-verified label to the feedback log and syncs it to BPA (mocked).
    Returns {"source", "confidence", "feedback_record", "bpa_response"}."""
    final_label = (final_label or "").strip()
    animal_id = (animal_id or "").strip()
    if not final_label:
        raise ValueError("Please choose or enter a breed before confirming.")
    if not animal_id:
        raise ValueError("Please enter the animal ID / ear-tag number before confirming.")

    source = decide_source(result, final_label, manual)
    confidence = float(prob_map.get(final_label, 0.0))   # 0.0 for a manually typed unknown breed
    record = log_feedback(image_path, result, final_label, source, worker_id=(worker_id or "unknown"))
    response = sync_breed_to_bpa(animal_id=animal_id, breed=final_label, confidence=round(confidence, 4),
                                 flw_id=(worker_id or "unknown"), farmer_village=(village or None))
    return {"source": source, "confidence": confidence, "feedback_record": record, "bpa_response": response}


def _read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue   # ignore a half-written line instead of crashing the UI
    return rows


def read_feedback_df():
    rows = []
    for r in _read_jsonl(config.FEEDBACK_LOG_PATH):
        pred = r.get("model_prediction", {})
        try:
            model_top1 = top1_label(pred)
        except (KeyError, IndexError):
            model_top1 = ""
        rows.append({
            "time (UTC)": r.get("timestamp", ""),
            "worker": r.get("worker_id", ""),
            "model top-1": model_top1,
            "model status": pred.get("status", ""),
            "final label": r.get("final_label", ""),
            "source": r.get("source", ""),
            "image": os.path.basename(r.get("image_path", "")),
        })
    cols = ["time (UTC)", "worker", "model top-1", "model status", "final label", "source", "image"]
    return pd.DataFrame(rows, columns=cols)


def read_bpa_df():
    rows = _read_jsonl(MOCK_BPA_DB_PATH)
    cols = ["animal_id", "breed", "model_confidence", "confirmed_by", "farmer_village", "synced_at"]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows], columns=cols)


def feedback_summary():
    return summarize_feedback()
