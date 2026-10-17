"""
inference.py
---------------------------------
The "confidence-based decision layer" from the synopsis (section 4.2):
  - Loads the TFLite model (the artifact an Android app would actually ship)
  - Runs a single image through it
  - If top-1 confidence >= CONFIDENCE_THRESHOLD -> auto-accept that breed
  - Otherwise -> return a ranked top-k shortlist and ask for human confirmation

Usable as a library (`predict_breed(path)`) or from the command line:
    python src/inference.py path/to/image.jpg
"""

import json
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

import config


def _load_label_map():
    with open(config.LABEL_MAP_PATH) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _preprocess_image(path):
    img = Image.open(path).convert("RGB").resize(config.IMAGE_SIZE)
    arr = np.array(img).astype(np.float32)
    return np.expand_dims(arr, axis=0)


class BreedClassifier:
    """Thin wrapper around the TFLite interpreter -- this is what would run
    inside the mobile app / edge service."""

    def __init__(self, tflite_path=None, label_map_path=None):
        tflite_path = tflite_path or config.TFLITE_MODEL_PATH
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = _load_label_map()

    def predict_proba(self, image_path):
        x = _preprocess_image(image_path)
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        return probs

    def predict(self, image_path, threshold=None, top_k=None):
        """Returns a dict describing either an auto-accepted breed or a
        low-confidence shortlist, mirroring the block diagram's decision node."""
        threshold = config.CONFIDENCE_THRESHOLD if threshold is None else threshold
        top_k = config.TOP_K if top_k is None else top_k

        probs = self.predict_proba(image_path)
        order = np.argsort(probs)[::-1]
        ranked = [{"breed": self.labels[i], "confidence": round(float(probs[i]), 4)} for i in order]

        top1 = ranked[0]
        if top1["confidence"] >= threshold:
            return {
                "status": "auto_accepted",
                "breed": top1["breed"],
                "confidence": top1["confidence"],
                "message": f"High confidence ({top1['confidence']:.0%}) -> label auto-assigned.",
            }
        else:
            return {
                "status": "needs_review",
                "shortlist": ranked[:top_k],
                "message": (
                    f"Confidence below threshold ({threshold:.0%}). "
                    "Showing top candidates for the field worker to confirm or override."
                ),
            }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference.py <path_to_image>")
        sys.exit(1)

    clf = BreedClassifier()
    result = clf.predict(sys.argv[1])
    print(json.dumps(result, indent=2))
