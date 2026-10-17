"""
feedback_loop.py
---------------------------------
Implements objective 7 / the "Verified labels fed back for periodic model
retraining" arrow in the block diagram.

Every time a field worker accepts, overrides, or manually enters a breed
label, that event is appended to logs/feedback_log.jsonl. A separate
retraining job can later replay this log to fine-tune the model on real,
field-verified corrections -- exactly the loop most academic prototypes stop
short of.
"""

import json
import os
import time

import config


def log_feedback(image_path, model_prediction, final_label, source, worker_id="unknown"):
    """
    image_path       : path (or storage key) of the captured image
    model_prediction : dict returned by BreedClassifier.predict()
    final_label       : the breed the FLW actually confirmed
    source            : "auto_accepted" | "worker_confirmed" | "worker_overrode" | "manual_entry"
    worker_id         : FLW identifier (anonymised / hashed in a real deployment)
    """
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image_path": image_path,
        "model_prediction": model_prediction,
        "final_label": final_label,
        "source": source,
        "worker_id": worker_id,
    }
    with open(config.FEEDBACK_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def summarize_feedback():
    """Quick health-check of the feedback log: how often the model's top-1
    guess actually matched what the worker confirmed -- a real-world accuracy
    signal, independent of the offline test-set accuracy."""
    if not os.path.exists(config.FEEDBACK_LOG_PATH):
        return {"total": 0, "agreement_rate": None}

    total, agreed = 0, 0
    with open(config.FEEDBACK_LOG_PATH) as f:
        for line in f:
            rec = json.loads(line)
            total += 1
            pred = rec["model_prediction"]
            top1 = pred.get("breed") or (pred.get("shortlist", [{}])[0].get("breed"))
            if top1 == rec["final_label"]:
                agreed += 1

    return {
        "total": total,
        "agreement_rate": round(agreed / total, 3) if total else None,
    }


if __name__ == "__main__":
    # Small self-contained demo: simulate a few field confirmations
    demo_events = [
        ("field_photo_1.jpg", {"status": "auto_accepted", "breed": "Gir", "confidence": 0.91},
         "Gir", "auto_accepted"),
        ("field_photo_2.jpg", {"status": "needs_review",
                                "shortlist": [{"breed": "Murrah_Buffalo", "confidence": 0.55},
                                              {"breed": "Jaffrabadi_Buffalo", "confidence": 0.30}]},
         "Jaffrabadi_Buffalo", "worker_overrode"),
        ("field_photo_3.jpg", {"status": "auto_accepted", "breed": "Sahiwal", "confidence": 0.85},
         "Sahiwal", "worker_confirmed"),
    ]
    for image_path, pred, final_label, source in demo_events:
        log_feedback(image_path, pred, final_label, source, worker_id="FLW_1042")

    print("Logged", len(demo_events), "feedback events ->", config.FEEDBACK_LOG_PATH)
    print("Summary:", summarize_feedback())
