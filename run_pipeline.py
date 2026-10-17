#!/usr/bin/env python3
"""
run_pipeline.py
---------------------------------
Runs the ENTIRE BharatBreed AI pipeline end-to-end in one command, exactly
the way a mentor / evaluator would want to see it during a demo:

    1. Generate the sample dataset (skipped if data/raw already has images)
    2. Train the model (transfer learning, two-stage)
    3. Convert to TFLite (float32 + quantized)
    4. Run inference on a few sample images (confidence-aware decision)
    5. Log simulated field-worker feedback
    6. Sync a confirmed record to the (mocked) BPA endpoint

Run:
    python run_pipeline.py
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config  # noqa: E402


def _step(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    _step("STEP 1/6 - Preparing dataset")
    has_class_folders = os.path.isdir(config.RAW_DATA_DIR) and any(
        os.path.isdir(os.path.join(config.RAW_DATA_DIR, d))
        for d in os.listdir(config.RAW_DATA_DIR)
    )
    if not has_class_folders:
        subprocess.run([sys.executable, "src/generate_sample_dataset.py"], check=True)
    else:
        print(f"Found existing class folders in {config.RAW_DATA_DIR}, skipping generation.")

    _step("STEP 2/6 - Training model (transfer learning)")
    subprocess.run([sys.executable, "src/train.py"], check=True)

    _step("STEP 3/6 - Converting to TFLite for mobile/edge deployment")
    subprocess.run([sys.executable, "src/convert_tflite.py"], check=True)

    _step("STEP 4/6 - Running confidence-aware inference on sample images")
    from inference import BreedClassifier
    clf = BreedClassifier()
    sample_dir = os.path.join(config.RAW_DATA_DIR, config.SAMPLE_BREEDS[0])
    sample_img = os.path.join(sample_dir, os.listdir(sample_dir)[0])
    result = clf.predict(sample_img)
    print(f"Prediction for {sample_img}:\n{result}")

    _step("STEP 5/6 - Logging field-worker feedback")
    from feedback_loop import log_feedback, summarize_feedback
    final_label = result.get("breed") or result["shortlist"][0]["breed"]
    log_feedback(sample_img, result, final_label, source="worker_confirmed", worker_id="FLW_DEMO")
    print("Feedback summary:", summarize_feedback())

    _step("STEP 6/6 - Syncing confirmed record to (mocked) BPA")
    from bpa_sync_stub import sync_breed_to_bpa
    resp = sync_breed_to_bpa(
        animal_id="DEMO-ANIMAL-001",
        breed=final_label,
        confidence=result.get("confidence", 0.0),
        flw_id="FLW_DEMO",
    )
    print("BPA sync response:", resp)

    _step("PIPELINE COMPLETE")
    print(f"Trained model : {config.KERAS_MODEL_PATH}")
    print(f"TFLite model  : {config.TFLITE_MODEL_PATH}")
    print(f"Quantized     : {config.TFLITE_QUANT_MODEL_PATH}")
    print(f"Feedback log  : {config.FEEDBACK_LOG_PATH}")


if __name__ == "__main__":
    main()
