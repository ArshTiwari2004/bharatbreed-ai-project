"""
test_pipeline.py
---------------------------------
Lightweight sanity tests -- not a full unit-test suite, but enough to catch
the classic "it works on my machine" failure modes before a demo:
  - dataset folders exist and are non-empty
  - a trained Keras model can be loaded
  - the TFLite model produces a valid probability distribution
  - the confidence-based decision layer returns one of the two expected shapes
  - the feedback log and BPA mock sync both write successfully

Run:
    python -m pytest tests/ -v
    (or simply: python tests/test_pipeline.py)
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import config  # noqa: E402


def test_dataset_exists():
    assert os.path.isdir(config.RAW_DATA_DIR), "Run generate_sample_dataset.py first"
    classes = [d for d in os.listdir(config.RAW_DATA_DIR)
               if os.path.isdir(os.path.join(config.RAW_DATA_DIR, d))]
    assert len(classes) >= 2, "Need at least 2 breed classes"
    for c in classes:
        files = os.listdir(os.path.join(config.RAW_DATA_DIR, c))
        assert len(files) > 0, f"No images found for class {c}"


def test_label_map_matches_dataset():
    assert os.path.exists(config.LABEL_MAP_PATH), "Run train.py first"
    with open(config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    classes_on_disk = sorted(
        d for d in os.listdir(config.RAW_DATA_DIR)
        if os.path.isdir(os.path.join(config.RAW_DATA_DIR, d))
    )
    assert sorted(label_map.values()) == classes_on_disk


def test_tflite_model_outputs_valid_distribution():
    assert os.path.exists(config.TFLITE_MODEL_PATH), "Run convert_tflite.py first"
    from inference import BreedClassifier

    clf = BreedClassifier()
    sample_class = os.listdir(config.RAW_DATA_DIR)[0]
    sample_dir = os.path.join(config.RAW_DATA_DIR, sample_class)
    sample_img = os.path.join(sample_dir, os.listdir(sample_dir)[0])

    probs = clf.predict_proba(sample_img)
    assert probs.shape[0] == len(clf.labels)
    assert np.isclose(probs.sum(), 1.0, atol=1e-3), "Softmax output should sum to ~1"
    assert (probs >= 0).all() and (probs <= 1).all()


def test_confidence_decision_layer_shape():
    from inference import BreedClassifier

    clf = BreedClassifier()
    sample_class = os.listdir(config.RAW_DATA_DIR)[0]
    sample_dir = os.path.join(config.RAW_DATA_DIR, sample_class)
    sample_img = os.path.join(sample_dir, os.listdir(sample_dir)[0])

    result = clf.predict(sample_img)
    assert result["status"] in ("auto_accepted", "needs_review")
    if result["status"] == "auto_accepted":
        assert "breed" in result and "confidence" in result
    else:
        assert "shortlist" in result and len(result["shortlist"]) > 0


def test_feedback_logging_roundtrip():
    from feedback_loop import log_feedback, summarize_feedback

    before = summarize_feedback()["total"] or 0
    log_feedback("test_image.jpg", {"status": "auto_accepted", "breed": "Gir", "confidence": 0.8},
                 "Gir", "worker_confirmed", worker_id="TEST_WORKER")
    after = summarize_feedback()["total"]
    assert after == before + 1


def test_bpa_mock_sync():
    from bpa_sync_stub import sync_breed_to_bpa

    resp = sync_breed_to_bpa("TEST-ANIMAL", "Gir", 0.8, "TEST_WORKER")
    assert resp["status"] == 200
    assert resp["synced"] is True


if __name__ == "__main__":
    tests = [
        test_dataset_exists,
        test_label_map_matches_dataset,
        test_tflite_model_outputs_valid_distribution,
        test_confidence_decision_layer_shape,
        test_feedback_logging_roundtrip,
        test_bpa_mock_sync,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
