"""
Central configuration for BharatBreed AI.
Change values here rather than hunting through scripts.
"""

import os

# Quiet down TensorFlow's C++ backend logging (INFO/WARNING spam) before TF is imported anywhere.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")          # data/raw/<breed_name>/*.jpg
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "bharatbreed_model.keras")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "bharatbreed_model.tflite")
TFLITE_QUANT_MODEL_PATH = os.path.join(MODELS_DIR, "bharatbreed_model_int8.tflite")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")
FEEDBACK_LOG_PATH = os.path.join(LOGS_DIR, "feedback_log.jsonl")
TRAINING_HISTORY_PATH = os.path.join(LOGS_DIR, "training_history.json")

# ---------------------------------------------------------------------------
# Image / training hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = (160, 160)          # small enough for fast CPU demo, large enough for real transfer learning
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Backbone: "efficientnet" or "resnet50"
BACKBONE = "efficientnet"
EPOCHS_HEAD = 3          # epochs training only the new classification head
EPOCHS_FINE_TUNE = 2     # epochs fine-tuning the top backbone layers
FINE_TUNE_AT_LAYERS_FROM_END = 20   # unfreeze last N layers of backbone during fine-tuning
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5

# ---------------------------------------------------------------------------
# Confidence-aware decision layer
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.70      # >= threshold -> auto-accept
TOP_K = 3                        # shortlist size shown when below threshold

# ---------------------------------------------------------------------------
# Sample / synthetic dataset (used only for the offline demo pipeline)
# ---------------------------------------------------------------------------
SAMPLE_BREEDS = [
    "Gir",
    "Sahiwal",
    "Murrah_Buffalo",
    "Tharparkar",
    "Red_Sindhi",
    "Jaffrabadi_Buffalo",
]
IMAGES_PER_CLASS_SAMPLE = 60
