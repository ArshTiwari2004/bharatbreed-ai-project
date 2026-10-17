"""
train.py
---------------------------------
End-to-end training entry point:
  1. Build train/val/test tf.data pipelines
  2. Build a transfer-learning model (frozen backbone)
  3. Train the classification head
  4. Unfreeze top backbone layers and fine-tune at a low LR
  5. Evaluate on the held-out test split
  6. Save the Keras model + label map + training history

Run:
    python src/train.py
"""

import json
import os
import time

import tensorflow as tf

import config
from data_prep import build_datasets
from model import build_model, unfreeze_for_fine_tuning


def main():
    t0 = time.time()
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = build_datasets()
    num_classes = len(class_names)

    model, backbone = build_model(num_classes, freeze_backbone=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE_HEAD),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\n=== Stage 1: training classification head (backbone frozen) ===")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_HEAD,
        verbose=2,
    )

    print("\n=== Stage 2: fine-tuning top backbone layers ===")
    model = unfreeze_for_fine_tuning(model, backbone)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE_FINE_TUNE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FINE_TUNE,
        verbose=2,
    )

    print("\n=== Evaluating on held-out test split ===")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    model.save(config.KERAS_MODEL_PATH)
    print(f"Saved trained model -> {config.KERAS_MODEL_PATH}")

    history = {
        "backbone": config.BACKBONE,
        "classes": class_names,
        "stage1_head": {k: [float(v) for v in vals] for k, vals in history_head.history.items()},
        "stage2_fine_tune": {k: [float(v) for v in vals] for k, vals in history_fine.history.items()},
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(config.TRAINING_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history -> {config.TRAINING_HISTORY_PATH}")
    print(f"\nTotal time: {history['train_seconds']}s")


if __name__ == "__main__":
    main()
