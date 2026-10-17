"""
data_prep.py
---------------------------------
Loads images from data/raw/<breed_name>/*.jpg, splits into train / val / test,
and builds tf.data pipelines with the augmentation described in the synopsis
(random crop, brightness/contrast jitter, horizontal flip) to compensate for
a small, India-specific dataset.
"""

import json
import os

import tensorflow as tf

import config


def _list_image_paths_and_labels():
    class_names = sorted(
        d for d in os.listdir(config.RAW_DATA_DIR)
        if os.path.isdir(os.path.join(config.RAW_DATA_DIR, d))
    )
    if not class_names:
        raise RuntimeError(
            f"No class folders found in {config.RAW_DATA_DIR}. "
            "Run `python src/generate_sample_dataset.py` for a demo dataset, "
            "or place real breed photos in data/raw/<breed_name>/."
        )

    paths, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(config.RAW_DATA_DIR, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)

    if len(paths) < 10:
        raise RuntimeError("Fewer than 10 images found in total -- dataset looks empty/corrupt.")

    return paths, labels, class_names


def _decode_and_resize(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, config.IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label


def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    # random crop-and-resize (zoom jitter)
    pad = 20
    img = tf.image.resize_with_crop_or_pad(
        img, config.IMAGE_SIZE[0] + pad, config.IMAGE_SIZE[1] + pad
    )
    img = tf.image.random_crop(img, size=[*config.IMAGE_SIZE, 3])
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


def build_datasets():
    """Returns (train_ds, val_ds, test_ds, class_names)."""
    paths, labels, class_names = _list_image_paths_and_labels()

    # Deterministic shuffle + split (stratified-ish via global shuffle; dataset
    # is small and roughly balanced by construction of generate_sample_dataset.py)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(paths), seed=config.RANDOM_SEED, reshuffle_each_iteration=False)

    n = len(paths)
    n_test = max(1, int(n * config.TEST_SPLIT))
    n_val = max(1, int(n * config.VALIDATION_SPLIT))
    n_train = n - n_val - n_test

    train_ds = ds.take(n_train)
    remaining = ds.skip(n_train)
    val_ds = remaining.take(n_val)
    test_ds = remaining.skip(n_val)

    train_ds = (
        train_ds.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(512, seed=config.RANDOM_SEED)
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        test_ds.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.LABEL_MAP_PATH, "w") as f:
        json.dump({str(i): name for i, name in enumerate(class_names)}, f, indent=2)

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Images: {n} total -> train {n_train} / val {n_val} / test {n_test}")

    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    build_datasets()
