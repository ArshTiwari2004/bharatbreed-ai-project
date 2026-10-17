"""
generate_sample_dataset.py
---------------------------------
Generates a small SYNTHETIC image dataset so the entire training / inference /
TFLite-export / feedback-loop pipeline can be run and verified end-to-end on a
laptop with no internet access and no real breed photographs.

This is a stand-in only. For the real project, replace the contents of
data/raw/<breed_name>/ with actual photographs (e.g. the public "Indian
Bovine Breeds" dataset on Kaggle, ICAR repository images, or field-collected
photos), keeping the same folder-per-class structure. Nothing else in the
pipeline needs to change.

Run:
    python src/generate_sample_dataset.py
"""

import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import config

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

# Each "breed" gets a distinct base colour + shape motif + horn/marking style,
# loosely standing in for coat colour, body shape and horn curvature -- the
# real visual cues described in the synopsis -- so that a real CNN has
# something learnable to separate classes on.
BREED_VISUAL_SIGNATURE = {
    "Gir":                 {"base": (176, 124, 60),  "spot": (90, 55, 25),   "shape": "convex_forehead"},
    "Sahiwal":              {"base": (168, 96, 60),   "spot": (120, 70, 40),  "shape": "loose_skin"},
    "Murrah_Buffalo":       {"base": (35, 35, 38),    "spot": (10, 10, 12),   "shape": "curled_horn"},
    "Tharparkar":           {"base": (225, 222, 210), "spot": (190, 188, 178),"shape": "lyre_horn"},
    "Red_Sindhi":           {"base": (150, 45, 35),   "spot": (100, 25, 20),  "shape": "compact_body"},
    "Jaffrabadi_Buffalo":   {"base": (25, 25, 30),    "spot": (60, 60, 65),   "shape": "drooping_horn"},
}


def _draw_animal_like_blob(draw, w, h, base_color, spot_color, shape_key, rng):
    """Draws a crude blob + horn-like motif so classes are visually separable."""
    cx, cy = w // 2 + rng.randint(-10, 10), h // 2 + rng.randint(-10, 10)
    body_w, body_h = rng.randint(70, 95), rng.randint(45, 65)

    draw.ellipse(
        [cx - body_w, cy - body_h, cx + body_w, cy + body_h],
        fill=base_color,
    )

    # Random coat texture / spots
    for _ in range(rng.randint(6, 14)):
        sx = cx + rng.randint(-body_w, body_w)
        sy = cy + rng.randint(-body_h, body_h)
        r = rng.randint(4, 12)
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=spot_color)

    # Horn / head motif, shape depends on breed signature
    hx, hy = cx - body_w + 15, cy - body_h - 5
    if shape_key in ("curled_horn", "drooping_horn"):
        draw.arc([hx - 20, hy - 30, hx + 20, hy + 10], start=0, end=250, fill=(15, 15, 15), width=4)
    elif shape_key == "lyre_horn":
        draw.line([hx, hy, hx - 15, hy - 25], fill=(40, 40, 40), width=4)
        draw.line([hx, hy, hx + 15, hy - 25], fill=(40, 40, 40), width=4)
    elif shape_key == "convex_forehead":
        draw.ellipse([hx - 10, hy - 20, hx + 25, hy + 5], fill=base_color, outline=(80, 50, 20))
    else:
        draw.line([hx, hy, hx - 10, hy - 15], fill=(50, 50, 50), width=3)


def make_image(breed, rng, img_size=(300, 300)):
    w, h = img_size
    bg = (rng.randint(150, 210), rng.randint(170, 220), rng.randint(140, 190))  # grassy/sky-ish background
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    sig = BREED_VISUAL_SIGNATURE[breed]
    base = tuple(min(255, max(0, c + rng.randint(-15, 15))) for c in sig["base"])
    spot = tuple(min(255, max(0, c + rng.randint(-10, 10))) for c in sig["spot"])
    _draw_animal_like_blob(draw, w, h, base, spot, sig["shape"], rng)

    # Simulate rural field-capture noise: blur, brightness jitter, JPEG-ish noise
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.8)))
    arr = np.array(img).astype(np.int16)
    brightness = rng.randint(-30, 30)
    arr = np.clip(arr + brightness, 0, 255).astype(np.uint8)
    noise = np.random.normal(0, 6, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    return img


def main():
    rng = random.Random(config.RANDOM_SEED)
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

    total = 0
    for breed in config.SAMPLE_BREEDS:
        breed_dir = os.path.join(config.RAW_DATA_DIR, breed)
        os.makedirs(breed_dir, exist_ok=True)
        for i in range(config.IMAGES_PER_CLASS_SAMPLE):
            img = make_image(breed, rng)
            img.save(os.path.join(breed_dir, f"{breed}_{i:03d}.jpg"), quality=90)
            total += 1
        print(f"[ok] {breed}: {config.IMAGES_PER_CLASS_SAMPLE} images -> {breed_dir}")

    print(f"\nDone. {total} synthetic placeholder images written to {config.RAW_DATA_DIR}")
    print("Replace these folders with real breed photographs before real training.")


if __name__ == "__main__":
    main()
