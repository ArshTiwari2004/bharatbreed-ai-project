"""
convert_tflite.py
---------------------------------
Converts the trained Keras model into TensorFlow Lite format for on-device /
edge inference on the low-to-mid-range Android phones used by Field Level
Workers (objective 4 / module "Mobile Optimisation & Deployment" in the
synopsis).

Produces two artifacts:
  - bharatbreed_model.tflite       (float32, full precision)
  - bharatbreed_model_int8.tflite  (dynamic-range quantized, ~4x smaller,
                                     faster on-device, small accuracy trade-off)

Run:
    python src/convert_tflite.py
"""

import logging
import os
import time

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)  # keep the SavedModel/TFLite conversion logs from drowning out results

import config
import model  # noqa: F401 - required so the custom PreprocessLayer is registered before model load


def convert(quantize: bool):
    model = tf.keras.models.load_model(config.KERAS_MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range quantization
        out_path = config.TFLITE_QUANT_MODEL_PATH
    else:
        out_path = config.TFLITE_MODEL_PATH

    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(out_path) / 1024
    return out_path, size_kb


def benchmark_latency(tflite_path, n_runs=20):
    """Rough single-image inference latency benchmark (stand-in for on-device profiling)."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    import numpy as np
    dummy = np.random.uniform(0, 255, size=input_details[0]["shape"]).astype(input_details[0]["dtype"])

    # warm-up
    interpreter.set_tensor(input_details[0]["index"], dummy)
    interpreter.invoke()

    start = time.time()
    for _ in range(n_runs):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
    elapsed = (time.time() - start) / n_runs * 1000
    return elapsed  # ms/inference on THIS machine (CPU) -- indicative only


if __name__ == "__main__":
    if not os.path.exists(config.KERAS_MODEL_PATH):
        raise SystemExit(f"No trained model found at {config.KERAS_MODEL_PATH}. Run `python src/train.py` first.")

    fp32_path, fp32_size = convert(quantize=False)
    print(f"[ok] Float32 TFLite model -> {fp32_path} ({fp32_size:.1f} KB)")
    fp32_latency = benchmark_latency(fp32_path)
    print(f"     avg inference latency (this machine): {fp32_latency:.2f} ms/image")

    int8_path, int8_size = convert(quantize=True)
    print(f"[ok] Quantized TFLite model -> {int8_path} ({int8_size:.1f} KB)")
    int8_latency = benchmark_latency(int8_path)
    print(f"     avg inference latency (this machine): {int8_latency:.2f} ms/image")

    print(f"\nSize reduction from quantization: {fp32_size / int8_size:.1f}x smaller")
