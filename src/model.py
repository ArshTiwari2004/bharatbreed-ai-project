"""
model.py
---------------------------------
Builds the CNN classifier: a pretrained ResNet50 / EfficientNetB0 backbone
(transfer learning) + a small classification head, exactly as described in
section 4.2 / 4.3 of the synopsis.

Note on offline environments: downloading ImageNet weights requires internet
access to Keras' weight-file host. If that host is unreachable (e.g. a
network-restricted sandbox / CI runner), we fall back to random initialisation
so the pipeline still runs end-to-end instead of crashing. On a normal
machine / Colab with internet access, `weights="imagenet"` is used
automatically and gives the real transfer-learning benefit described in the
synopsis.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.saving import register_keras_serializable

import config


@register_keras_serializable(package="bharatbreed")
class PreprocessLayer(layers.Layer):
    """Wraps a keras.applications preprocess_input function as a proper,
    serialisable Keras layer (a plain Lambda referencing an external function
    cannot be safely saved/reloaded in the modern .keras format)."""

    def __init__(self, backbone_name, **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        if backbone_name == "efficientnet":
            self._fn = tf.keras.applications.efficientnet.preprocess_input
        elif backbone_name == "resnet50":
            self._fn = tf.keras.applications.resnet50.preprocess_input
        else:
            raise ValueError(f"Unknown backbone_name '{backbone_name}'")

    def call(self, inputs):
        return self._fn(inputs)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"backbone_name": self.backbone_name})
        return cfg


def _load_backbone(input_shape):
    common_kwargs = dict(include_top=False, input_shape=input_shape, pooling="avg")

    if config.BACKBONE == "efficientnet":
        backbone_cls = tf.keras.applications.EfficientNetB0
    elif config.BACKBONE == "resnet50":
        backbone_cls = tf.keras.applications.ResNet50
    else:
        raise ValueError(f"Unknown BACKBONE '{config.BACKBONE}'")

    try:
        backbone = backbone_cls(weights="imagenet", **common_kwargs)
        print(f"[model] Loaded {config.BACKBONE} with pretrained ImageNet weights.")
    except Exception as e:  # noqa: BLE001 - deliberate broad catch for offline fallback
        print(f"[model] Could not download ImageNet weights ({e}). "
              f"Falling back to random initialisation for this run only.")
        backbone = backbone_cls(weights=None, **common_kwargs)

    return backbone


def build_model(num_classes, input_shape=None, freeze_backbone=True):
    input_shape = input_shape or (*config.IMAGE_SIZE, 3)
    backbone = _load_backbone(input_shape)
    backbone.trainable = not freeze_backbone

    inputs = layers.Input(shape=input_shape)
    x = PreprocessLayer(config.BACKBONE, name="preprocess")(inputs)
    x = backbone(x, training=False if freeze_backbone else None)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="breed_probs")(x)

    model = models.Model(inputs, outputs, name=f"bharatbreed_{config.BACKBONE}")
    return model, backbone


def unfreeze_for_fine_tuning(model, backbone):
    """Unfreezes the last N layers of the backbone for fine-tuning (stage 2)."""
    backbone.trainable = True
    freeze_until = max(0, len(backbone.layers) - config.FINE_TUNE_AT_LAYERS_FROM_END)
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
    return model
