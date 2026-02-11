import tensorflow as tf
from tensorflow.keras import layers, models

def build_production_model(config):

    # ============================================================
    # Enable Mixed Precision (GPU optimization)
    # ============================================================

    if config.MIXED_PRECISION:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')

    # ============================================================
    # Load MobileNetV2 pretrained base
    # ============================================================

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=config.IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )

    # ============================================================
    # PHASE 1: Freeze entire base model (CRITICAL FIX)
    # ============================================================

    base_model.trainable = False

    # ============================================================
    # Input Layer
    # ============================================================

    inputs = tf.keras.Input(shape=config.IMG_SIZE + (3,))

    # ============================================================
    # GPU-based Data Augmentation
    # ============================================================

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    # ============================================================
    # MobileNetV2 preprocessing
    # ============================================================

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # IMPORTANT: training=False keeps BatchNorm stable
    x = base_model(x, training=False)

    # ============================================================
    # Classification Head
    # ============================================================

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # IMPORTANT: float32 output for mixed precision stability
    outputs = layers.Dense(
        config.NUM_CLASSES,
        activation='softmax',
        dtype='float32'
    )(x)

    # ============================================================
    # Build Model
    # ============================================================

    model = models.Model(inputs, outputs)

    return model
