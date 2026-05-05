import tensorflow as tf
from tensorflow.keras import layers, models

def build_production_model(config, fine_tune=False):
    if config.MIXED_PRECISION:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')

    # 1. Input is now exactly 224x224 from the processed folder
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # 2. Mild Augmentation (Medical Safe)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)

    # 3. MobileNet Preprocessing (Scaling only)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # 4. Load Base Model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    if fine_tune:
        base_model.trainable = True
        # BatchNorm Protection is still critical for Batch Size 16
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        print(">>> Model Factory: Base Model UNFROZEN (BN Protected)")
    else:
        base_model.trainable = False
        print(">>> Model Factory: Base Model FROZEN")

    # 5. Build the Architecture
    x = base_model(x, training=fine_tune)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    # Optimized Head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(config.NUM_CLASSES, activation='softmax', dtype='float32')(x)

    return models.Model(inputs, outputs)