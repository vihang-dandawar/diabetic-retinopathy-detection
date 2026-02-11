import os
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
import config as cfg
from model_factory import build_production_model
# Enable mixed precision (production GPU optimization)
import config as cfg
if cfg.MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

# 1. GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e: print(e)

# 2. Optimized Data Loading
def load_data():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.TRAIN_DIR,
        image_size=cfg.IMG_SIZE,
        batch_size=cfg.BATCH_SIZE,
        label_mode='int'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.VAL_DIR,
        image_size=cfg.IMG_SIZE,
        batch_size=cfg.BATCH_SIZE,
        label_mode='int'
    )

    # Safe for large datasets
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds


# 3. Training Execution
def run_training():

    # Fix: create directories automatically
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.BACKUP_DIR, exist_ok=True)

    train_ds, val_ds = load_data()

    print("Calculating weights for imbalance...")

    y_train = np.concatenate([y for x, y in train_ds.take(200)], axis=0)

    cw = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    cw_dict = dict(enumerate(cw))

    model = build_production_model(cfg)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.INITIAL_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.BackupAndRestore(backup_dir=cfg.BACKUP_DIR),

        tf.keras.callbacks.ModelCheckpoint(
            cfg.FINAL_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=cfg.MIN_LR,
            verbose=1
        ),

        tf.keras.callbacks.CSVLogger(
            os.path.join(cfg.LOG_DIR, "training.csv"),
            append=True
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(cfg.CHECKPOINT_DIR, "epoch_{epoch:03d}.keras"),
            save_freq='epoch'
        )
    ]

    print("--- Starting Production Training ---")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.EPOCHS,
        callbacks=callbacks,
        class_weight=cw_dict
    )

if __name__ == "__main__":
    run_training()