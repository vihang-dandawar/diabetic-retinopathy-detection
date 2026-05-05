import os
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
import config as cfg
from model_factory import build_production_model

# 1. GPU Memory Config - Optimized for 2GB T1000
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Keep 1800MB limit to prevent Windows from crashing the display driver
        tf.config.set_logical_device_configuration(
            gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=1800)])
        print(">>> GPU Memory configured for stability.")
    except RuntimeError as e: print(e)

if cfg.MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

# 2. Focal Loss:
def categorical_focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        loss = -y_true * (tf.pow(1.0 - y_pred, gamma)) * tf.math.log(y_pred)
        return tf.reduce_sum(alpha * loss, axis=-1)
    return focal_loss

def load_data():
    processed_train = os.path.join(cfg.BASE_DIR, "data_processed/train")
    processed_val = os.path.join(cfg.BASE_DIR, "data_processed/val")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        processed_train, image_size=(224, 224), batch_size=cfg.BATCH_SIZE, label_mode='categorical')
    val_ds = tf.keras.utils.image_dataset_from_directory(
        processed_val, image_size=(224, 224), batch_size=cfg.BATCH_SIZE, label_mode='categorical')
    
    # repeat() is necessary because we use steps_per_epoch
    return train_ds.repeat().prefetch(tf.data.AUTOTUNE), val_ds.repeat().prefetch(tf.data.AUTOTUNE)

def run_training():
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.BACKUP_DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.BASE_DIR, "models"), exist_ok=True)

    model_path = os.path.join(cfg.BASE_DIR, "models/dr_model_final_85.keras")
    train_ds, val_ds = load_data()

    if os.path.exists(model_path):
        print(f">>> Loading existing model: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Ensure Phase 2 settings: Unfrozen base, locked BatchNorm
        for layer in model.layers:
            if 'mobilenetv2' in layer.name.lower():
                layer.trainable = True
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                        sub_layer.trainable = False
        current_lr = 1e-5
        print(">>> Fine-tuning mode active (Full Model Unfrozen).")
    else:
        print(">>> Starting Phase 1 (Frozen Base)")
        model = build_production_model(cfg, fine_tune=False)
        current_lr = 1e-4

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=current_lr),
        loss=categorical_focal_loss(),
        metrics=["accuracy"]
    )

    # --- PRODUCTION CALLBACKS (OVERNIGHT OPTIMIZED) ---
    callbacks =[
        # Resumes exactly where you left off if power cuts
        tf.keras.callbacks.BackupAndRestore(backup_dir=cfg.BACKUP_DIR),
        
        # MANDATORY: Only saves the absolute best accuracy found all night
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        
        tf.keras.callbacks.CSVLogger(os.path.join(cfg.LOG_DIR, "final_processed_log.csv"), append=True),
        
        # Slower LR reduction to allow deep exploration
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-8, verbose=1),
        
        # HIGH PATIENCE: Allows the model to train all night to reach 85%
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=100, restore_best_weights=True, verbose=1)
    ]

    print(f"Training session started. Target: 85% Accuracy.")
    print(f"Updates every 1500 steps. Batch Size: {cfg.BATCH_SIZE}")
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=200,            
        steps_per_epoch=1500,  
        validation_steps=400,  
        callbacks=callbacks
    )

if __name__ == "__main__":
    run_training()