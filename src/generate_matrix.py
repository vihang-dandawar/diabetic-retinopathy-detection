import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import config as cfg

# ============================================================
# 1. ATTEMPT TO IMPORT LEGACY BRIDGE (REQUIRED FOR LAPTOP)
# ============================================================
try:
    import tf_keras as legacy_keras
    print("✅ Legacy Bridge (tf_keras) active for evaluation.")
except ImportError:
    print("❌ Error: tf_keras not found. Please run: pip install tf_keras --user")
    import sys
    sys.exit()

# 2. GPU Memory Setup (Strict 1800MB limit for stability)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_logical_device_configuration(
            gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=1800)])
    except: pass

# 3. Custom loss placeholder for loading
@tf.keras.utils.register_keras_serializable()
def focal_loss(y_true, y_pred):
    return tf.reduce_sum(y_true, axis=-1)

def run_evaluation():
    # Define Class Names directly
    CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    
    # Use relative path for stability on Windows
    model_path = "models/dr_model_final_85.keras"
    
    # If running from inside 'src', go up one level
    if not os.path.exists(model_path):
        model_path = os.path.join("..", model_path)

    test_dir = os.path.join(cfg.BASE_DIR, "data_processed", "test")
    
    print(f"--- Starting Final Evaluation ---")
    print(f"Loading Model via Bridge: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return

    # USE THE LEGACY BRIDGE TO LOAD THE MODEL
    model = legacy_keras.models.load_model(
        model_path, 
        custom_objects={'focal_loss': focal_loss},
        compile=False
    )
    print("✅ Model loaded successfully!")

    # Load Test Dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(224, 224),
        batch_size=16,
        label_mode='categorical',
        shuffle=False 
    )

    y_true = []
    y_pred = []

    print("Extracting predictions (this may take 2-3 minutes on CPU)...")
    for images, labels in test_ds:
        # Convert to float32 for CPU inference compatibility
        preds = model.predict(images.numpy().astype('float32'), verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 4. PRINT CLASSIFICATION REPORT
    print("\n" + "="*60)
    print("📊 FINAL CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(report)

    # 5. GENERATE CONFUSION MATRIX
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix: Diabetic Retinopathy Detection')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # Save the figure
    log_dir = os.path.join(cfg.BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, "confusion_matrix.png")
    
    plt.savefig(save_path)
    print(f"\n✅ Confusion Matrix image saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_evaluation()