import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import config as cfg

# 1. GPU Memory Setup (Strict 1800MB limit for 2GB Card)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_logical_device_configuration(
            gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=1800)])
    except: pass

# 2. Custom loss needed to load model
def categorical_focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss(y_true, y_pred):
        return tf.reduce_sum(y_true, axis=-1)
    return focal_loss

def run_evaluation():
    # Define Class Names directly to avoid Config error
    CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    
    # Use the winner model path
    model_path = os.path.join(cfg.BASE_DIR, "models", "dr_model_final_85.keras")
    # Point to the processed test data
    test_dir = os.path.join(cfg.BASE_DIR, "data_processed", "test")
    
    print(f"--- Starting Final Evaluation ---")
    print(f"Loading Model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return

    # Load Model with custom objects
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'focal_loss': categorical_focal_loss()},
        compile=False
    )

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

    print("Extracting predictions (this may take 2-3 minutes)...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 3. PRINT CLASSIFICATION REPORT
    print("\n" + "="*60)
    print("📊 FINAL CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(report)

    # 4. GENERATE CONFUSION MATRIX
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix: Diabetic Retinopathy Detection')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # Create logs folder if it doesn't exist
    os.makedirs(os.path.join(cfg.BASE_DIR, "logs"), exist_ok=True)
    save_path = os.path.join(cfg.BASE_DIR, "logs", "confusion_matrix.png")
    
    plt.savefig(save_path)
    print(f"\n✅ Confusion Matrix image saved to: {save_path}")
    print("Open this image to use in your presentation/report!")
    plt.show()

if __name__ == "__main__":
    run_evaluation()