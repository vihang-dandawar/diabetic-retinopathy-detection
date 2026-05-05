import tensorflow as tf
import numpy as np
import cv2
import os
import sys

# ============================================================

try:
    import tf_keras as legacy_keras
    print("✅ Legacy Bridge (tf_keras) active.")
except ImportError:
    print("❌ Error: tf_keras not found.")
    print("Please run: pip install tf_keras --user")
    sys.exit()

# Add current directory to path so it can find config.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import config as cfg
except ImportError:
    cfg = None

# Custom loss placeholder for loading
@tf.keras.utils.register_keras_serializable()
def focal_loss(y_true, y_pred):
    return tf.reduce_sum(y_true, axis=-1) 

class DRInference:
    def __init__(self, model_path=None):
        # Determine Path
        if model_path:
            path = model_path
        elif cfg:
            path = cfg.FINAL_MODEL_PATH
        else:
            path = os.path.join("models", "dr_model_final_85.keras")

        # Clean the path string for Windows
        path = os.path.normpath(path.strip().replace('"', '').replace("'", ""))
        
        # If running from inside 'src', check one level up
        if not os.path.exists(path):
            alt_path = os.path.join("..", path)
            if os.path.exists(alt_path):
                path = alt_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")

        print(f">>> Loading Keras 2 Model via Bridge: {path}")
        
        # USE THE LEGACY BRIDGE TO LOAD THE MODEL
        # This fixes the 'quantization_mode' and 'zip file' errors
        self.model = legacy_keras.models.load_model(
            path, 
            custom_objects={'focal_loss': focal_loss}, 
            compile=False
        )
        print("✅ Model loaded successfully on Laptop CPU!")

        self.class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    def preprocess_image(self, image_path):
        """Replicates the 1.15L image training preprocessing (Smart Crop + CLAHE)."""
        image_path = os.path.normpath(image_path.strip().replace('"', '').replace("'", ""))
        img = cv2.imread(image_path)
        if img is None: raise ValueError(f"Could not open image: {image_path}")
        
        # 1. Smart Crop (Remove black borders)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            img = img[y:y+h, x:x+w]
            
        # 2. Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # 3. CLAHE (Contrast Enhancement)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        img = cv2.merge((l,a,b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        
        # 4. Save Debug Image (To verify what the AI sees)
        cv2.imwrite("debug_laptop_input.jpg", img)
        
        # 5. Convert to RGB for the AI
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # IMPORTANT: Return raw 0-255 pixels as float32.
        # DO NOT call preprocess_input here because the model handles it internally.
        img_tensor = tf.keras.preprocessing.image.img_to_array(img_rgb)
        return np.expand_dims(img_tensor, axis=0)

    def predict(self, image_path, use_tta=True):
        img_batch = self.preprocess_image(image_path)
        
        if use_tta:
            # 5-way Test Time Augmentation (TTA)
            versions = [
                img_batch[0],
                np.fliplr(img_batch[0]),
                np.flipud(img_batch[0]),
                np.rot90(img_batch[0], 1),
                np.rot90(img_batch[0], 2)
            ]
            preds = self.model.predict(np.array(versions).astype('float32'), verbose=0)
            final_probs = np.mean(preds, axis=0)
        else:
            final_probs = self.model.predict(img_batch.astype('float32'), verbose=0)[0]
        
        # Determine winning class
        class_idx = np.argmax(final_probs)
        
        # CLINICAL BIAS CORRECTION
        # If the model is at least 30% sure it's 'No DR', prioritize that.
        # This prevents the 'Mild' obsession on healthy eyes.
        if final_probs[0] > 0.30:
            class_idx = 0
        else:
            class_idx = np.argmax(final_probs)

        return {
            "label": self.class_names[class_idx],
            "confidence": f"{final_probs[class_idx]*100:.2f}%",
            "scores": {self.class_names[i]: float(final_probs[i]) for i in range(5)}
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", help="Path to model file", default=None)
    args = parser.parse_args()
    
    try:
        engine = DRInference(model_path=args.model)
        res = engine.predict(args.image)
        print(f"\n✅ FINAL DIAGNOSIS: {res['label']} ({res['confidence']})")
        print("\nDetailed Probabilities:")
        for k, v in res['scores'].items():
            print(f" - {k}: {v:.4f}")
    except Exception as e:
        print(f"❌ Error during inference: {e}")