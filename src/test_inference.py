import tensorflow as tf
import numpy as np
from PIL import Image
import config as cfg

class DRInference:
    def __init__(self, model_path=None):
        path = model_path or cfg.FINAL_MODEL_PATH
        print(f"Loading model from {path}...")
        self.model = tf.keras.models.load_model(path)
        self.class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    def preprocess(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(cfg.IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # Apply MobileNetV2 scaling
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_path):
        processed_img = self.preprocess(image_path)
        predictions = self.model.predict(processed_img, verbose=0)
        
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return {
            "class_id": int(class_idx),
            "label": self.class_names[class_idx],
            "confidence": round(float(confidence), 4),
            "raw_scores": {self.class_names[i]: float(predictions[0][i]) for i in range(5)}
        }

# For Testing standalone
if __name__ == "__main__":
    import sys
    engine = DRInference()
    if len(sys.argv) > 1:
        result = engine.predict(sys.argv[1])
        print(result)
    else:
        print("Please provide an image path: python inference.py test.jpg")