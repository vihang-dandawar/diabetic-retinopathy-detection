import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_BASE_DIR = "data"            # Your original data folder
OUTPUT_BASE_DIR = "data_processed"  # Where the clean images will go
IMG_SIZE = 224                     # Optimized size for MobileNetV2

def process_fundus_image(img_path, output_path, size=224):
    """Applies Smart Crop and CLAHE to a single image."""
    img = cv2.imread(img_path)
    if img is None: 
        return False
    
    # 1. Smart Crop: Remove black borders
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (the eye)
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        img = img[y:y+h, x:x+w]

    # 2. Resize to 224x224
    img = cv2.resize(img, (size, size))
    
    # 3. Apply CLAHE (Contrast Enhancement)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
     
    # 4. Save processed image
    cv2.imwrite(output_path, img)
    return True

def start_preprocessing():
    """Loops through train, val, and test folders."""
    for split in ['train', 'val', 'test']:
        input_split_path = os.path.join(INPUT_BASE_DIR, split)
        output_split_path = os.path.join(OUTPUT_BASE_DIR, split)
        
        if not os.path.exists(input_split_path):
            print(f"Skipping {split}, folder not found.")
            continue
            
        print(f"Processing {split} split...")
        
        # Get all subfolders (0, 1, 2, 3, 4)
        categories = os.listdir(input_split_path)
        
        for cat in categories:
            img_paths = glob(os.path.join(input_split_path, cat, "*.*"))
            out_cat_path = os.path.join(output_split_path, cat)
            os.makedirs(out_cat_path, exist_ok=True)
            
            for path in tqdm(img_paths, desc=f"Class {cat}"):
                filename = os.path.basename(path)
                out_path = os.path.join(out_cat_path, filename)
                process_fundus_image(path, out_path, size=IMG_SIZE)

if __name__ == "__main__":
    start_preprocessing()
    print("Preprocessing Complete! You can now train on 'data_processed'")