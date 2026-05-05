from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import os
import sys
import uuid
import shutil

# Fix pathing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
from test_inference import DRInference

app = FastAPI()

# Load model (Using the 85% version as primary)
MODEL_PATH = os.path.join(BASE_DIR, "models/dr_model_final_85.keras")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "models/dr_model_85_percent.keras")

engine = DRInference(model_path=MODEL_PATH)

@app.post("/predict")
async def predict_eye(file: UploadFile = File(...)):
    # Create temp directory
    os.makedirs("temp", exist_ok=True)
    temp_path = f"temp/{uuid.uuid4()}_{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Run inference using the fixed logic engine
        result = engine.predict(temp_path, use_tta=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)