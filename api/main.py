from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn
import os
import sys
import uuid
import shutil

# --- CRITICAL PATH FIX ---
# This ensures Python can see the 'src' folder from the 'api' folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.test_inference import DRInference
from src.clinical_utils import init_db, save_patient_record

app = FastAPI(title="Diabetic Retinopathy Diagnostic API")

# Initialize Database and Model Engine
init_db()
MODEL_PATH = os.path.join(BASE_DIR, "models", "dr_model_final_85.keras")
engine = DRInference(model_path=MODEL_PATH)

@app.post("/predict")
async def predict_eye(
    file: UploadFile = File(...), 
    patient_name: str = Form("Unknown"), 
    patient_age: int = Form(0)
):
    # Create temp directory
    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    file_ext = file.filename.split(".")[-1]
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.{file_ext}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. Run AI Diagnosis
        result = engine.predict(temp_path)
        
        # 2. Save result to SQLite Database
        save_patient_record(
            name=patient_name, 
            age=patient_age, 
            diagnosis=result['label'], 
            confidence=result['confidence']
        )
        return result
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/history")
def get_history():
    import sqlite3
    db_path = os.path.join(BASE_DIR, "clinical_data.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Get last 20 patients
    cursor.execute("SELECT name, age, diagnosis, confidence, timestamp FROM patients ORDER BY timestamp DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()
    return rows



@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    from src.clinical_utils import verify_user
    if verify_user(username, password):
        return {"status": "success", "message": "Logged in"}
    else:
        raise HTTPException(status_code=401, detail="Invalid Credentials")



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)