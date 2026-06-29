# 👁️ Diabetic Retinopathy Diagnostic System

A production-grade, deep learning pipeline designed to detect and grade the severity of Diabetic Retinopathy (DR) from retinal fundus images.

## 🚀 Project Overview
This project implements an end-to-end medical AI solution, moving from raw data engineering to a full-stack deployment. It is optimized to run on constrained hardware (2GB NVIDIA T1000) while maintaining clinical-grade diagnostic stability.

### Key Metrics (Current Baseline)
- **Training Accuracy:** ~87.9%
- **Validation Accuracy:** ~81.8%
- **Healthy Retina Recall:** 94% (High Specificity)
- **Architecture:** MobileNetV2 (Alpha 1.3 optimized)

---

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow 2.10, Keras, PyTorch (Experimental)
- **Data Engineering:** OpenCV (CLAHE, Smart Contours)
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Optimization:** Categorical Focal Loss, Mixed Precision, Test Time Augmentation (TTA)

---

## 📂 Project Structure
```text
BEProject/
├── api/              # FastAPI Backend Server
├── frontend/         # Streamlit Web Dashboard
├── models/           # Pre-trained .keras / .h5 models
├── src/              # Core Logic (Preprocessing, Factory, Training)
├── data/             # Raw Images (Ignored by Git)
├── data_processed/   # CLAHE Enhanced Images (Ignored by Git)
└── logs/             # Confusion Matrix and Training Logs
🧬 Preprocessing Pipeline
To ensure the AI can detect microscopic lesions, every image passes through a 3-stage OpenCV pipeline:
Smart Crop: Uses contour detection to remove black borders.
CLAHE: Contrast Limited Adaptive Histogram Equalization to highlight vessels and hemorrhages.
Spatial Normalization: Standardizing inputs to 224x224 pixels.

🚦 How to Run:
1. Data Preparation
python src/preprocess_dataset.py

2. Launch Backend (API)
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
python api/main.py

3. Launch Frontend (UI)
streamlit run frontend/app.py

📈 Future Scope
Explainable AI: Implementation of Grad-CAM heatmaps for clinical transparency.
Edge Deployment: Conversion to TFLite for mobile screening.
