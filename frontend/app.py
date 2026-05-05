import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="AI Retina Diagnostic", layout="wide")

st.title("👁️ Diabetic Retinopathy Diagnostic System")
st.markdown("---")

# Sidebar Information
st.sidebar.header("About the System")
st.sidebar.info("""
- **Model:** Fine-tuned MobileNetV2
- **Pre-processing:** OpenCV CLAHE + Smart Crop
- **Ensemble:** 5-Way Test Time Augmentation (TTA)
- **Deployment:** FastAPI + Streamlit
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Section")
    input_mode = st.radio("Choose Input Method:", ("Upload Image", "Capture via Camera"))
    
    input_file = None
    if input_mode == "Upload Image":
        input_file = st.file_uploader("Upload Retinal Fundus Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    else:
        input_file = st.camera_input("Take a photo of the retina")

with col2:
    st.subheader("Diagnostic Results")
    if input_file is not None:
        # Display the image
        img = Image.open(input_file)
        st.image(img, caption="Target Image for Analysis", width=400)
        
        if st.button("Start AI Analysis"):
            with st.spinner("Applying CLAHE Preprocessing & TTA Analysis..."):
                try:
                    # Prepare file for API
                    files = {"file": input_file.getvalue()}
                    response = requests.post("http://localhost:8000/predict", files=files)
                    res = response.json()
                    
                    # 1. Result Alert
                    label = res['label']
                    conf = res['confidence']
                    
                    if label == "No DR":
                        st.success(f"Diagnosis: **{label}** ({conf})")
                    elif label in ["Mild", "Moderate"]:
                        st.warning(f"Diagnosis: **{label}** ({conf})")
                    else:
                        st.error(f"Diagnosis: **{label}** ({conf}) - URGENT REFERRAL")

                    # 2. Score Breakdown
                    st.write("### Severity Probability Distribution")
                    for cls, score in res['scores'].items():
                        st.write(f"**{cls}**")
                        st.progress(score)
                        
                except Exception as e:
                    st.error(f"Connection Error: Is the Backend running? {e}")
    else:
        st.info("Please upload or capture an image to begin diagnosis.")