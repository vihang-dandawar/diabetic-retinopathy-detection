import streamlit as st
import requests
import pandas as pd
import os
import sys

# --- ULTIMATE PATH FIX ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import clinical utils using the corrected path
from src.clinical_utils import generate_pdf_report

st.set_page_config(page_title="RetinaAI Portal", layout="wide")

# 1. SESSION INITIALIZATION
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# 2. LOGIN SCREEN
if not st.session_state['logged_in']:
    st.title("🔒 Doctor Login - RetinaAI")
    st.write("Authorized Personnel Only")
    
    with st.form("login_form"):
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            try:
                # Attempt login request
                res = requests.post("http://127.0.0.1:8000/login", data={"username": user, "password": pw})
                
                if res.status_code == 200:
                    st.session_state['logged_in'] = True
                    st.success("Login Successful! Redirecting...")
                    st.rerun() # Updated for Python 3.13 compatibility
                else:
                    st.error("Invalid Username or Password")
            except requests.exceptions.ConnectionError:
                st.error("Backend Server is not responding. Ensure api/main.py is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                
    st.info("Default Credentials: admin / admin***")

# 3. MAIN APPLICATION (Visible only if logged in)
else:
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2865/2865913.png", width=80)
    st.sidebar.write(f"👤 **Logged in as:** Administrator")
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    st.sidebar.markdown("---")
    menu = st.sidebar.radio("Navigation", ["New Patient Scan", "Patient Records Archive"])

    if menu == "New Patient Scan":
        st.title("👁️ Retinopathy AI Screening")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Patient Intake")
            p_name = st.text_input("Patient Full Name")
            p_age = st.number_input("Age", 1, 120, 30)
            uploaded_file = st.file_uploader("Upload Fundus Photo", type=["jpg", "jpeg", "png"])
            if uploaded_file: 
                st.image(uploaded_file, caption="Input Preview", width=350)

        with col2:
            st.subheader("2. AI Diagnostic Results")
            if uploaded_file and st.button("🔍 Run Deep Analysis"):
                if not p_name:
                    st.warning("Please enter patient name before analysis.")
                else:
                    with st.spinner("Executing Legacy Bridge & TTA Ensemble..."):
                        try:
                            files = {"file": uploaded_file.getvalue()}
                            payload = {"patient_name": p_name, "patient_age": str(p_age)}
                            response = requests.post("http://127.0.0.1:8000/predict", data=payload, files=files)
                            
                            if response.status_code == 200:
                                res = response.json()
                                st.success(f"Primary Diagnosis: {res['label']}")
                                st.info(f"Analysis Confidence: {res['confidence']}")
                                
                                # Show probabilities
                                with st.expander("View Probability Distribution"):
                                    for cls, score in res['scores'].items():
                                        st.write(f"{cls}")
                                        st.progress(score)
                                
                                # Generate PDF
                                report_data = {
                                    "name": p_name, 
                                    "age": p_age, 
                                    "diagnosis": res['label'], 
                                    "confidence": res['confidence']
                                }
                                pdf_bytes = generate_pdf_report(report_data)
                                st.download_button(
                                    label="📩 Download Clinical PDF Report",
                                    data=pdf_bytes,
                                    file_name=f"DR_Report_{p_name}.pdf",
                                    mime="application/pdf"
                                )
                            else:
                                st.error("Inference Error. Check backend logs.")
                        except Exception as e:
                            st.error(f"Connection Error: {e}")

    else:
        st.title("📜 Clinical Records Archive")
        try:
            history_res = requests.get("http://127.0.0.1:8000/history")
            if history_res.status_code == 200:
                history = history_res.json()
                if history:
                    df = pd.DataFrame(history, columns=["Name", "Age", "Diagnosis", "Confidence", "Timestamp"])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No records found in database.")
            else:
                st.error("Failed to fetch history.")
        except:
            st.error("Database Connection Failed.")