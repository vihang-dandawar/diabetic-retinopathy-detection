import sqlite3
import hashlib
from fpdf import FPDF
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clinical_data.db")

# 1. DATABASE LOGIC
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, age INTEGER, diagnosis TEXT, confidence TEXT, timestamp DATETIME
        )''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, password TEXT NOT NULL
        )''')
    cursor.execute("SELECT * FROM users WHERE username='admin'")
    if not cursor.fetchone():
        hashed_pw = hashlib.sha256("admin123".encode()).hexdigest()
        cursor.execute("INSERT INTO users VALUES (?, ?)", ("admin", hashed_pw))
    conn.commit()
    conn.close()

def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_pw))
    user = cursor.fetchone()
    conn.close()
    return user is not None

def save_patient_record(name, age, diagnosis, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO patients (name, age, diagnosis, confidence, timestamp) VALUES (?, ?, ?, ?, ?)', 
                   (name, age, diagnosis, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# 2. PROFESSIONAL PDF REPORT LOGIC
class DRReport(FPDF):
    def header(self):
        # Hospital Logo placeholder (Blue Bar)
        self.set_fill_color(31, 73, 125)
        self.rect(0, 0, 210, 35, 'F')
        
        self.set_font('helvetica', 'B', 22)
        self.set_text_color(255, 255, 255)
        self.cell(0, 15, 'RETINA AI DIAGNOSTICS', 0, 1, 'C')
        
        self.set_font('helvetica', 'I', 10)
        self.cell(0, -2, 'Advanced Automated Ocular Screening Report', 0, 1, 'C')
        self.ln(25)

    def footer(self):
        self.set_y(-30)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, 'This is a computer-generated report based on Deep Learning analysis (MobileNetV2).', 0, 1, 'C')
        self.cell(0, 5, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(patient_data):
    # Clinical Definitions for each stage
    clinical_notes = {
        "No DR": "No clinical signs of retinopathy detected. The retinal structures appear within normal physiological limits.",
        "Mild": "Tiny bulges (microaneurysms) in the retinal blood vessels are detected. Annual monitoring advised.",
        "Moderate": "Hemorrhages or lipid leaks (exudates) detected. Suggest consultation with a retina specialist.",
        "Severe": "Significant vessel blockage found. High risk of vision loss. Urgent clinical intervention required.",
        "Proliferative": "Neovascularization (new fragile vessel growth) detected. Immediate specialist referral and treatment needed."
    }

    pdf = DRReport()
    pdf.add_page()
    
    # --- Patient Information Section ---
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("helvetica", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "  PATIENT INFORMATION", 0, 1, 'L', True)
    pdf.ln(2)
    
    pdf.set_font("helvetica", '', 11)
    pdf.cell(100, 8, f"Name: {patient_data['name']}", 0, 0)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%d %b %Y, %H:%M')}", 0, 1)
    pdf.cell(100, 8, f"Age: {patient_data['age']}", 0, 0)
    pdf.cell(0, 8, f"Report ID: RAD-{hashlib.md5(patient_data['name'].encode()).hexdigest()[:8].upper()}", 0, 1)
    pdf.ln(10)

    # --- AI Diagnostic Section ---
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("helvetica", 'B', 12)
    pdf.cell(0, 10, "  DIAGNOSTIC SUMMARY", 0, 1, 'L', True)
    pdf.ln(5)

    # Styling for the Result Box
    diag = patient_data['diagnosis']
    if diag == "No DR":
        box_color = (0, 128, 0) # Green
    elif diag in ["Mild", "Moderate"]:
        box_color = (255, 140, 0) # Orange
    else:
        box_color = (200, 0, 0) # Red

    pdf.set_draw_color(*box_color)
    pdf.set_line_width(1)
    pdf.set_font("helvetica", 'B', 20)
    pdf.set_text_color(*box_color)
    pdf.cell(0, 25, f"STATUS: {diag}", 1, 1, 'C')
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", 'B', 11)
    pdf.ln(5)
    pdf.cell(0, 8, f"Analysis Confidence: {patient_data['confidence']}", 0, 1, 'C')
    pdf.ln(5)

    # --- Clinical Findings Section ---
    pdf.set_font("helvetica", 'B', 12)
    pdf.cell(0, 10, "  CLINICAL INTERPRETATION", 0, 1, 'L', True)
    pdf.ln(2)
    pdf.set_font("helvetica", '', 11)
    # Get clinical note based on diagnosis name
    note = next((v for k, v in clinical_notes.items() if k in diag), "Clinical observation required.")
    pdf.multi_cell(0, 8, note, 0, 'L')
    pdf.ln(10)

    # --- Technical Methodology Section ---
    pdf.set_font("helvetica", 'B', 10)
    pdf.cell(0, 8, "TECHNICAL METHODOLOGY", 0, 1)
    pdf.set_font("helvetica", '', 9)
    pdf.set_text_color(100, 100, 100)
    method_text = (
        "This diagnosis was performed using a MobileNetV2 Neural Network trained on the EyePacs and APTOS datasets. "
        "The image underwent Contrast Limited Adaptive Histogram Equalization (CLAHE) and 5-way Test Time Augmentation "
        "to reach a stabilized consensus."
    )
    pdf.multi_cell(0, 5, method_text)

    # Signature Area
    pdf.ln(15)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(140, pdf.get_y(), 190, pdf.get_y())
    pdf.set_y(pdf.get_y() + 2)
    pdf.set_x(140)
    pdf.set_font("helvetica", 'B', 10)
    pdf.cell(50, 5, "Authorized AI Auditor", 0, 1, 'C')

    return bytes(pdf.output())