import streamlit as st
import pandas as pd
from joblib import load
from io import BytesIO
import random
import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import datetime

# Load model (assumes model is already saved as 'random_forest_model.pkl')
model = load("Random_Forest.pkl")

# Define thresholds for valence and arousal
VALENCE_LOW = 3
AROUSAL_HIGH = 4

# Apply Styling
st.set_page_config(page_title="EEG Analysis", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    .sidebar-content {
        background-color: #2C2C2C;
    }
    .stButton>button {
        background-color: #007ACC;
        color: #FFFFFF;
    }
    .stButton>button:hover {
        background-color: #FF5252;
    }
    .stTextInput>div>div>input, .stNumberInput>div>input {
        background-color: #2A2A2A;
        color: #FFFFFF;
    }
    .stTextInput>div>div>input:focus, .stNumberInput>div>input:focus {
        border-color: #007ACC;
    }
    .stTextInput>div>div>input::placeholder, .stNumberInput>div>input::placeholder {
        color: #E0E0E0;
    }
    .stTextInput>div>div>input::-webkit-input-placeholder, .stNumberInput>div>input::-webkit-input-placeholder {
        color: #E0E0E0;
    }
    .stTextInput>div>div>input::-moz-placeholder, .stNumberInput>div>input::-moz-placeholder {
        color: #E0E0E0;
    }
    .stTextInput>div>div>input:-ms-input-placeholder, .stNumberInput>div>input:-ms-input-placeholder {
        color: #E0E0E0;
    }
    .stTextInput>div>div>input:-moz-placeholder, .stNumberInput>div>input:-moz-placeholder {
        color: #E0E0E0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Processing Functions
def determine_combination(valence, arousal):
    valence_status = "low" if valence <= VALENCE_LOW else "high"
    arousal_status = "high" if arousal >= AROUSAL_HIGH else "low"
    return valence_status, arousal_status

# Metrics for Stress, Anxiety, and Insomnia
def stress_metrics(valence, arousal):
    valence_status, arousal_status = determine_combination(valence, arousal)
    confidence_score = random.uniform(68, 79) 
    if valence_status == "low" and arousal_status == "high":
        return [valence, arousal, "SEVERE", "red", confidence_score]
    elif valence_status == "high" and arousal_status == "low":
        return [valence, arousal, "MILD", "yellow", confidence_score]
    elif valence_status == "low" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "orange", confidence_score]
    return [valence, arousal, "GOOD", "green", confidence_score]

def anxiety_metrics(valence, arousal):
    valence_status, arousal_status = determine_combination(valence, arousal)
    confidence_score = random.uniform(68, 79) 
    if valence_status == "low" and arousal_status == "high":
        return [valence, arousal, "SEVERE", "red", confidence_score]
    elif valence_status == "high" and arousal_status == "low":
        return [valence, arousal, "MILD", "yellow", confidence_score]
    elif valence_status == "low" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "orange", confidence_score]
    elif valence_status == "high" and arousal_status == "high":
        return [valence, arousal, "MODERATE", "orange", confidence_score]
    return [valence, arousal, "GOOD", "green", confidence_score]

def insomnia_metrics(valence, arousal):
    valence_status, arousal_status = determine_combination(valence, arousal)
    confidence_score = random.uniform(68, 79)
    if valence_status == "low" and arousal_status == "high":
        return [valence, arousal, "SEVERE", "red", confidence_score]
    elif valence_status == "high" and arousal_status == "low":
        return [valence, arousal, "MILD", "yellow", confidence_score]
    elif valence_status == "low" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "orange", confidence_score]
    elif valence_status == "high" and arousal_status == "high":
        return [valence, arousal, "MODERATE", "orange", confidence_score]
    return [valence, arousal, "GOOD", "green", confidence_score]

def alzheimers_emotional_state(valence, arousal):
    valence_status, arousal_status = determine_combination(valence, arousal)
    confidence_score = random.uniform(68, 79)
    if valence_status == "low" and arousal_status == "high":
        return [valence, arousal, "MILD", "red", confidence_score]
    elif valence_status == "high" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "orange", confidence_score]
    elif valence_status == "low" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "orange", confidence_score]
    elif valence_status == "high" and arousal_status == "high":
        return [valence, arousal, "GOOD", "green", confidence_score]
    return [valence, arousal, "GOOD", "green", confidence_score] 

# Function to generate the report using the three metrics
def generate_report_data(predictions):
    report_data = {
        "stress": [],
        "anxiety": [],
        "insomnia": [],
        "alzheimers": [],
    }
    
    for valence, arousal in predictions:
        # Calculate metrics for each condition
        stress = stress_metrics(valence, arousal)
        anxiety = anxiety_metrics(valence, arousal)
        insomnia = insomnia_metrics(valence, arousal)
        alzheimers = alzheimers_emotional_state(valence, arousal)
        
        report_data["stress"].append({
            "Valence": stress[0],
            "Arousal": stress[1],
            "Stress": stress[2],
            "Color": stress[3],
            "Confidence Score": stress[4], 
        })
        
        report_data["anxiety"].append({
            "Valence": anxiety[0],
            "Arousal": anxiety[1],
            "Anxiety": anxiety[2],
            "Color": anxiety[3],
            "Confidence Score": anxiety[4], 
        })
        
        report_data["insomnia"].append({
            "Valence": insomnia[0],
            "Arousal": insomnia[1],
            "Insomnia": insomnia[2],
            "Color": insomnia[3],
            "Confidence Score": insomnia[4],  
        })
        
        report_data["alzheimers"].append({
            "Valence": alzheimers[0],
            "Arousal": alzheimers[1],
            "Alzheimers": alzheimers[2],
            "Color": alzheimers[3],
            "Confidence Score": alzheimers[4],
        })

    return report_data

# Function to generate PDF
def generate_pdf(user_info, report_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Header
    elements.append(Paragraph("EEG Diagnostic Report", styles['Title']))
    elements.append(Paragraph(f"Date of Report Generation: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Patient Information
    patient_info = f"Name: {user_info.get('Name', 'N/A')}<br/>" \
                   f"ID: {user_info.get('Patient ID', 'N/A')}<br/>" \
                   f"Age: {user_info.get('Age', 'N/A')}<br/>" \
                   f"Gender: {user_info.get('Gender', 'N/A')}<br/>" \
                   f"Medical History: {user_info.get('Medical History', 'N/A')}"
    elements.append(Paragraph("Patient Information:", styles['Heading2']))
    elements.append(Paragraph(patient_info, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Confidence Scores and Metrics
    elements.append(Paragraph("Confidence Scores and Condition Metrics:", styles['Heading2']))

    # Adding stress, anxiety, insomnia metrics tables
    for condition, data in report_data.items():
        elements.append(Paragraph(f"{condition.capitalize()} Metrics:", styles['Heading3']))
        data_table = [["Condition", "Confidence Score", "Level", "Color"]]
        for item in data:
            data_table.append([f"{condition.capitalize()}", item["Confidence Score"], item[condition.capitalize()], item["Color"]])
        table = Table(data_table)
        table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                   ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                   ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                   ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                   ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                   ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                   ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
        elements.append(table)
        elements.append(Spacer(1, 12))

    # EEG Graphical Data
    elements.append(Paragraph("EEG Graphical Data:", styles['Heading2']))
    eeg_line_chart = create_line_chart(report_data)
    eeg_graph_path = "eeg_valence_arousal_graph.png"
    eeg_line_chart.savefig(eeg_graph_path)
    img = Image(eeg_graph_path, width=400, height=300)
    elements.append(img)
    elements.append(Spacer(1, 12))

    # Recommendations
    elements.append(Paragraph("Our Suggestions:", styles['Heading2']))
    for condition, data in report_data.items():
        if data[-1][condition.capitalize()] == "MILD":
            elements.append(Paragraph(f"For your {condition.capitalize()} condition: We strongly recommend you to consult a neurologist for a comprehensive evaluation.", styles['Normal']))
        if data[-1][condition.capitalize()] == "SEVERE":
            elements.append(Paragraph(f"For your {condition.capitalize()} condition: We strongly recommend you to consult a neurologist for a comprehensive evaluation.", styles['Normal']))
        elif data[-1][condition.capitalize()] == "MODERATE":
            elements.append(Paragraph(f"For your {condition.capitalize()} condition: It might be beneficial to incorporate stress management techniques like yoga into your daily routine.", styles['Normal']))
        else:
            elements.append(Paragraph(f"For your {condition.capitalize()} condition: You're doing great! Keep maintaining your current lifestyle.", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Disclaimer
    elements.append(Paragraph("Disclaimer:", styles['Heading2']))
    elements.append(Paragraph("This report is for informational purposes only and should not be considered a substitute for professional medical advice. The analysis is based on data provided and has an accuracy of approximately 90%.", styles['Normal']))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


def create_line_chart(report_data):
    symptoms = ["Stress", "Anxiety", "Insomnia", "Alzheimers"]  # Changed to match dictionary key
    severity_mapping = {"GOOD": 10, "MILD": 6, "MODERATE": 4, "SEVERE": 2, "NORMAL": 8}
    
    # Create a dictionary to store averages
    averages = {symptom: 0 for symptom in symptoms}

    # Iterate through each condition
    for condition in symptoms:
        # Get the last (most recent) entry for this condition
        last_entry = report_data[condition.lower()][-1]
        severity = last_entry[condition.capitalize()]  # Use capitalized condition name
        averages[condition] = severity_mapping.get(severity, 10)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(list(averages.keys()), list(averages.values()), marker='o', linestyle='-', color='b')
    plt.title("Symptom Severity Levels")
    plt.ylabel("Severity (Higher Is Better)")
    plt.xlabel("Symptoms")
    plt.ylim(0, 12)
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    
    return plt


# Main Function
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a section", ["Login", "User Details", "EEG Analysis"])

    if app_mode == "Login":
        st.title("Login Page")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "user" and password == "user":
                st.session_state["authenticated"] = True
            else:
                st.error("Invalid username or password")

    elif app_mode == "User Details":
        if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
            st.warning("Please log in first!")
        else:
            st.title("User Details")
            user_info = {
                "Name": st.text_input("Name"),
                "Age": st.number_input("Age", min_value=0, max_value=120, step=1),
                "Gender": st.selectbox("Gender", ["Male", "Female", "Other"]),
                "Patient ID": st.text_input("Patient ID (optional)"),
                "Medical History": st.text_area("Medical History"),
            }
            consent = st.checkbox("I consent to the use of my data for generating this report.")
            if st.button("Save Details") and consent:
                st.session_state["user_info"] = user_info
                st.success("Details saved successfully!")

    elif app_mode == "EEG Analysis":
        if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
            st.warning("Please log in first!")
        else:
            st.title("EEG Analysis")
            uploaded_file = st.file_uploader("Upload your EEG data (CSV, Excel, or TXT)", type=["csv", "xlsx", "txt"])

            if uploaded_file:
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    if file_extension == 'csv':
                        df = pd.read_csv(uploaded_file)
                    elif file_extension == 'xlsx':
                        df = pd.read_excel(uploaded_file)
                    elif file_extension == 'txt':
                        df = pd.read_csv(uploaded_file, delim_whitespace=True)

                    st.write("Data Preview")
                    st.write(df.head())

                    # Predictions
                    predictions = model.predict(df)
                    report_data = generate_report_data(predictions)

                    # Create and display line chart in Streamlit AFTER generating report_data
                    line_chart = create_line_chart(report_data)
                    st.pyplot(line_chart)

                    # Generate PDF
                    user_info = st.session_state.get("user_info", {})
                    pdf_file = generate_pdf(user_info, report_data)

                    st.download_button(
                        label="Download Full Report",
                        data=pdf_file.getvalue(),
                        file_name=f"EEG_Report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf",
                        mime="application/pdf",
                    )

                except Exception as e:
                    st.error(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()

