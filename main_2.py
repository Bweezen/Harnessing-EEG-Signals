import streamlit as st
import pandas as pd
from joblib import load
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import datetime

# Load pre-trained model
model = load("Random_Forest.pkl")

# Define default thresholds (can be adjusted based on data)
DEFAULT_THRESHOLDS = {
    "valence_low": 3,
    "arousal_high": 4,
}

# Streamlit app configuration
st.set_page_config(page_title="EEG Analysis", layout="wide")

# Helper Functions
def normalize_data(df):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def determine_combination(valence, arousal, thresholds):
    valence_status = "low" if valence <= thresholds["valence_low"] else "high"
    arousal_status = "high" if arousal >= thresholds["arousal_high"] else "low"
    return valence_status, arousal_status

def classify_condition(valence, arousal, thresholds):
    valence_status, arousal_status = determine_combination(valence, arousal, thresholds)
    if valence_status == "low" and arousal_status == "high":
        return "SEVERE", "red"
    elif valence_status == "low" and arousal_status == "low":
        return "MODERATE", "orange"
    elif valence_status == "high" and arousal_status == "low":
        return "MILD", "yellow"
    return "GOOD", "green"

def generate_report_data(predictions, thresholds):
    report_data = []
    for valence, arousal in predictions:
        severity, color = classify_condition(valence, arousal, thresholds)
        report_data.append({
            "Valence": valence,
            "Arousal": arousal,
            "Severity": severity,
            "Color": color
        })
    return report_data

def create_line_chart(report_data):
    symptoms = ["Stress", "Anxiety", "Insomnia", "Alzheimers"]  # Matches dictionary keys
    severity_mapping = {"GOOD": 10, "MILD": 6, "MODERATE": 4, "SEVERE": 2, "NORMAL": 8}

    # Initialize averages dictionary
    averages = {symptom: 0 for symptom in symptoms}

    # Iterate through symptoms to calculate severity
    for condition in symptoms:
        condition_data = report_data.get(condition.lower(), [])
        if condition_data:
            # Get the severity of the last entry
            last_entry = condition_data[-1]  # Ensure this is a valid dict
            severity_label = last_entry.get(condition.capitalize(), "GOOD")  # Default to "GOOD"
            averages[condition] = severity_mapping.get(severity_label, 10)  # Map to severity score

    # Ensure valid numeric values for plotting
    x_values = list(averages.keys())  # Symptom names
    y_values = list(averages.values())  # Severity scores (numeric)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.title("Symptom Severity Levels")
    plt.ylabel("Severity (Higher Is Better)")
    plt.xlabel("Symptoms")
    plt.ylim(0, 12)
    plt.grid(True, alpha=0.6)
    plt.tight_layout()

    return plt

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
    patient_info = f"""Name: {user_info.get('Name', 'N/A')}<br/>
                   Age: {user_info.get('Age', 'N/A')}<br/>
                   Gender: {user_info.get('Gender', 'N/A')}<br/>
                   Medical History: {user_info.get('Medical History', 'N/A')}"""
    elements.append(Paragraph("Patient Information:", styles['Heading2']))
    elements.append(Paragraph(patient_info, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Metrics Table
    elements.append(Paragraph("Condition Metrics:", styles['Heading2']))
    table_data = [["Valence", "Arousal", "Severity", "Color"]]
    for item in report_data:
        table_data.append([item['Valence'], item['Arousal'], item['Severity'], item['Color']])
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Disclaimer
    elements.append(Paragraph("Disclaimer:", styles['Heading2']))
    elements.append(Paragraph("This report is for informational purposes only and should not be considered a substitute for professional medical advice.", styles['Normal']))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

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

                    # Preprocess data
                    df = normalize_data(df)
                    
                    # Predictions
                    predictions = model.predict(df)
                    report_data = generate_report_data(predictions, DEFAULT_THRESHOLDS)

                    # Create and display line chart in Streamlit
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
