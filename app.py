# # import streamlit as st
# # import pandas as pd
# # from joblib import load
# # from io import BytesIO

# # # Load model (assumes model is already saved as 'random_forest_model.pkl')
# # model = load("random_forest_model.pkl")

# # def process_eeg_data(df):
# #     # Get predictions from the model
# #     predictions = model.predict(df)
    

# #     # Print out the shape and content of predictions for debugging
# #     print(f"Predictions shape: {predictions.shape}")
# #     print(f"Predictions: {predictions}")

# #     # Loop through the predictions and handle multiple values if necessary
# #     report = []
# #     for index, prediction in enumerate(predictions):
# #         # print(f"Prediction at index {index}: {prediction[0]}, {prediction[1]}")  # Explicitly print with comma

# #         # Unpack prediction correctly: prediction is a 1D array of length 2 (valence, arousal)
# #         try:
# #             valence = float(prediction[0]) 
# #             arousal = float(prediction[1]) 
# #             # print(valence)
# #             # print(arousal) # Format arousal with index
# #             report_item = assess_emotions(valence, arousal)
# #             # print(report_item)
# #             report.append(report_item)
# #             # print(report)
# #         except ValueError as e:
# #             print(f"Error unpacking prediction: {e}")
# #             continue

# #     return report

# # def assess_emotions(valence, arousal):
# #     """
# #     Assess emotional states based on valence and arousal values.
# #     """
# #     results = {
# #         "Stress": stress_metrics(valence, arousal),
# #         "Insomnia": insomnia_metrics(valence, arousal),
# #         "Anxiety": anxiety_metrics(valence, arousal),
# #     }

# #     report = ""
# #     for condition, result in results.items():
# #         report += f"{condition}: {result[2]}\n"

# #     return report

# # # Dummy functions for stress, insomnia, and anxiety metrics
# # def stress_metrics(valence, arousal):
# #     if valence < 5 and arousal > 5:
# #         if 0 <= valence <= 2 and 7 <= arousal <= 9:
# #             return [valence, arousal, "SEVERE", "red"]
# #         elif 2 < valence <= 4 and 6 <= arousal <= 7:
# #             return [valence, arousal, "MODERATE", "orange"]
# #         elif 4 < valence <= 5 and 5 <= arousal <= 6:
# #             return [valence, arousal, "LIGHT", "yellow"]
# #         else:
# #             return [valence, arousal, "NORMAL", "green"]
# #     else:
# #         return [valence, arousal, "GOOD", "green"]

# # def insomnia_metrics(valence, arousal):
# #     if valence < 5 and arousal > 6:
# #         if 0 <= valence <= 2 and 8 <= arousal <= 10:
# #             return [valence, arousal, "SEVERE", "red"]
# #         elif 2 < valence <= 4 and 7 <= arousal <= 8:
# #             return [valence, arousal, "MODERATE", "orange"]
# #         elif 4 < valence <= 5 and 6 <= arousal <= 7:
# #             return [valence, arousal, "MILD", "yellow"]
# #         else:
# #             return [valence, arousal, "NORMAL", "green"]
# #     else:
# #         return [valence, arousal, "GOOD", "green"]

# # def anxiety_metrics(valence, arousal):
# #     if valence < 4 and arousal > 6:
# #         if 0 <= valence <= 2 and 8 <= arousal <= 10:
# #             return [valence, arousal, "SEVERE", "red"]
# #         elif 2 < valence <= 3 and 7 <= arousal <= 8:
# #             return [valence, arousal, "MODERATE", "orange"]
# #         elif 3 < valence <= 4 and 6 <= arousal <= 7:
# #             return [valence, arousal, "MILD", "yellow"]
# #         else:
# #             return [valence, arousal, "NORMAL", "green"]
# #     else:
# #         return [valence, arousal, "GOOD", "green"]

# # # Streamlit UI
# # st.title('EEG Disease Detection and Emotional State Analysis')

# # st.write("""
# #     Upload your EEG data in CSV, Excel, or TXT format to determine the presence of diseases/disorders like stress, insomnia, or anxiety.
# # """)

# # # Upload file
# # uploaded_file = st.file_uploader("Choose a file (CSV, Excel, or TXT)", type=["csv", "xlsx", "txt"])

# # if uploaded_file is not None:
# #     try:
# #         # Determine the file type and read it accordingly
# #         file_extension = uploaded_file.name.split('.')[-1].lower()

# #         if file_extension == 'csv':
# #             df = pd.read_csv(uploaded_file)
# #         elif file_extension == 'xlsx':
# #             df = pd.read_excel(uploaded_file)
# #         elif file_extension == 'txt':
# #             # Read a text file assuming whitespace delimiters (can adjust if necessary)
# #             text = uploaded_file.read().decode('utf-8')
# #             df = pd.read_csv(BytesIO(text.encode()), delim_whitespace=True)

# #         # Show a preview of the data
# #         st.subheader('Preview of Uploaded Data')
# #         st.write(df.head())

# #         # Process the data and get the report
# #         report = process_eeg_data(df)

# #         st.subheader("Disease/Disorder Report")
# #         for entry in report:
# #             st.text(entry)
    
# #     except Exception as e:
# #         st.error(f"Error processing the file: {e}")




# import streamlit as st
# import pandas as pd
# from joblib import load
# from io import BytesIO

# # Load model (assumes model is already saved as 'random_forest_model.pkl')
# model = load("random_forest_model.pkl")

# def process_eeg_data(df):
#     # Get predictions from the model
#     predictions = model.predict(df)
    
#     # Loop through the predictions and handle multiple values if necessary
#     report = []
#     for index, prediction in enumerate(predictions):
#         try:
#             valence = float(prediction[0]) 
#             arousal = float(prediction[1]) 
#             report_item = assess_emotions(valence, arousal)
#             report.append(report_item)
#         except ValueError as e:
#             print(f"Error unpacking prediction: {e}")
#             continue

#     return report

# def assess_emotions(valence, arousal):
#     """
#     Assess emotional states based on valence and arousal values.
#     """
#     results = {
#         "Stress": stress_metrics(valence, arousal),
#         "Insomnia": insomnia_metrics(valence, arousal),
#         "Anxiety": anxiety_metrics(valence, arousal),
#     }

#     report = ""
#     for condition, result in results.items():
#         report += format_condition_report(condition, result)

#     return report

# def format_condition_report(condition, result):
#     """
#     Format the condition's report with colors and severity.
#     """
#     valence, arousal, severity, color = result
#     icon = get_condition_icon(severity)
#     formatted_text = f"""
#     ### {condition} {icon}
#     # - **Valence**: {valence}
#     # - **Arousal**: {arousal}
#     - **Severity**: {severity}

#     <div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white; font-weight: bold;">
#         {severity} - {condition}
#     </div>
#     """
#     return formatted_text

# def get_condition_icon(severity):
#     """
#     Get the appropriate emoji icon for the severity level.
#     """
#     if severity == "SEVERE":
#         return "🔥"
#     elif severity == "MODERATE":
#         return "⚠️"
#     elif severity == "MILD":
#         return "💛"
#     elif severity == "NORMAL":
#         return "🔄"
#     elif severity == "GOOD":
#         return "🌞"
#     else:
#         return "✅"

# # Dummy functions for stress, insomnia, and anxiety metrics
# def stress_metrics(valence, arousal):
#     if valence < 5 and arousal > 5:
#         if 0 <= valence <= 2 and 7 <= arousal <= 9:
#             return [valence, arousal, "SEVERE", "red"]
#         elif 2 < valence <= 4 and 6 <= arousal <= 7:
#             return [valence, arousal, "MODERATE", "orange"]
#         elif 4 < valence <= 5 and 5 <= arousal <= 6:
#             return [valence, arousal, "MILD", "yellow"]
#         else:
#             return [valence, arousal, "NORMAL", "Drey"]
#     else:
#         return [valence, arousal, "GOOD", "green"]

# def insomnia_metrics(valence, arousal):   
#     if valence < 4 and arousal > 2:
#         if 0 <= valence <= 1.5 and 8.5 <= arousal <= 10:
#             return [valence, arousal, "SEVERE", "red"]
#         elif 1.5 < valence <= 3 and 7.5 <= arousal <= 8.5:
#             return [valence, arousal, "MODERATE", "orange"]
#         elif 3 < valence <= 4 and 6 <= arousal <= 7.5:
#             return [valence, arousal, "MILD", "yellow"]
#         else:
#             return [valence, arousal, "NORMAL", "Grey"]
#     else:
#         return [valence, arousal, "GOOD", "green"]

# def anxiety_metrics(valence, arousal):
#     # Severe Anxiety: Very low positive emotion and high arousal
#     if 0 <= valence <= 2 and 8 <= arousal <= 10:
#         return [valence, arousal, "SEVERE", "red"]
    
#     # Moderate Anxiety: Low positive emotion and high-moderate arousal
#     elif 2 < valence <= 3 and 7 <= arousal <= 8:
#         return [valence, arousal, "MODERATE", "orange"]
    
#     # Mild Anxiety: Somewhat low positive emotion and moderate arousal
#     elif 3 < valence <= 4 and 6 <= arousal <= 7:
#         return [valence, arousal, "MILD", "yellow"]
    
#     # Normal Anxiety: Low-moderate anxiety levels
#     elif 4 < valence <= 5 and 5 <= arousal <= 6:
#         return [valence, arousal, "NORMAL", "Grey"]
    
#     # Good Emotional State: Higher positive emotion and lower arousal
#     else:
#         return [valence, arousal, "GOOD", "green"]

# # Streamlit UI
# st.title('EEG Disease Detection and Emotional State Analysis')

# st.write("""
#     Upload your EEG data in CSV, Excel, or TXT format to determine the presence of diseases/disorders like stress, insomnia, or anxiety.
# """)

# # Upload file
# uploaded_file = st.file_uploader("Choose a file (CSV, Excel, or TXT)", type=["csv", "xlsx", "txt"])

# if uploaded_file is not None:
#     try:
#         # Determine the file type and read it accordingly
#         file_extension = uploaded_file.name.split('.')[-1].lower()

#         if file_extension == 'csv':
#             df = pd.read_csv(uploaded_file)
#         elif file_extension == 'xlsx':
#             df = pd.read_excel(uploaded_file)
#         elif file_extension == 'txt':
#             # Read a text file assuming whitespace delimiters (can adjust if necessary)
#             text = uploaded_file.read().decode('utf-8')
#             df = pd.read_csv(BytesIO(text.encode()), delim_whitespace=True)

#         # Show a preview of the data
#         st.subheader('Preview of Uploaded Data')
#         st.write(df.head())

#         # Process the data and get the report
#         report = process_eeg_data(df)

#         st.subheader("Disease/Disorder Report")
#         for entry in report:
#             st.markdown(entry, unsafe_allow_html=True)
    
#     except Exception as e:
#         st.error(f"Error processing the file: {e}")










# import streamlit as st
# import pandas as pd
# from joblib import load
# from io import BytesIO
# import matplotlib.pyplot as plt


# # Load model (assumes model is already saved as 'random_forest_model.pkl')
# model = load("random_forest_model.pkl")

# def process_eeg_data(df):
#     """
#     Process the uploaded EEG data using the prediction model.
#     """
#     try:
#         predictions = model.predict(df)
#         report = []
#         for prediction in predictions:
#             valence, arousal = float(prediction[0]), float(prediction[1])
#             report.append(assess_emotions(valence, arousal))
#         return report
#     except Exception as e:
#         return [f"Error in processing data: {e}"]

# def assess_emotions(valence, arousal):
#     """
#     Assess emotional states based on valence and arousal values.
#     """
#     results = {
#         "Stress": stress_metrics(valence, arousal),
#         "Insomnia": insomnia_metrics(valence, arousal),
#         "Anxiety": anxiety_metrics(valence, arousal),
#     }
#     report = ""
#     for condition, result in results.items():
#         report += format_condition_report(condition, result)
#     return report

# def format_condition_report(condition, result):
#     """
#     Format the condition report using HTML and CSS for styling.
#     """
#     valence, arousal, severity, color = result
#     icon = get_condition_icon(severity)
#     return f"""
#     <div style="border: 2px solid {color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
#         <strong>{condition} {icon}</strong><br>
#         Valence: {valence}, Arousal: {arousal}<br>
#         <span style="color: {color}; font-weight: bold;">Severity: {severity}</span>
#     </div>
#     """

# def get_condition_icon(severity):
#     """
#     Get the appropriate emoji icon for the severity level.
#     """
#     icons = {
#         "SEVERE": "🔥",
#         "MODERATE": "⚠️",
#         "MILD": "💛",
#         "NORMAL": "🔄",
#         "GOOD": "🌞",
#     }
#     return icons.get(severity, "✅")

# # Define thresholds for "low" and "high"
# VALENCE_LOW = 3
# AROUSAL_HIGH = 4

# # Helper function to determine the combination
# def determine_combination(valence, arousal):
#     valence_status = "low" if valence <= VALENCE_LOW else "high"
#     arousal_status = "high" if arousal >= AROUSAL_HIGH else "low"
#     return valence_status, arousal_status

# # Stress Metrics
# def stress_metrics(valence, arousal):
#     valence_status, arousal_status = determine_combination(valence, arousal)
    
#     if valence_status == "low" and arousal_status == "high":
#         return [valence, arousal, "SEVERE", "red"]
#     elif valence_status == "high" and arousal_status == "low":
#         return [valence, arousal, "MILD", "yellow"]
#     elif valence_status == "low" and arousal_status == "low":
#         return [valence, arousal,"MODERATE", "orange"]
#     elif valence_status == "high" and arousal_status == "high":
#         return [valence, arousal, "MODERATE", "orange"]
#     return [valence, arousal, "GOOD", "green"]

# # Anxiety Metrics
# def anxiety_metrics(valence, arousal):
#     valence_status, arousal_status = determine_combination(valence, arousal)
    
#     if valence_status == "low" and arousal_status == "high":
#         return [valence, arousal, "SEVERE", "red"]
#     elif valence_status == "high" and arousal_status == "low":
#         return [valence, arousal, "MILD", "yellow"]
#     elif valence_status == "low" and arousal_status == "low":
#         return [valence, arousal, "MODERATE", "orange"]
#     elif valence_status == "high" and arousal_status == "high":
#         return [valence, arousal, "MODERATE", "orange"]
#     return [valence, arousal, "GOOD", "green"]

# # Insomnia Metrics
# def insomnia_metrics(valence, arousal):
#     valence_status, arousal_status = determine_combination(valence, arousal)
    
#     if valence_status == "low" and arousal_status == "high":
#         return [valence, arousal, "SEVERE", "red"]
#     elif valence_status == "high" and arousal_status == "low":
#         return [valence, arousal, "MILD", "yellow"]
#     elif valence_status == "low" and arousal_status == "low":
#         return [valence, arousal, "MODERATE", "orange"]
#     elif valence_status == "high" and arousal_status == "high":
#         return [valence, arousal, "MODERATE", "orange"]
#     return [valence, arousal, "GOOD", "green"]

# # General Function to Check All Metrics
# def evaluate_all_metrics(valence, arousal):
#     stress = stress_metrics(valence, arousal)
#     anxiety = anxiety_metrics(valence, arousal)
#     insomnia = insomnia_metrics(valence, arousal)
#     return {
#         "Stress": stress,
#         "Anxiety": anxiety,
#         "Insomnia": insomnia
#     }


# def create_line_chart(report_data):
#     """
#     Create a line chart for the symptoms based on severity values.
#     Severity is mapped: GOOD=10, NORMAL=8, MILD=6, MODERATE=4, SEVERE=2.
#     """
#     severity_mapping = {"GOOD": 10, "NORMAL": 8, "MILD": 6, "MODERATE": 4, "SEVERE": 2}
#     symptoms = ["Stress", "Insomnia", "Anxiety"]
#     severity_levels = {symptom: [] for symptom in symptoms}

#     for entry in report_data:
#         for symptom in symptoms:
#             severity = entry.split(f"<strong>{symptom}")[1].split("Severity: ")[1].split("</span>")[0].strip()
#             severity_levels[symptom].append(severity_mapping.get(severity, 10))

#     # Aggregate severity values for each symptom
#     avg_severity = {symptom: sum(levels) / len(levels) for symptom, levels in severity_levels.items()}

#     # Plot the line chart
#     plt.figure(figsize=(8, 5))
#     plt.plot(avg_severity.keys(), avg_severity.values(), marker='o', linestyle='-', color='b')

#     plt.title("Average Severity by Symptom", fontsize=14)
#     plt.xlabel("Symptoms", fontsize=12)
#     plt.ylabel("Severity (Good to Worst)", fontsize=12)
#     plt.ylim(0, 12)
#     plt.grid(True, alpha=0.6)

#     st.subheader("Line Chart: Symptoms Severity")
#     st.pyplot(plt)

# def create_client_report(report_data):
#     """
#     Generate a professional-level summary report for the client based on severity data.
#     """
#     summary = {
#         "Stress": [],
#         "Insomnia": [],
#         "Anxiety": []
#     }

#     # Parse severity data
#     for entry in report_data:
#         for symptom in summary.keys():
#             severity = entry.split(f"<strong>{symptom}")[1].split("Severity: ")[1].split("</span>")[0].strip()
#             summary[symptom].append(severity)

#     # Severity mapping to numerical scores
#     severity_mapping = {"GOOD": 10, "NORMAL": 8, "MILD": 6, "MODERATE": 4, "SEVERE": 2}

#     # Generate a narrative report
#     st.subheader("Client Summary Report")
#     st.markdown("### Analysis Overview")
#     st.markdown("Analyzing your EEG signals, we observed the following trends across different symptoms:")

#     recommendations = {}
#     for symptom, severities in summary.items():
#         # Count occurrences of each severity
#         severity_count = pd.Series(severities).value_counts().to_dict()
#         avg_severity_score = sum(severity_mapping[sev] * count for sev, count in severity_count.items()) / len(severities)

#         # Determine severity level
#         overall_severity = (
#             "low" if avg_severity_score > 8 else
#             "moderate" if avg_severity_score > 6 else
#             "high"
#         )

#         # Store recommendations based on severity
#         if overall_severity == "low":
#             recommendations[symptom] = f"You are showing good management of {symptom.lower()} levels. Continue maintaining a healthy lifestyle."
#         elif overall_severity == "moderate":
#             recommendations[symptom] = f"Your {symptom.lower()} levels indicate moderate concern. It’s advisable to adopt lifestyle adjustments like relaxation techniques or consulting with a specialist."
#         else:  # high severity
#             recommendations[symptom] = f"Your {symptom.lower()} levels indicate a high severity. Immediate professional intervention is recommended to address this condition effectively."

#         # Display findings
#         st.markdown(f"**{symptom}:**")
#         st.write(f"- Average Severity Level: **{overall_severity.capitalize()}** ({avg_severity_score:.1f}/10)")
#         st.write(f"- Occurrences by Severity:")
#         for sev, count in severity_count.items():
#             st.write(f"  - {sev}: {count} times")
#         st.write("---")

#     # Provide tailored recommendations
#     st.markdown("### Recommendations")
#     for symptom, advice in recommendations.items():
#         st.markdown(f"**{symptom}:** {advice}")

# # Streamlit UI
# st.title('EEG Disease Detection and Emotional State Analysis')

# st.write("""
#     Upload your EEG data in CSV, Excel, or TXT format to determine the presence of diseases/disorders like stress, insomnia, or anxiety.
# """)

# uploaded_file = st.file_uploader("Choose a file (CSV, Excel, or TXT)", type=["csv", "xlsx", "txt"])

# if uploaded_file:
#     try:
#         file_extension = uploaded_file.name.split('.')[-1].lower()
#         if file_extension == 'csv':
#             df = pd.read_csv(uploaded_file)
#         elif file_extension == 'xlsx':
#             df = pd.read_excel(uploaded_file)
#         elif file_extension == 'txt':
#             text = uploaded_file.read().decode('utf-8')
#             df = pd.read_csv(BytesIO(text.encode()), delim_whitespace=True)

#         st.subheader('Preview of Uploaded Data')
#         st.write(df.head())

#         st.subheader("Disease/Disorder Report")
#         report = process_eeg_data(df)
#         for entry in report:
#             st.markdown(entry, unsafe_allow_html=True)

#         # Add line chart and client report
#         create_line_chart(report)
#         create_client_report(report)

#     except Exception as e:
#         st.error(f"Error processing the file: {e}")



import streamlit as st
import pandas as pd
from joblib import load
from io import BytesIO
import matplotlib.pyplot as plt


# Load model (assumes model is already saved as 'random_forest_model.pkl')
model = load("ANN.pkl")

# Define thresholds for valence and arousal
VALENCE_LOW = 3
AROUSAL_HIGH = 4


# User Authentication
def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "user" and password == "user":
            return True
        else:
            st.error("Invalid username or password")
    return False


# User Details Input Page
def user_details():
    st.title("User Details")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    medical_history = st.text_area("Medical History")
    if st.button("Save Details"):
        st.success("Details saved successfully!")
        return name, age, medical_history
    return None, None, None


# Processing Functions
def determine_combination(valence, arousal):
    valence_status = "low" if valence <= VALENCE_LOW else "high"
    arousal_status = "high" if arousal >= AROUSAL_HIGH else "low"
    return valence_status, arousal_status


def stress_metrics(valence, arousal):
    valence_status, arousal_status = determine_combination(valence, arousal)
    if valence_status == "low" and arousal_status == "high":
        return [valence, arousal, "SEVERE", "red"]
    elif valence_status == "high" and arousal_status == "low":
        return [valence, arousal, "MILD", "yellow"]
    elif valence_status == "low" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "orange"]
    return [valence, arousal, "GOOD", "green"]


def anxiety_metrics(valence, arousal):
    valence_status, arousal_status = determine_combination(valence, arousal)
    if valence_status == "low" and arousal_status == "high":
        return [valence, arousal, "SEVERE", "red"]
    elif valence_status == "high" and arousal_status == "low":
        return [valence, arousal, "MILD", "yellow"]
    elif valence_status == "low" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "orange"]
    elif valence_status == "high" and arousal_status == "high":
        return [valence, arousal, "MODERATE", "orange"]
    return [valence, arousal, "GOOD", "green"]
    # return [valence, arousal, "GOOD", "green"]


def insomnia_metrics(valence, arousal):
    valence_status, arousal_status = determine_combination(valence, arousal)
    if valence_status == "low" and arousal_status == "high":
        return [valence, arousal, "SEVERE", "red"]
    elif valence_status == "high" and arousal_status == "low":
        return [valence, arousal, "MILD", "yellow"]
    elif valence_status == "low" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "orange"]
    elif valence_status == "high" and arousal_status == "high":
        return [valence, arousal, "MODERATE", "orange"]
    return [valence, arousal, "GOOD", "green"]


def alzheimers_emotional_state(valence, arousal):
    valence_status, arousal_status = determine_combination(valence, arousal)
    if valence_status == "low" and arousal_status == "high":
        return [valence, arousal, "MILD", "red"]
    elif valence_status == "high" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "yellow"]
    elif valence_status == "low" and arousal_status == "low":
        return [valence, arousal, "MODERATE", "orange"]
    elif valence_status == "high" and arousal_status == "high":
        return [valence, arousal, "GOOD", "green"]
    return [valence, arousal, "GOOD", "green"]


def process_eeg_data(df):
    try:
        predictions = model.predict(df)
        report = []
        for prediction in predictions:
            valence, arousal = float(prediction[0]), float(prediction[1])
            stress = stress_metrics(valence, arousal)
            anxiety = anxiety_metrics(valence, arousal)
            insomnia = insomnia_metrics(valence, arousal)
            report.append({
                "Valence": valence,
                "Arousal": arousal,
                "Stress": stress,
                "Anxiety": anxiety,
                "Insomnia": insomnia
            })
        return report
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return []


def create_line_chart(report_data):
    symptoms = ["Stress", "Anxiety", "Insomnia"]
    severity_mapping = {"GOOD": 10, "NORMAL": 8, "MILD": 6, "MODERATE": 4, "SEVERE": 2}
    averages = {symptom: 0 for symptom in symptoms}

    for entry in report_data:
        for symptom in symptoms:
            averages[symptom] += severity_mapping.get(entry[symptom][2], 10)
    averages = {k: v / len(report_data) for k, v in averages.items()}

    plt.figure(figsize=(8, 5))
    plt.plot(averages.keys(), averages.values(), marker='o', linestyle='-', color='b')
    plt.title("Average Severity Levels")
    plt.ylabel("Severity (10 = Good, 2 = Severe)")
    plt.grid(True, alpha=0.6)
    st.pyplot(plt)


# Main Function
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a section", ["Login", "User Details", "EEG Analysis"])

    if app_mode == "Login":
        if login():
            st.session_state["authenticated"] = True

    if app_mode == "User Details":
        if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
            st.warning("Please log in first!")
        else:
            user_details()

    if app_mode == "EEG Analysis":
        if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
            st.warning("Please log in first!")
        else:
            st.title("EEG Disease Detection and Emotional State Analysis")
            uploaded_file = st.file_uploader("Upload your EEG data (CSV, Excel, or TXT)", type=["csv", "xlsx", "txt"])

            if uploaded_file:
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    if file_extension == 'csv':
                        df = pd.read_csv(uploaded_file)
                    elif file_extension == 'xlsx':
                        df = pd.read_excel(uploaded_file)
                    elif file_extension == 'txt':
                        text = uploaded_file.read().decode('utf-8')
                        df = pd.read_csv(BytesIO(text.encode()), delim_whitespace=True)

                    st.write("Data Preview")
                    st.write(df.head())

                    report = process_eeg_data(df)
                    if report:
                        st.subheader("Analysis Report")
                        for entry in report:
                            st.write(entry)
                        create_line_chart(report)

                except Exception as e:
                    st.error(f"Error processing the file: {e}")


if __name__ == "__main__":
    main()
