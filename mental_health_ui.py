import streamlit as st
import joblib
import numpy as np
from ctransformers import AutoModelForCausalLM
from fpdf import FPDF
import os

# Load trained model and label encoders
model = joblib.load('mental_health_xgb_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load LLaMA model
llama_model = AutoModelForCausalLM.from_pretrained(
    "E:\Agoro AI Project\llama-2-7b-bible-ggml-f16-q4_0.gguf",
    model_type="llama", 
    gpu_layers=0 
)

# Define categorical features
categorical_cols = ['Gender', 'family_history', 'work_interfere', 'remote_work', 'benefits',
                    'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 
                    'mental_health_consequence']

# Function to get user input
def get_user_input():
    inputs = {}
    inputs['Age'] = st.number_input('🔹Age', min_value=18, max_value=100, value=30)
    inputs['Gender'] = st.selectbox('🔹Gender', ['Male', 'Female', 'Other'])
    inputs['family_history'] = st.selectbox('🔹Do you have a family history of mental illness?', ['Yes', 'No'])
    inputs['work_interfere'] = st.selectbox('🔹If you have a mental health condition, do you feel that it interferes with your work?', ['Never', 'Sometimes', 'Often', 'Always'])
    inputs['remote_work'] = st.selectbox('🔹Do you work remotely (outside of an office) at least 50% of the time?', ['Yes', 'No'])
    inputs['benefits'] = st.selectbox('🔹Does your employer provide mental health benefits?', ['Yes', 'No'])
    inputs['care_options'] = st.selectbox('🔹Do you know the options for mental health care your employer provides?', ['Yes', 'No'])
    inputs['wellness_program'] = st.selectbox('🔹Has your employer ever discussed mental health as part of an employee wellness program?', ['Yes', 'No'])
    inputs['seek_help'] = st.selectbox('🔹Does your employer provide resources to learn more about mental health issues and how to seek help?', ['Yes', 'No'])
    inputs['anonymity'] = st.selectbox('🔹Is your anonymity protected if you choose to take advantage of mental health or substance abuse?', ['Yes', 'No'])
    inputs['leave'] = st.selectbox('🔹How easy is it for you to take medical leave for a mental health condition?', ['Very easy', 'Somewhat easy', 'Neutral', 'Somewhat difficult', 'Very difficult'])
    inputs['mental_health_consequence'] = st.selectbox('🔹Do you think that discussing a mental health issue with your employer would have negative consequences?', ['Yes', 'No', 'Maybe'])
    return inputs

# Function to encode user inputs
def encode_inputs(inputs):
    encoded_input = []
    for feature, value in inputs.items():
        if feature in categorical_cols:
            le = label_encoders.get(feature, None)
            if le:
                if value in le.classes_:
                    encoded_value = le.transform([value])[0]
                else:
                    encoded_value = len(le.classes_)
                encoded_input.append(encoded_value)
            else:
                encoded_input.append(value)
        else:
            encoded_input.append(value)
    return np.array(encoded_input).reshape(1, -1)

# Function to generate explanation from LLaMA
def generate_explanation(prediction, probability):
    prompt = f"""
    A mental health model predicts the patient is {prediction} therapy with {probability:.2%} probability. Suggest effective coping strategies, including ways to manage stress specifically in a tech workplace in detail. Outline actionable next steps, such as self-help techniques, workplace accommodations. Also, list relevant government mental health resources, including helplines and counseling services. Use plain text with simple punctuation and avoid any special characters or fancy quotes.
    Do not restate the prompt.
    """
    response = llama_model(prompt, max_new_tokens=300)
    return response

# Function to save report as PDF
def save_report(prediction, probability, explanation):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, "Mental Health Prediction Report", ln=True, align="C")
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Prediction: {prediction}", ln=True)
    pdf.cell(0, 10, f"Probability of Seeking Treatment: {probability:.2%}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "Explanation & Coping Strategies:", ln=True)
    
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, explanation)

    pdf.ln(10)
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "Mental Health Support Helplines:", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, """
    - Vandrevala Foundation Helpline: 1860 266 2345
    - Snehi Mental Health Support: +91 95822 16880
    - iCall Psychosocial Helpline: +91 9152987821
    - AASRA Suicide Prevention: 91-22-27546669
    """)

    pdf_path = "mental_health_report.pdf"
    pdf.output(pdf_path)
    return pdf_path


st.title("👩‍⚕️ Mental Health Treatment Prediction")
user_inputs = get_user_input()

if st.button('Predict'):
    encoded_input = encode_inputs(user_inputs)
    prediction = model.predict(encoded_input)
    probability = model.predict_proba(encoded_input)[0][1]

    # Determine result
    result_text = "Likely to seek treatment" if prediction[0] else "Unlikely to seek treatment"
    
    st.subheader("Prediction Result")
    st.write(f"Probability of seeking treatment: {probability:.2%}")
    st.write(f"Prediction: {result_text}")

    # Get LLaMA explanation
    explanation = generate_explanation(result_text, probability)
    st.subheader("Explanation & Coping Strategies")
    st.write(explanation)

    # Display government mental health helplines
    st.subheader("Mental Health Support Helplines")
    st.write("""
    - **Vandrevala Foundation Helpline**: 1860 266 2345  
    - **Snehi Mental Health Support**: +91 95822 16880  
    - **iCall Psychosocial Helpline**: +91 9152987821  
    - **AASRA Suicide Prevention**: 91-22-27546669  
    """)

    
    pdf_path = save_report(result_text, probability, explanation)
    
    with open(pdf_path, "rb") as file:
        st.download_button(label="Download Report", data=file, file_name="mental_health_report.pdf", mime="application/pdf")
