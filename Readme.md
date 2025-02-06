# Mental Health Analysis in the Tech Workplace

## Overview
This project focuses on predicting attitudes towards mental health and assessing the likelihood of individuals seeking therapy based on various workplace and personal factors. It utilizes machine learning models to analyze responses from the "Mental Health in Tech" survey and offers insights into the prevalence and perception of mental health disorders within the tech industry.

## Features
- **Predicts Attitudes Towards Mental Health:** Determines whether an individual is likely or unlikely to seek therapy based on provided survey responses.
- **Natural Language Explanations:** Integrates a local LLaMA model to provide detailed explanations for predictions.
- **Coping Mechanisms & Next Steps:** Suggests personalized strategies for mental well-being.
- **Government & Professional Resources:** Offers mental health helplines and professional support options.
- **User-Friendly Interface:** Built with Streamlit for easy interaction and result interpretation.
- **PDF Report Generation:** Allows users to download a detailed analysis of their mental health prediction.

## Technologies Used
- **Machine Learning:** XGBoost model for prediction
- **NLP Integration:** LLaMA model using `ctransformers` for generating responses
- **Web Framework:** Streamlit
- **Data Processing:** NumPy, Pandas, Scikit-learn
- **PDF Generation:** `fpdf` library

## How It Works
1. **User Input:** The application collects responses related to workplace mental health policies, personal history, and attitudes.
2. **Prediction Model:** The trained XGBoost model predicts whether the user is likely to seek therapy.
3. **LLaMA-Based Explanation:** The local LLaMA model generates a detailed explanation for the prediction, including potential coping mechanisms and next steps.
4. **PDF Report:** Users can download a structured report summarizing their assessment.

## Installation & Setup
### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Install Dependencies

pip install -r requirements.txt


### Run the Application

streamlit run mental_health_ui.py


## Future Improvements
- Enhance the model with additional datasets for improved accuracy.
- Implement a chatbot feature for real-time mental health assistance.
- Expand coping strategies with verified expert advice.

## Contributors
- **Shreya** - Machine Learning, LLM Integration, Streamlit Development

## Acknowledgments
- **Mental Health in Tech Survey** dataset
- Open-source LLaMA model for NLP explanations
- Various mental health organizations providing helplines and resources

## License
This project is open-source and available under the MIT License.

---
This project aims to foster mental health awareness in the tech industry and encourage individuals to seek the support they need.

