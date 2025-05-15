import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load("financial_inclusion_model.joblib")
country_encoder = joblib.load("country_encoder.joblib")
location_type_encoder = joblib.load("location_type_encoder.joblib")
job_type_encoder = joblib.load("job_type_encoder.joblib")
cellphone_access_encoder = joblib.load("cellphone_access_encoder.joblib")
marital_status_encoder = joblib.load("marital_status_encoder.joblib")
relationship_with_head_encoder = joblib.load("relationship_with_head_encoder.joblib")
education_level_encoder = joblib.load("education_level_encoder.joblib")
gender_of_respondent_encoder = joblib.load("gender_of_respondent_encoder.joblib")

@st.cache_resource
def load_model():
    model = joblib.load("financial_inclusion_model.joblib")
    return model

model = load_model()

st.title("💳 Financial Inclusion Predictor")
st.write("Predict whether an individual is likely to have a bank account.")

st.sidebar.header("User Input Features")

country = st.sidebar.selectbox("Country", ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
year = st.sidebar.selectbox("Year", [2016, 2017, 2018])
location_type = st.sidebar.selectbox("Location Type", ['Urban', 'Rural'])
cellphone_access = st.sidebar.selectbox("Cellphone Access", ['Yes', 'No'])
household_size = st.sidebar.slider("Household Size", 1, 20, 3)
age_of_respondent = st.sidebar.slider("age_of_respondent", 16, 100, 30)
gender_of_respondent = st.sidebar.selectbox("gender_of_respondent", ['Male', 'Female'])
relationship_with_head = st.sidebar.selectbox("Relationship with Head", ['Head of Household', 'Spouse', 'Child', 'Parent', 'Other Relative', 'Other non-relatives'])
education_level = st.sidebar.selectbox("Education Level", ['No formal education', 'Primary education', 'Secondary education', 'Tertiary education', 'Vocational training'])
job_type = st.sidebar.selectbox("Job Type", ['Self employed', 'Government Dependent', 'Formally employed Private', 'Formally employed Government', 'Farming and Fishing', 'Remittance Dependent', 'Other Income', 'No Income', 'Informally employed'])
marital_status = st.sidebar.selectbox("marital_status", ['Divorced/Seperated', 'Dont know', 'Married/Living together', 'Single/Never Married', 'Widowed'])


def preprocess_input():
    df = pd.DataFrame({
        'country': country_encoder.transform([country]),
        'year': [year],
        'location_type': location_type_encoder.transform([location_type]),
        'cellphone_access': cellphone_access_encoder.transform([cellphone_access]),
        'household_size': [household_size],
        'age_of_respondent': [age_of_respondent],
        'gender_of_respondent': gender_of_respondent_encoder.transform([gender_of_respondent]),
        'relationship_with_head': relationship_with_head_encoder.transform([relationship_with_head]),
        'education_level': education_level_encoder.transform([education_level]),
        'job_type': job_type_encoder.transform([job_type]),
        'marital_status': marital_status_encoder.transform([marital_status])
    })
    return df

if st.button("Predict"):
    input_df = preprocess_input()

    # Make sure the columns are in the same order used during training
    expected_columns = [
        'country',
        'year',
        'location_type',
        'cellphone_access',
        'household_size',
        'age_of_respondent',
        'gender_of_respondent',
        'relationship_with_head',
        'education_level',
        'job_type',
        'marital_status'
    ]
    input_df = input_df[expected_columns]

    # Optional debug
    st.write("Input DataFrame:")
    st.write(input_df)

    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0].max()

        label = "✅ Has Bank Account" if prediction == 1 else "❌ Does NOT Have Bank Account"
        st.subheader("Prediction Result")
        st.write(label)
        st.write(f"📊 Confidence: {proba * 100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed. You may need to apply preprocessing: {e}")