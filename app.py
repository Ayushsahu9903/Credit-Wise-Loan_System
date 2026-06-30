
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

# Load artifacts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("ohe.pkl")

st.title("🏦 Loan Approval Prediction")
st.write("Fill in the applicant details and click **Predict**.")

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income", min_value=0.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
    existing_loans = st.number_input("Existing Loans", min_value=0, value=0)
    savings = st.number_input("Savings", min_value=0.0)
    collateral = st.number_input("Collateral Value", min_value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0)
    loan_term = st.number_input("Loan Term (months)", min_value=1, value=360)

with col2:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    employment = st.selectbox("Employment Status", ["Salaried","Self-employed","Unemployed"])
    marital = st.selectbox("Marital Status", ["Married","Single"])
    purpose = st.selectbox("Loan Purpose", ["Business","Car","Education","Home","Personal"])
    area = st.selectbox("Property Area", ["Rural","Semiurban","Urban"])
    gender = st.selectbox("Gender", ["Female","Male"])
    employer = st.selectbox("Employer Category", ["Government","MNC","Private","Unemployed"])
    credit_score = st.slider("Credit Score",300,900,700)
    dti = st.slider("DTI Ratio",0.0,1.0,0.30)

if st.button("Predict Loan Status", type="primary"):
    edu = 0 if education=="Graduate" else 1

    cat = pd.DataFrame({
        "Employment_Status":[employment],
        "Marital_Status":[marital],
        "Loan_Purpose":[purpose],
        "Property_Area":[area],
        "Gender":[gender],
        "Employer_Category":[employer]
    })

    encoded = ohe.transform(cat)
    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(cat.columns)
    )

    numeric = pd.DataFrame({
        "Applicant_Income":[applicant_income],
        "Coapplicant_Income":[coapplicant_income],
        "Age":[age],
        "Dependents":[dependents],
        "Existing_Loans":[existing_loans],
        "Savings":[savings],
        "Collateral_Value":[collateral],
        "Loan_Amount":[loan_amount],
        "Loan_Term":[loan_term],
        "Education_Level":[edu],
        "DTI_Ratio_sq":[dti**2],
        "Credit_Score_sq":[credit_score**2],
        "Applicant_Income_log":[np.log1p(applicant_income)]
    })

    final_df = pd.concat([numeric, encoded_df], axis=1)

    expected = scaler.feature_names_in_
    final_df = final_df.reindex(columns=expected, fill_value=0)

    X = scaler.transform(final_df)

    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]
        confidence = max(prob)*100
    else:
        confidence = None

    if pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    if confidence is not None:
        st.metric("Confidence", f"{confidence:.2f}%")
