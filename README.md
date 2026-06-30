# 🏦 Loan Approval Prediction using Machine Learning

A Machine Learning web application that predicts whether a loan application is likely to be **Approved** or **Rejected** based on an applicant's financial and personal information.

The project is built using **Python**, **Scikit-learn**, and **Streamlit**, with a trained **Logistic Regression** model deployed through an interactive web interface.

---

## 📌 Project Overview

Financial institutions receive thousands of loan applications every day. This project uses Machine Learning to analyze applicant information and predict loan approval, helping automate and speed up the decision-making process.

The application allows users to enter applicant details through a user-friendly Streamlit interface and instantly receive a prediction along with the model's confidence score.

---

## 🚀 Features

- Predicts Loan Approval Status
- Interactive Streamlit Web Application
- Logistic Regression Classification Model
- Data Preprocessing using Scikit-learn
- Feature Engineering
- One-Hot Encoding
- Standard Scaling
- Real-time Predictions
- Confidence Score Display

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📊 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Categorical Encoding
6. Feature Scaling
7. Model Training
8. Model Evaluation
9. Model Serialization (.pkl)
10. Streamlit Deployment

---

## 📂 Project Structure

```
Loan-Approval-Prediction/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── ohe.pkl
├── requirements.txt
├── loan_app.ipynb
├── loan_approval_data.csv
└── README.md
```

---

## 📥 Input Features

The application uses the following applicant information:

- Applicant Income
- Coapplicant Income
- Age
- Dependents
- Existing Loans
- Savings
- Collateral Value
- Loan Amount
- Loan Term
- Education Level
- Employment Status
- Marital Status
- Loan Purpose
- Property Area
- Gender
- Employer Category
- Credit Score
- Debt-to-Income (DTI) Ratio

---

## ⚙️ Feature Engineering

The following engineered features were created:

- Applicant Income Log Transformation
- Credit Score Squared
- DTI Ratio Squared

These features help improve the predictive performance of the machine learning model.

---

## 🤖 Model Used

**Logistic Regression**

The model was selected after comparing multiple machine learning algorithms and provided reliable performance for the loan approval classification task.

---

## ▶️ How to Run the Project

### Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### Navigate to the project folder

```bash
cd your-repository-name
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📈 Future Improvements

- Deploy on Streamlit Community Cloud
- Compare multiple ML models
- Hyperparameter tuning
- SHAP Explainability
- Improved UI/UX
- Database integration
- User Authentication

---

## 👨‍💻 Author

**Ayush Sahu**

GitHub: https://github.com/Ayushsahu9903

LinkedIn: https://www.linkedin.com/in/ayush-sahu-4751a4178/

---

## ⭐ If you found this project useful

Please consider giving this repository a **Star ⭐** on GitHub.
