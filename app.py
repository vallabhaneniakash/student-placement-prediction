"""
Streamlit app for Student Placement Prediction.
Run train_model.py first to create model.pkl, then start this app with:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Placement Prediction", page_icon="🎓")

st.title("🎓 Student Placement Prediction")
st.write("Enter the student's details to predict whether they are likely to get placed.")

# Load the trained model
try:
    saved = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("model.pkl not found. Please run `python train_model.py` first.")
    st.stop()

model = saved["model"]

# Input form
st.subheader("Student Details")
cgpa = st.slider("CGPA", min_value=5.0, max_value=10.0, value=7.5, step=0.1)
iq = st.number_input("IQ Score", min_value=70, max_value=160, value=100, step=1)
communication = st.slider("Communication Skills (1-10)", 1, 10, 6)
technical = st.slider("Technical Skills (1-10)", 1, 10, 6)
internship = st.selectbox("Internship Experience", ["No", "Yes"])

# Prediction
if st.button("Predict Placement"):
    student = pd.DataFrame(
        [[cgpa, iq, communication, technical, 1 if internship == "Yes" else 0]],
        columns=["CGPA", "IQ", "Communication_Skills", "Technical_Skills", "Internship_Experience"],
    )

    prediction = model.predict(student)[0]
    probability = model.predict_proba(student)[0][1]

    if prediction == 1:
        st.success(f"✅ Likely to be PLACED (confidence: {probability * 100:.1f}%)")
    else:
        st.error(f"❌ Unlikely to be placed (placement probability: {probability * 100:.1f}%)")

# Model performance
st.divider()
st.subheader("Model Performance")
st.metric("Accuracy", f"{saved['accuracy'] * 100:.2f}%")

cm = saved["cm"]
cm_df = pd.DataFrame(
    cm,
    index=["Actual: Not Placed", "Actual: Placed"],
    columns=["Predicted: Not Placed", "Predicted: Placed"],
)
st.write("Confusion Matrix:")
st.table(cm_df)
