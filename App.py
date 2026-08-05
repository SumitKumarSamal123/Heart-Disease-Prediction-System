import streamlit as st 
import pandas as pd 
import joblib 
from datetime import datetime
import os
import shap
import matplotlib.pyplot as plt 

model = joblib.load("logistic_regression_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")
heart_df = pd.read_csv("heart.csv")
background_data = joblib.load("background_data.pkl")
explainer = shap.LinearExplainer(model,background_data)
feature_names = joblib.load("feature_names.pkl")

#User friendly names
feature_map = {
    "Age": "Age",
    "RestingBP": "Resting Blood Pressure",
    "Cholesterol": "Cholesterol",
    "FastingBS": "Fasting Blood Sugar",
    "MaxHR": "Maximum Heart Rate",
    "Oldpeak": "ST Depression (Oldpeak)",

    "Sex_M": "Male",

    "ChestPainType_ATA": "Chest Pain - ATA",
    "ChestPainType_NAP": "Chest Pain - NAP",
    "ChestPainType_TA": "Chest Pain - TA",

    "RestingECG_Normal": "Resting ECG - Normal",
    "RestingECG_ST": "Resting ECG - ST",

    "ExerciseAngina_Y": "Exercise-Induced Angina",

    "ST_Slope_Flat": "ST Slope - Flat",
    "ST_Slope_Up": "ST Slope - Up"
}

# For storing the prediction part 
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probability" not in st.session_state:
    st.session_state.probability = None

if "risk" not in st.session_state:
    st.session_state.risk = None

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

if "shap_values" not in st.session_state:
    st.session_state.shap_values = None

# Sidebar
st.sidebar.title("❤️ Heart Disease Predictor")

st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")

st.sidebar.write("Algorithm : Logistic Regression")
st.sidebar.write("Accuracy : 87.13%")
st.sidebar.write("Dataset : Heart Disease Dataset")
st.sidebar.write("Model Version : 1.0")
st.sidebar.write("Training Samples : 918")
st.sidebar.write("Features : 11")
st.sidebar.write("Developer : Sumit Kumar Samal")

st.sidebar.markdown("---")

st.sidebar.info(
    "This application predicts the probability of heart disease "
    "using a trained Machine Learning model."
)

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    <p style="color:gray; font-size:17px;">
    Predict the likelihood of heart disease using a Machine Learning model
    trained on the UCI Heart Disease Dataset.
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("### Enter Patient Details")

col1,col2 = st.columns(2)
with col1:
    age = st.slider("🎂Age🎂",18,100,40)
    sex = st.selectbox("Sex",["M","F"])
    chest_pain = st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
    resting_bp = st.number_input("Resting Blood Pressure",80,200,120)
    cholesterol = st.number_input("🩸Cholesterol (mg/dl)🩸",100,600,200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate",60,220,150)
    exercise_angina = st.selectbox("🏃Exercise-Induced Angina🏃", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    predict = st.button("🔍 Predict Heart Disease", width="stretch")

if predict:
    # Create a raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }
 # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]
    # Generate SHAP values
    shap_values = explainer.shap_values(scaled_input)
    st.session_state.shap_values = shap_values
    st.session_state.prediction = prediction
    st.session_state.probability = probability  

    # Show result
    if probability >= 0.8:
        risk = "💀Very High Risk💀"
    elif probability >= 0.6:
        risk = "🔴High Risk"
    elif probability >= 0.4:
        risk = "🟡Moderate Risk"
    else:
        risk = "🟢Low Risk"

    st.session_state.risk = risk
    st.session_state.prediction_done = True
    
    # Record of Prediction 
    now = datetime.now()
    
    record = {
        "Date" : now.strftime("%d-%m-%Y"),
        "Time" : now.strftime("%I:%M:%S %p"),
        "Age" : age,
        "Sex" : sex,
        "Chest Pain" : chest_pain,
        "Resting BP" : resting_bp,
        "Cholesterol" : cholesterol,
        "Fasting BS" : fasting_bs,
        "Resting ECG" : resting_ecg,
        "Max HR" : max_hr,
        "Exercise Angina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST Slope": st_slope,
        "Prediction": risk,
        "Probability": f"{probability*100:.2f}%"
    }

    record_df = pd.DataFrame([record])
    filename = "prediction_history.csv"
    
    if os.path.exists(filename):
        record_df.to_csv(filename, mode="a", header=False, index=False)
    else:
        record_df.to_csv(filename, index=False)
    

if st.session_state.prediction_done:
    st.subheader("Prediction Result")
    probability = st.session_state.probability
    risk = st.session_state.risk 
    
    st.write("❤️ Heart Disease Probability")
    st.progress(int(probability*100))
    st.write(f"{probability*100:.2f}%")

    st.write("💚 No Heart Disease Probability")
    st.progress(int((1-probability)*100))
    st.write(f"{(1-probability)*100:.2f}%")
    st.sidebar.metric("Model Accuracy","87.13%")

    st.metric(
            label="Risk Level",
            value=risk
        )

    # Storing Information
    reasons = []
    
    if age > 55:
        reasons.append("Older Age (Age > 55)")

    if cholesterol > 240:
        reasons.append("High Cholesterol")

    if resting_bp > 140:
        reasons.append("High Blood Pressure")

    if max_hr < 120:
        reasons.append("Low Maximum Heart Rate")

    if oldpeak > 2:
        reasons.append("High ST Depression")

    if exercise_angina == "Y":
        reasons.append("Exercise-Induced Angina")

    if fasting_bs == 1:
        reasons.append("High Fasting Blood Sugar")

    if len(reasons) == 0:
        reasons.append("No major contributing factors detected from the rule-based explanation.")

    st.write(f"Reasons of {risk}")
    for reason in reasons:
        st.write("•", reason)

    #SHAP
    st.markdown("---")
    st.subheader("🧠 AI Model Explanation (SHAP)")
    shap_df = pd.DataFrame({"Feature": expected_columns,"Contribution": st.session_state.shap_values[0]})
    shap_df["Feature"] = shap_df["Feature"].map(lambda x: feature_map.get(x, x))
    shap_df["Impact"] = shap_df["Contribution"].abs()

    def get_effect(value):
        if value > 0:
            return f"🔴 Increased Risk (+{value:.3f})"
        elif value < 0:
            return f"🟢 Reduced Risk ({value:.3f})"
        else:
            return "⚪ No Significant Effect"

    shap_df["Effect"] = shap_df["Contribution"].apply(get_effect)

    shap_df = (shap_df.sort_values("Impact", ascending=False).head(10).reset_index(drop=True))

    st.dataframe(shap_df[["Feature", "Effect"]],width="stretch")

    st.info(
    "🔍 Features marked in red increased the predicted heart disease risk, "
    "while features marked in green reduced the predicted risk. "
    "These explanations are generated directly from the Machine Learning model using SHAP.")
    
    #Disclaimer
    st.warning(
        "This application is for educational purposes only."
        "It should not be used as medical advice."
    )

    # Prediction History inside the app
    st.markdown("---")

    with st.expander("📊 Prediction History"):
        if os.path.exists("prediction_history.csv"):
            history = pd.read_csv("prediction_history.csv")
            history = history.iloc[::-1].reset_index(drop = True)
            st.dataframe(
                history,
                width="stretch",
                height=300
            )

            col1, col2 = st.columns(2)
            # Prediction Download Button 
            with col1:
                with open("prediction_history.csv","rb") as file:
                    st.download_button(
                        label="📥 Download History",
                        data=file,
                        file_name="prediction_history.csv",
                        mime="text/csv",
                        width="stretch" 
                    )

            # Prediction History clear Button 
            with col2:
                if "confirm_delete" not in st.session_state:
                    st.session_state.confirm_delete = False

                if not st.session_state.confirm_delete:
                    if st.button("🗑️ Clear History", width="stretch"):
                        st.session_state.confirm_delete = True
                        st.rerun()

                if st.session_state.confirm_delete:

                    st.warning("⚠️ This action cannot be undone.")

                    col_yes, col_no = st.columns(2)

                    with col_yes:
                        if st.button("✅ Yes, Delete", width="stretch"):
                            if os.path.exists("prediction_history.csv"):
                                os.remove("prediction_history.csv")

                            st.session_state.confirm_delete = False
                            st.success("History deleted successfully.")
                            st.rerun()

                    with col_no:
                        if st.button("❌ Cancel", width="stretch"):
                            st.session_state.confirm_delete = False
                            st.rerun()

        else:
            st.info("No prediction history available.")

    st.markdown("---")

    st.subheader("⭐ User Feedback")

    st.write("Help us improve this application by sharing your experience.")

    rating = st.radio(
        "Rate your experience",
        ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
        horizontal=True
    )

    comment = st.text_area(
        "Comments (Optional)",
        placeholder="Tell us what you liked or what can be improved..."
    )

    submit_feedback = st.button("📤 Submit Feedback", width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        with st.expander("Developer"):
            st.write("Sumit Kumar Samal")
            st.write("Built using Python, Streamlit and Scikit-learn")

    with col2:
        with st.expander("About this Model"):
            st.write("Algorithm : Logistic Regression")
            st.write("Dataset : UCI Heart Disease Dataset")
            st.write("Accuracy : 87.13%")
            st.write("Training Samples : 918")