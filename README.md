# ❤️ Heart Disease Risk Prediction System

A Machine Learning web application that predicts the probability of heart disease using a Logistic Regression model trained on the UCI Heart Disease Dataset.

The application allows users to enter patient health information, receive a heart disease risk prediction, view the prediction probability, understand the model's decision through SHAP Explainable AI, and store prediction history for future reference.

---

## 🌐 Live Demo

🔗 https://heart-disease-prediction-system-by-sumit.streamlit.app

---

## 📸 Application Preview

> *(Add a screenshot of your application here after deployment.)*

---

## ✨ Features

- ❤️ Heart disease risk prediction
- 📊 Prediction probability visualization
- ⚠️ Risk level classification
- 🧠 SHAP Explainable AI visualization
- 📝 Rule-based explanation of contributing factors
- 📅 Automatic prediction history with date and time
- 📥 Download prediction history as CSV
- 🗑️ Clear prediction history
- 📱 Responsive and user-friendly Streamlit interface

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Logistic Regression

### Data Processing
- Pandas
- NumPy

### Explainable AI
- SHAP

### Deployment
- Streamlit Community Cloud

---

## 📂 Dataset

**Dataset:** UCI Heart Disease Dataset

The dataset contains patient health parameters such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- ST Slope

These features are used by the trained model to predict the likelihood of heart disease.

---

## 🤖 Machine Learning Model

Model Used:

- Logistic Regression

Training Accuracy:

- **87.13%**

The input data is preprocessed using feature encoding and scaling before prediction.

---

## 🧠 Explainable AI

This application uses **SHAP (SHapley Additive Explanations)** to explain each prediction.

SHAP helps users understand:

- Which features increase the predicted risk
- Which features decrease the predicted risk
- How each input contributes to the final prediction

This makes the model more transparent and interpretable.

---

## 📊 Prediction Output

For every prediction, the application provides:

- Heart Disease Probability
- No Heart Disease Probability
- Risk Level
- Rule-based explanation
- SHAP explanation

---

## 📝 Prediction History

The application automatically stores every prediction along with:

- Date
- Time
- Patient Inputs
- Predicted Risk
- Prediction Probability

Users can:

- View history
- Download history as CSV
- Clear history

---

## 📦 Installation

Clone the repository

```bash
git clone https://github.com/SumitKumarSamal123/Heart-Disease-Prediction-System.git
```

Go to the project folder

```bash
cd Heart-Disease-Prediction-System
```

Install the required libraries

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run App.py
```

---

## 📁 Project Structure

```
Heart-Disease-Prediction-System/
│
├── App.py
├── heart.csv
├── logistic_regression_heart.pkl
├── scaler.pkl
├── columns.pkl
├── feature_names.pkl
├── background_data.pkl
├── requirements.txt
└── README.md
```

---

## ⚠️ Disclaimer

This application is developed for **educational and learning purposes only**.

It should **not** be considered a substitute for professional medical advice, diagnosis, or treatment.

Always consult a qualified healthcare professional for medical decisions.

---

## 👨‍💻 Developer

**Sumit Kumar Samal**

B.Tech Computer Science Engineering Student

Interested in:
- Machine Learning
- Artificial Intelligence
- Data Science

GitHub:
https://github.com/SumitKumarSamal123

---

## ⭐ If you found this project useful

Please consider giving this repository a ⭐ on GitHub.