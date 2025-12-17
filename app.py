import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('loan_approval_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get form values
    applicant_income = float(request.form['ApplicantIncome'])
    coapplicant_income = float(request.form['CoapplicantIncome'])
    loan_amount = float(request.form['LoanAmount'])
    credit_history = float(request.form['Credit_History'])

    # Create DataFrame with SAME columns used in training
    input_data = pd.DataFrame([{
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Credit_History': credit_history
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    result = "Approved ✅" if prediction == 'Y' else "Rejected ❌"

    return render_template(
        'index.html',
        prediction_text=f"Loan Status: {result}"
    )

if __name__ == "__main__":
    app.run(debug=True)
