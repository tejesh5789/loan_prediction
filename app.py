from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('model_decisiontree.pkl', 'rb') as f:
    model = pickle.load(f)

with open('labelEncode_data.pkl', 'rb') as f:
    label_encode = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        income = float(request.form['Income'])
        credit_score = float(request.form['Credit_Score'])
        loan_amount = float(request.form['Loan_Amount'])
        dti_ratio = float(request.form['DTI_Ratio'])
        emp_status_input = request.form['Emp_status']  # employed or unemployed

        # Transform employment status using LabelEncoder
        emp_status = label_encode.transform([emp_status_input])[0]

        # Prepare features for model
        features = np.array([[income, credit_score, loan_amount, dti_ratio, emp_status]])

        # Predict
        prediction = model.predict(features)[0]
        result = "Approved" if prediction == 0 else "Rejected"

        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
