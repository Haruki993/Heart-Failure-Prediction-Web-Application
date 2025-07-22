import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle
import os

# Create Flask app
app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the scaler
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Define the feature names and numerical features for preprocessing
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Define feature descriptions for the form
feature_descriptions = {
    'age': 'Age (years)',
    'sex': 'Sex (1 = Male, 0 = Female)',
    'cp': 'Chest Pain Type (0-3):\n0: Asymptomatic, 1: Atypical Angina, 2: Non-anginal Pain, 3: Typical Angina',
    'trestbps': 'Resting Blood Pressure (mm/Hg)',
    'chol': 'Serum Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)',
    'restecg': 'Resting ECG Results (0-2):\n0: Probable/definite left ventricular hypertrophy, 1: Normal, 2: ST-T wave abnormality',
    'thalach': 'Maximum Heart Rate Achieved',
    'exang': 'Exercise Induced Angina (1 = Yes, 0 = No)',
    'oldpeak': 'ST Depression Induced by Exercise Relative to Rest',
    'slope': 'Slope of Peak Exercise ST Segment (0-2):\n0: Downsloping, 1: Flat, 2: Upsloping',
    'ca': 'Number of Major Vessels Colored by Fluoroscopy (0-3)',
    'thal': 'Thalassemia (1-3):\n1: Normal, 2: Fixed Defect, 3: Reversible Defect'
}

@app.route('/')
def home():
    """Render the home page with the input form"""
    return render_template('index.html', feature_descriptions=feature_descriptions)

@app.route('/predict', methods=['POST'])
def predict():
    """Process form data and make a prediction"""
    try:
        # Get form data
        input_data = {}
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None or value == '':
                return render_template('index.html', 
                                      error=f"Please provide a value for {feature_descriptions[feature]}",
                                      feature_descriptions=feature_descriptions)
            try:
                input_data[feature] = float(value)
            except ValueError:
                return render_template('index.html', 
                                      error=f"Invalid value for {feature_descriptions[feature]}. Please enter a number.",
                                      feature_descriptions=feature_descriptions)
        
        # Convert to DataFrame for preprocessing
        input_df = pd.DataFrame([input_data])
        
        # Scale numerical features
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Get risk percentage (probability of class 1)
        risk_percentage = prediction_proba[1] * 100
        
        # Determine result message
        if prediction == 1:
            result = "At Risk"
            message = "The model predicts that you may be at risk for heart disease."
            alert_class = "danger"
        else:
            result = "Not At Risk"
            message = "The model predicts that you are not at risk for heart disease."
            alert_class = "success"
        
        # Return prediction result
        return render_template('index.html', 
                              prediction=True,
                              result=result,
                              risk_percentage=f"{risk_percentage:.2f}%",
                              message=message,
                              alert_class=alert_class,
                              feature_descriptions=feature_descriptions,
                              input_data=input_data)
    
    except Exception as e:
        # Handle any unexpected errors
        return render_template('index.html', 
                              error=f"An error occurred: {str(e)}",
                              feature_descriptions=feature_descriptions)

# API endpoint for programmatic access
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for making predictions"""
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate input data
        if not data or not all(feature in data for feature in feature_names):
            return jsonify({
                'error': 'Missing required features',
                'required_features': feature_names
            }), 400
        
        # Create input DataFrame
        input_data = {feature: float(data[feature]) for feature in feature_names}
        input_df = pd.DataFrame([input_data])
        
        # Scale numerical features
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        # Make prediction
        prediction = int(model.predict(input_df)[0])
        prediction_proba = model.predict_proba(input_df)[0].tolist()
        
        # Return prediction as JSON
        return jsonify({
            'prediction': prediction,
            'prediction_proba': prediction_proba,
            'risk_percentage': f"{prediction_proba[1] * 100:.2f}%",
            'result': "At Risk" if prediction == 1 else "Not At Risk"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)