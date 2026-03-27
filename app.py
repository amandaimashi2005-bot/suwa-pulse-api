# ============================================================
# SUWA-PULSE Heart Disease Prediction API
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Allows FlutterFlow to connect

# Load the trained model and scaler
model  = pickle.load(open('suwa_pulse_heart_model.pkl', 'rb'))
scaler = pickle.load(open('suwa_pulse_scaler.pkl', 'rb'))

# Home route - just to check API is running
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': '🫀 SUWA-PULSE API is running!',
        'status': 'healthy'
    })

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data sent from FlutterFlow or Firebase
        data = request.get_json()

        # Extract the 13 features
        features = [[
            data['age'],
            data['sex'],
            data['cp'],
            data['trestbps'],
            data['chol'],
            data['fbs'],
            data['restecg'],
            data['thalach'],
            data['exang'],
            data['oldpeak'],
            data['slope'],
            data['ca'],
            data['thal']
        ]]

        # Convert to numpy array
        features = np.array(features)

        # Make prediction
        prediction  = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # Determine risk level
        if probability < 0.35:
            risk_level = "Low"
        elif probability < 0.70:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        # Send back the result
        return jsonify({
            'prediction':   int(prediction),
            'probability':  round(float(probability) * 100, 2),
            'risk_level':   risk_level,
            'message':      'Prediction successful'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)