from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the models
with open('models/rf_sr_model.pkl', 'rb') as f:
    rf_sr_model = pickle.load(f)
with open('models/rf_abw_model.pkl', 'rb') as f:
    rf_abw_model = pickle.load(f)
with open('models/xgb_biomass_model.pkl', 'rb') as f:
    xgb_biomass_model = pickle.load(f)
with open('models/xgb_revenue_model.pkl', 'rb') as f:
    xgb_revenue_model = pickle.load(f)

# Define the feature columns
feature_columns = [col for col in pd.read_csv('data/features_df.csv').columns if col not in ['cycle_id', 'SR', 'ABW', 'biomass', 'revenue']]

@app.route('/predict/sr', methods=['POST'])
def predict_sr():
    data = request.get_json()
    features = pd.DataFrame(data, index=[0])
    features = features[feature_columns].astype(float)
    prediction = rf_sr_model.predict(features)
    return jsonify({'Survival Rate Prediction': prediction[0]})

@app.route('/predict/abw', methods=['POST'])
def predict_abw():
    data = request.get_json()
    features = pd.DataFrame(data, index=[0])
    features = features[feature_columns].astype(float)
    prediction = rf_abw_model.predict(features)
    return jsonify({'Average Body Weight Prediction': prediction[0]})

@app.route('/predict/biomass', methods=['POST'])
def predict_biomass():
    data = request.get_json()
    features = pd.DataFrame(data, index=[0])
    features = features[feature_columns].astype(float)
    prediction = xgb_biomass_model.predict(features)
    return jsonify({'Biomass Prediction': prediction[0]})

@app.route('/predict/revenue', methods=['POST'])
def predict_revenue():
    data = request.get_json()
    features = pd.DataFrame(data, index=[0])
    features = features[feature_columns].astype(float)
    prediction = xgb_revenue_model.predict(features)
    return jsonify({'Revenue Prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
