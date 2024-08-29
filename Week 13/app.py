from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the pre-trained models
logistic_model = joblib.load('logistic_model.pkl')
rf_model = joblib.load('rf_model.pkl')
gb_model = joblib.load('gb_model.pkl')

# Load the preprocessing objects
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    df = pd.DataFrame([data])
    
    # Preprocessing
    df = pd.get_dummies(df)
    X_imputed = imputer.transform(df)
    X_scaled = scaler.transform(X_imputed)
    
    # Predictions
    pred_logistic = logistic_model.predict(X_scaled)
    pred_rf = rf_model.predict(X_scaled)
    pred_gb = gb_model.predict(X_scaled)
    
    # Decoding the labels
    pred_logistic = label_encoder.inverse_transform(pred_logistic)
    pred_rf = label_encoder.inverse_transform(pred_rf)
    pred_gb = label_encoder.inverse_transform(pred_gb)
    
    return render_template('index.html', 
                           pred_logistic=pred_logistic[0],
                           pred_rf=pred_rf[0],
                           pred_gb=pred_gb[0])

if __name__ == '__main__':
    app.run(debug=True)
