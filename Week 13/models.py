import joblib

# Save the models
joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(gb_model, 'gb_model.pkl')

# Save the preprocessing objects
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
