from flask import Flask, redirect, url_for, session, request, jsonify
import os
from functools import wraps
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'clave_secreta')

xgb_model = joblib.load('model/xgb_model.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')
scaler = joblib.load('model/scaler.pkl')

def require_oauth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'oauth_token' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("data:", data)
    
    X = FormatData(data)
    
    data_predict = X[0].reshape(1,-1)
    
    probs = xgb_model.predict_proba(data_predict)[0]
    pred_class_idx = np.argmax(probs)
    
    # Decodificar la clase predicha
    type_attack = label_encoder.inverse_transform([pred_class_idx])[0]
    p_accuracy = probs[pred_class_idx]
    is_attack = type_attack != 'normal'
    
    print('\nProbabilidades para todas las clases')
    for class_index, prob in enumerate(probs):
        class_name = label_encoder.inverse_transform([class_index])[0]
        print(f" Clase '{class_name}': {prob:.4f}")
    
    
    return jsonify({
        "p_accuracy": round(float(p_accuracy),4),
        "is_attack": bool(is_attack),
        "type_attack": type_attack if is_attack else 'normal'
    })

def FormatData(data):
    pd.set_option('future.no_silent_downcasting', True)
    df = pd.DataFrame(data, index=[0])
    
    cols_to_drop = ['src_ip', 'dst_ip', 'proto', 'service', 'state',
                'dns_query', 'dns_qtype_name', 'http_method','conn_state',
                'http_refer', 'http_user_agent', 'http_content_type', 'ssl_version']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    X = df.replace('-',0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    X_scaled = scaler.transform(X)
    
    return X_scaled

if __name__ == "__main__":
    app.run(host ="0.0.0.0", port = 5000, debug = True)
