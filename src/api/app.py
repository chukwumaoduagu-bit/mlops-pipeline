# src/api/app.py
import os
import mlflow
import numpy as np
from flask import Flask, request, jsonify

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load model from hardcoded path (matches your mlruns structure)
model_path = "mlruns/1/models/m-bac02099befc4ea4b5bb1b3326da91a9/artifacts"
model = mlflow.sklearn.load_model(model_path)

app = Flask(__name__)

# ✅ Home route
@app.route("/")
def home():
    return "✅ MLOps Prediction API is live! Send POST to /predict"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' in JSON body"}), 400
        features = np.array(data["features"]).reshape(1, -1)
        pred = model.predict(features)[0]
        return jsonify({"prediction": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)