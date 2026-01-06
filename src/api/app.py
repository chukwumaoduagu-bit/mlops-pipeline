# src/api/app.py
import os
import numpy as np
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor

# Create a fallback dummy model (works everywhere)
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(np.random.rand(100, 8), np.random.rand(100))

# Try to load real model ONLY if artifacts exist (local dev only)
try:
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_path = "mlruns/1/models/m-bac02099befc4ea4b5bb1b3326da91a9/artifacts"
    if os.path.exists(model_path):
        model = mlflow.sklearn.load_model(model_path)
        print("✅ Loaded real trained model")
    else:
        print("⚠️ Model artifacts not found — using dummy model")
except Exception as e:
    print(f"⚠️ Failed to load real model: {e}")
    # Keep dummy model (no action needed)

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ MLOps Prediction API is live! Send POST to /predict"

@app.route("/health")
def health():
    return {"status": "healthy", "model": "loaded"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "features" not in 
            return jsonify({"error": "Missing 'features' in JSON body"}), 400
        features = np.array(data["features"]).reshape(1, -1)
        pred = model.predict(features)[0]
        return jsonify({"prediction": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)