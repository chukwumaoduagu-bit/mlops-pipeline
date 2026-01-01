 # src/api/app.py
import mlflow
import numpy as np
from flask import Flask, request, jsonify
import os

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load model from the actual artifact path
# Since you only have one model, we can hardcode the path for now
model_path = "mlruns/1/models/m-bac02099befc4ea4b5bb1b3326da91a9/artifacts"
model = mlflow.sklearn.load_model(model_path)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({"prediction": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)