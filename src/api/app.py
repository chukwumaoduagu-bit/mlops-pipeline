 # src/api/app.py
import os
from flask import Flask, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Create model in memory (no file loading)
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(np.random.rand(100, 8), np.random.rand(100))

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        pred = model.predict(features)[0]
        return jsonify({"prediction": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)