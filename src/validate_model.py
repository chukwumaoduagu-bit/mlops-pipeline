import joblib
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def validate_model():
    model_path = "mlruns/1/models/m-e77df8e09a614f98b3cf62999eea4c82/model.pkl"
    
    if not os.path.exists(model_path):
        print("❌ Model not found!")
        sys.exit(1)
    
    model = joblib.load(model_path)
    test_input = np.array([[8.0, 0.5, 6.0, 1.0, 300.0, 3.0, 35.0, -120.0]])
    
    try:
        prediction = model.predict(test_input)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        sys.exit(1)
    
    if np.isnan(prediction).any():
        print("❌ Prediction contains NaN!")
        sys.exit(1)
    
    if not (-1000 < prediction[0] < 1000):
        print(f"❌ Prediction out of expected range: {prediction[0]}")
        sys.exit(1)
    
    print(f"✅ Model validation passed! Prediction: {prediction[0]:.3f}")

if __name__ == "__main__":
    validate_model()
