import joblib
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def validate_model():
    model_path = "mlruns/1/models/m-e77df8e09a614f98b3cf62999eea4c82/model.pkl"
…          pip install joblib
      - name: Validate model
        run: python src/validate_model.py
