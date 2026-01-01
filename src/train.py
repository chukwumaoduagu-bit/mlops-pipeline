# src/train.py
import os
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

# Set tracking URI (filesystem is OK for now)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("california-housing")

# Load data
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = 10

# Start MLflow run and capture the run object
with mlflow.start_run() as run:
    mlflow.log_param("n_estimators", n_estimators)
    
    # Train model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mlflow.log_metric("mse", mse)
    print(f"✅ Model MSE: {mse:.4f}")
    
    # ✅ Log model using MLflow's built-in method (required for registry)
    mlflow.sklearn.log_model(model, "model")
    print(f"✅ Model logged to MLflow at runs:/{run.info.run_id}/model")
    
    # ✅ Register the model — inside the run context!
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, "CaliforniaHousingModel")
    print(f"✅ Model registered as 'CaliforniaHousingModel'")
    