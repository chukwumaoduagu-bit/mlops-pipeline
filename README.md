# 🚀 End-to-End MLOps Pipeline

This project demonstrates a **production-grade MLOps pipeline** using:
- **Scikit-learn** for training a regression model
- **MLflow** for experiment tracking and model registry
- **Flask** for serving predictions via REST API
- **Docker** for containerized deployment

The pipeline trains a model on the **California Housing dataset**, registers it, and serves real-time predictions in a portable Docker container.

---

## ✨ Features

- ✅ **Model Training**: Automated training with parameter logging
- ✅ **Experiment Tracking**: Metrics, parameters, and artifacts stored via MLflow
- ✅ **Model Registry**: Versioned models (`CaliforniaHousingModel`)
- ✅ **REST API**: Predictions served via `/predict` endpoint
- ✅ **Dockerized**: Fully containerized for reproducible deployment
- ✅ **SQLite Backend**: Uses `mlflow.db` (avoids deprecated filesystem tracking)

---

## 📦 Requirements

- Python 3.12+
- Docker
- `pip` packages: `flask`, `mlflow`, `scikit-learn`, `numpy`

Install dependencies:
```bash
pip install -r requirements.txt<- name: Deploy to Cloud Trigger deploy -->
