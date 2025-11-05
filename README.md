# battery-soc-ml

SOC Prediction – API, Streamlit App, and MQTT Demo

This project predicts a battery's future State of Charge (SOC) using time-series data from lithium‑ion cells (NASA dataset files `B0005.mat`, `B0006.mat`, `B0018.mat`). It provides:
- A FastAPI REST service for SOC prediction
- A Streamlit demo app
- Optional Docker Compose setup (API, UI, and Mosquitto broker)
- Scripts for EDA, model training, and MQTT testing

The trained model and scaler are stored under `acd-aidev/artifacts/`.

## Repository layout
- `acd-aidev/api/`: FastAPI application (`main.py`)
- `acd-aidev/app_streamlit/`: Streamlit UI (`app.py`)
- `acd-aidev/src/soc/`: Data loading and modeling utilities
- `acd-aidev/scripts/`: EDA, training, and MQTT test scripts
- `acd-aidev/data/raw/`: Battery MAT files (NASA)
- `acd-aidev/artifacts/`: Trained `model.joblib` and `scaler.joblib`
- `acd-aidev/docker/`: Dockerfiles and Mosquitto config
- `acd-aidev/docker-compose.yml`: One‑command multi‑service setup

## 1) Setup
Use Python 3.10+ in a virtual environment.

```bash
cd acd-aidev
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Data
Ensure the NASA raw files exist (already provided in the repo):
- `acd-aidev/data/raw/B0005.mat`
- `acd-aidev/data/raw/B0006.mat`
- `acd-aidev/data/raw/B0018.mat`

## 3) (Optional) Exploratory Data Analysis
```bash
cd acd-aidev
python scripts/eda.py
# Outputs (figures) are saved under outputs/eda/ if implemented
```

## 4) Train the model
By default, artifacts are written to `acd-aidev/artifacts/`.

```bash
cd acd-aidev
python scripts/train_model.py \
  --data_dir data \
  --train_batteries B0005 \
  --horizon_s 60 \
  --artifacts_dir artifacts

# After training you should have:
# artifacts/model.joblib
# artifacts/scaler.joblib
```

## 5) Run the API (FastAPI)
```bash
cd acd-aidev
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Health check: `http://localhost:8000/health`
- Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":[3.8,-1.0,25.0,0.8]}'
```

Expected input features: `[voltage_v, current_a, temperature_c, soc]` → outputs predicted SOC at the specified horizon.

## 6) Run the Streamlit app
```bash
cd acd-aidev
streamlit run app_streamlit/app.py
# Default API URL inside the app is http://localhost:8000
```

## 7) Docker Compose (API + UI + MQTT)
```bash
cd acd-aidev
docker compose up --build

# Services:
# API:      http://localhost:8000
# Streamlit: http://localhost:8501
# MQTT:     broker at localhost:1883 (anonymous)
```

## 8) MQTT test
Publish battery messages and test end‑to‑end.

```bash
cd acd-aidev
python scripts/mqtt_test.py

# Example publish (topic/payload the script may use):
# topic:   battery/B0006
# payload: {"voltage_v":3.8, "current_a":-1.0, "temperature_c":25.0, "soc":0.80}
```

## How it works (short)
- SOC labels are derived from current and time (coulomb counting) per cycle.
- A regression model (e.g., GradientBoostingRegressor) predicts SOC at a future horizon (default 60 s) using features `[voltage_v, current_a, temperature_c, soc]`.
- The API loads `model.joblib` and `scaler.joblib` to serve predictions; the Streamlit UI calls the API.

## Notes
- If you retrain with different settings, regenerate `artifacts/` before starting the API/UI.
- Environment variables for MQTT and API URLs can be adjusted in scripts or app configuration as needed.