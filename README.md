# Battery SOC Prediction - Machine Learning Project

This project predicts a battery's future State of Charge (SOC) using time-series data from lithium-ion cells (NASA dataset files `B0005.mat`, `B0006.mat`, `B0018.mat`). It provides a comprehensive solution with:

- **FastAPI REST service** for SOC prediction
- **Advanced Streamlit dashboard** with multiple analysis pages
- **Live data analysis** with reliability scoring and anomaly detection
- **HIL (Hardware-in-the-Loop) testing** with digital twin simulation
- **Docker Compose setup** (API, UI, and Mosquitto broker)
- Scripts for EDA, model training, and MQTT testing

## 🚀 New Features

### 1. Live Data Analysis & Anomaly Detection
- **80-20 Data Split**: 80% for training, 20% for live testing
- **Real-time Comparison**: Compare model predictions with ground truth
- **Reliability Scoring**: Calculate reliability scores using MAE, RMSE, and R² metrics
- **Synthetic Anomaly Generation**: Create various anomaly types (voltage spike, current spike, temperature spike, etc.)
- **Anomaly Detection**: Statistical methods for anomaly detection and visualization

### 2. HIL (Hardware-in-the-Loop) Testing - Digital Twin
- **Digital Twin Model**: Predict current, voltage, and SOC from time and temperature inputs
- **Real Data Comparison**: Compare HIL model outputs with real battery data
- **Multiple Scenario Testing**: Test with different temperature and load profiles
- **Custom Scenario Creation**: User-defined temperature and load profiles

## 📁 Repository Layout

```
acd-aidev/
├── api/                    # FastAPI application
├── app_streamlit/          # Streamlit UI
│   ├── app.py             # Main page
│   └── pages/
│       ├── 1_Live_Data_Analysis.py  # Live data analysis page
│       └── 2_HIL_Testing.py         # HIL testing page
├── src/soc/                # Source code modules
│   ├── battery_data.py    # Data loading and processing
│   ├── modeling.py        # Model training and prediction
│   ├── live_data.py       # Live data preparation
│   ├── streaming.py        # Data streaming mechanism
│   ├── reliability.py     # Reliability score calculation
│   ├── anomaly.py         # Anomaly generation and detection
│   ├── hil_model.py       # HIL digital twin model
│   └── hil_testing.py     # HIL testing functions
├── scripts/                # Utility scripts
│   ├── eda.py             # Exploratory data analysis
│   ├── train_model.py     # Model training
│   └── mqtt_test.py        # MQTT testing
├── data/raw/               # NASA battery data files
├── artifacts/              # Trained model files
└── docker/                 # Docker configurations
```

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

## 4) Train the Model

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

## 6) Run the Streamlit App

```bash
cd acd-aidev
streamlit run app_streamlit/app.py
# Default API URL inside the app is http://localhost:8000
```

The Streamlit app consists of three pages:

### Main Page
- Simple SOC prediction interface
- API interaction

### Live Data Analysis
- Split data into 80-20 for training and testing
- Compare model predictions with live data
- Calculate reliability scores
- Add and detect synthetic anomalies
- Detailed graphs and metrics

### HIL Testing
- Train digital twin model
- Compare with real data
- Create custom scenarios (temperature and load profiles)
- Multi-scenario testing

## 7) Docker Compose (API + UI + MQTT)

```bash
cd acd-aidev
docker compose up --build

# Services:
# API:      http://localhost:8000
# Streamlit: http://localhost:8501
# MQTT:     broker at localhost:1883 (anonymous)
```

## 8) MQTT Test

Publish battery messages and test end-to-end.

```bash
cd acd-aidev
python scripts/mqtt_test.py

# Example publish (topic/payload the script may use):
# topic:   battery/B0006
# payload: {"voltage_v":3.8, "current_a":-1.0, "temperature_c":25.0, "soc":0.80}
```

## 📈 Usage Scenarios

### Live Data Analysis

1. Navigate to "Live Data Analysis" page in Streamlit
2. Select battery ID from sidebar and click "Veriyi Yükle ve Hazırla" (Load and Prepare Data)
3. Optionally add synthetic anomalies
4. Compare model predictions with real data
5. View reliability scores and anomaly detection results

### HIL Testing

1. Navigate to "HIL Testing" page in Streamlit
2. Click "HIL Modelini Eğit" (Train HIL Model) in sidebar
3. Click "Gerçek Veriyi Yükle" (Load Real Data)
4. Select test scenario:
   - **Test with Real Data**: Uses real temperature and current profiles
   - **Custom Scenario**: Create your own temperature and load profiles
   - **Multi-Scenario Test**: Automatically test multiple scenarios

## 🔬 Technical Details

### Model Features
- **Input Features**: `[voltage_v, current_a, temperature_c, soc]`
- **Target**: Future SOC at a specified horizon (default 60 seconds)
- **Model**: GradientBoostingRegressor
- **SOC Calculation**: Coulomb counting method for per-cycle SOC estimation

### Reliability Score
- MAE (Mean Absolute Error) based score
- RMSE (Root Mean Squared Error) based score
- R² (Coefficient of Determination) based score
- Weighted combination for 0-1 reliability score

### Anomaly Types
- **Voltage Spike**: Sudden voltage increase
- **Voltage Drop**: Sudden voltage decrease
- **Current Spike**: Abnormal current increase
- **Temperature Spike**: Temperature anomaly
- **SOC Jump**: Unrealistic SOC change
- **Sensor Failure**: Sensor failure simulation

### HIL Digital Twin
- **Inputs**: Time (s), Temperature (°C), Optional load profile
- **Outputs**: Voltage (V), Current (A), SOC (0-1)
- **Training**: Learning from real battery data
- **Simulation**: Predicting battery behavior in different scenarios

## How It Works

- SOC labels are derived from current and time (coulomb counting) per cycle.
- A regression model (e.g., GradientBoostingRegressor) predicts SOC at a future horizon (default 60 s) using features `[voltage_v, current_a, temperature_c, soc]`.
- The API loads `model.joblib` and `scaler.joblib` to serve predictions; the Streamlit UI calls the API or uses models directly.

## Notes

- If you retrain with different settings, regenerate `artifacts/` before starting the API/UI.
- When using Docker, ensure `artifacts/` and `data/` directories are properly mounted.
- HIL model training requires sufficient data (at least one cycle).
- Anomaly detection threshold values can be adjusted for statistical methods.
