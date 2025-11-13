# Battery SOC Prediction - Machine Learning Project

This project predicts a battery's future State of Charge (SOC) using time-series data from lithium-ion cells (NASA dataset files `B0005.mat`, `B0006.mat`, `B0018.mat`). It provides a FastAPI REST service, an advanced Streamlit dashboard, and MQTT support for testing with different battery data.

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


## 🔧 Setup

```bash
cd acd-aidev
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 📊 Data

- The NASA raw files must exist under `data/raw/`: `B0005.mat`, `B0006.mat`, `B0018.mat` (provided).



## 🚀 API (FastAPI)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Health check: `http://localhost:8000/health`
- Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":[3.8,-1.0,25.0,0.8]}'
```

## 🖥️ Streamlit Demo

```bash
streamlit run app_streamlit/app.py
```

The Streamlit app consists of three pages:

### 1. Main Page
- Simple SOC prediction interface
- API interaction

### 2. Live Data Analysis
- Split data into 80-20 for training and testing
- Compare model predictions with live data
- Calculate reliability scores
- Add and detect synthetic anomalies
- Detailed graphs and metrics

### 3. HIL Testing
- Train digital twin model
- Compare with real data
- Create custom scenarios (temperature and load profiles)
- Multi-scenario testing

## 🐳 Docker Compose

```bash
docker compose up --build
```

Services:
- **API**: `http://localhost:8000`
- **Streamlit**: `http://localhost:8501`
- **MQTT**: `localhost:1883` (anonymous)

## 📡 MQTT Test

```bash
python scripts/mqtt_test.py
```

Example publish:
- Topic: `battery/B0006`
- Payload: `{"voltage_v":3.8, "current_a":-1.0, "temperature_c":25.0, "soc":0.80}`

## 📈 Usage Scenarios

### Live Data Analysis

1. Navigate to "Live Data Analysis" page in Streamlit
2. Select battery ID from sidebar and click "Load and Prepare Data" button
3. Optionally add synthetic anomalies
4. Compare model predictions with real data
5. View reliability scores and anomaly detection results

### HIL Testing

1. Navigate to "HIL Testing" page in Streamlit
2. Click "Train HIL Model" button in sidebar
3. Click "Load Real Data" button
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

## 📝 Notes

- When the model is retrained, remember to restart the API and Streamlit
- When using Docker, ensure that `artifacts/` and `data/` directories are properly mounted
- HIL model training requires sufficient data (at least one cycle)
- Anomaly detection threshold values can be adjusted for statistical methods

