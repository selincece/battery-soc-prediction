# SOC Prediction Project

Bu proje, NASA lityum-iyon batarya verileri (B0005, B0006, B0018) ile anlık SOC'tan belirli bir ufukta (varsayılan 60 sn) gelecekteki SOC tahminini yapar; FastAPI ile REST servis sunar, Streamlit ile demo arayüz sağlar ve MQTT üzerinden farklı batarya verisi ile testi destekler.

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Veri
- `data/raw/` altında `B0005.mat`, `B0006.mat`, `B0018.mat` mevcut olmalıdır (sağlanmış).

## EDA
```bash
python scripts/eda.py
# Çıktılar: outputs/eda/*.png
```

## Model Eğitimi
```bash
python scripts/train_model.py --data_dir data --train_batteries B0005 --horizon_s 60 --artifacts_dir artifacts
```
- Eğitim sonunda `artifacts/model.joblib` ve `artifacts/scaler.joblib` oluşturulur.

## API (FastAPI)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Sağlık kontrolü: http://localhost:8000/health
# Örnek istek:
# curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
#   -d '{"features":[3.8,-1.0,25.0,0.8]}'
```

## Streamlit Demo
```bash
streamlit run app_streamlit/app.py
# Varsayılan API_URL: http://localhost:8000
```

## MQTT Testi
- Bir MQTT broker'ına bağlanır (compose ile mosquitto kullanabilirsiniz).
```bash
# Öntanımlı: MQTT_BROKER_HOST=localhost MQTT_BROKER_PORT=1883
python scripts/mqtt_test.py
# Yayınla (örnek):
#  topic: battery/B0006
#  payload: {"voltage_v":3.8, "current_a":-1.0, "temperature_c":25.0, "soc":0.80}
```

## Docker Compose
```bash
docker compose up --build
# API:      http://localhost:8000
# Streamlit http://localhost:8501
# MQTT:     localhost:1883 (anonymous)
```

## Notlar ve Varsayımlar
- SOC tahmini için veri içindeki akım ve zaman adımlarından kolomb sayımı ile per-cycle SOC kestirimi yapılır, ardından ufuk tahmini modeli (GradientBoostingRegressor) eğitilir.
- Özellikler: [voltage_v, current_a, temperature_c, soc]. Hedef: horizon sonraki SOC.
- Farklı bataryalar (B0006, B0018) MQTT ile yayınlandığında API üstünden tahmin alınabilir.
