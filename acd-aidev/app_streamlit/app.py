import os
from typing import List

import numpy as np
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="SOC Predictor", layout="centered")
st.title("SOC Tahmini Demo")

st.markdown("Basit bir arayüz ile API'ye istek atıp SOC tahmini alın.")

col1, col2, col3, col4 = st.columns(4)
voltage = col1.number_input("Voltaj (V)", value=3.8, step=0.01)
current = col2.number_input("Akım (A)", value=-1.0, step=0.1)
temp = col3.number_input("Sıcaklık (°C)", value=25.0, step=0.5)
soc_now = col4.slider("Anlık SOC", min_value=0.0, max_value=1.0, value=0.8, step=0.01)

if st.button("Tahmin Al"):
    payload = {"features": [voltage, current, temp, soc_now]}
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        pred = r.json().get("soc_pred")
        st.success(f"Tahmin edilen SOC (gelecek): {pred:.4f}")
        st.progress(min(max(pred, 0.0), 1.0))
    except Exception as e:
        st.error(f"Hata: {e}")

st.divider()
st.subheader("Toplu Tahmin")
count = st.number_input("Örnek sayısı", min_value=1, max_value=256, value=5)
if st.button("Rasgele Toplu Tahmin"):
    X = np.column_stack([
        np.random.uniform(3.2, 4.2, size=count),
        np.random.uniform(-2.0, 2.0, size=count),
        np.random.uniform(15, 40, size=count),
        np.random.uniform(0, 1, size=count),
    ]).tolist()
    try:
        r = requests.post(f"{API_URL}/predict/batch", json={"features": X}, timeout=15)
        r.raise_for_status()
        preds: List[float] = r.json().get("soc_pred", [])
        st.write(preds)
    except Exception as e:
        st.error(f"Hata: {e}")
