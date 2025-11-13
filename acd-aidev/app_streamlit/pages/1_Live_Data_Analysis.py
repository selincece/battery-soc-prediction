import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from soc.anomaly import (
    calculate_anomaly_score,
    detect_anomalies,
    generate_synthetic_anomalies,
    inject_anomalies_into_stream,
)
from soc.live_data import prepare_live_test_sequences, split_data_for_live_testing
from soc.modeling import load_artifacts, predict_soc
from soc.reliability import (
    calculate_reliability_score,
    evaluate_live_predictions,
    rolling_reliability,
)
from soc.streaming import SimulatedLiveStreamer

# Set page config
st.set_page_config(page_title="Live Data Analysis", layout="wide")

# Load model
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

try:
    model, scaler = load_artifacts(ARTIFACTS_DIR)
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler
except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.stop()

st.title("🔴 Canlı Veri Analizi ve Anomali Tespiti")

# Sidebar configuration
st.sidebar.header("Konfigürasyon")
battery_id = st.sidebar.selectbox("Batarya ID", ["B0005", "B0006", "B0018"], index=0)
train_ratio = st.sidebar.slider("Eğitim Oranı", 0.7, 0.9, 0.8, 0.05)
horizon_s = st.sidebar.number_input("Tahmin Ufku (saniye)", 30, 300, 60, 10)

# Anomaly settings
st.sidebar.subheader("Anomali Ayarları")
inject_anomalies = st.sidebar.checkbox("Sentetik Anomali Ekle", value=True)
anomaly_types = st.sidebar.multiselect(
    "Anomali Türleri",
    ["voltage_spike", "voltage_drop", "current_spike", "temperature_spike", "soc_jump"],
    default=["voltage_spike", "current_spike"],
)
anomaly_severity = st.sidebar.slider("Anomali Şiddeti", 0.1, 1.0, 0.3, 0.1)

# Load and prepare data
@st.cache_data
def load_live_test_data(battery_id: str, train_ratio: float, horizon_s: float):
    """Load and split data for live testing."""
    train_df, live_df = split_data_for_live_testing(
        DATA_DIR, battery_id, train_ratio=train_ratio, horizon_s=horizon_s
    )
    X, y, metadata = prepare_live_test_sequences(live_df, horizon_s=horizon_s)
    return train_df, live_df, X, y, metadata

if st.sidebar.button("Veriyi Yükle ve Hazırla"):
    with st.spinner("Veri yükleniyor..."):
        try:
            train_df, live_df, X, y, metadata = load_live_test_data(
                battery_id, train_ratio, horizon_s
            )
            st.session_state["live_df"] = live_df
            st.session_state["X"] = X
            st.session_state["y"] = y
            st.session_state["metadata"] = metadata
            st.session_state["data_loaded"] = True
            st.sidebar.success(f"Veri yüklendi: {len(live_df)} örnek")
        except Exception as e:
            st.error(f"Veri yükleme hatası: {e}")

# Main content
if not st.session_state.get("data_loaded", False):
    st.info("Lütfen sidebar'dan 'Veriyi Yükle ve Hazırla' butonuna tıklayın.")
    st.stop()

live_df = st.session_state["live_df"]
X = st.session_state["X"]
y = st.session_state["y"]
metadata = st.session_state["metadata"]

# Inject anomalies if requested
live_df_with_anomalies = live_df.copy()
if inject_anomalies and len(anomaly_types) > 0:
    with st.spinner("Anomaliler ekleniyor..."):
        for anomaly_type in anomaly_types:
            live_df_with_anomalies = generate_synthetic_anomalies(
                live_df_with_anomalies,
                anomaly_type=anomaly_type,
                severity=anomaly_severity,
                n_anomalies=3,
                random_seed=42,
            )
        
        # Rebuild sequences with anomalies
        from soc.live_data import build_supervised_sequences
        X_anomalous, y_anomalous, metadata_anomalous = prepare_live_test_sequences(
            live_df_with_anomalies, horizon_s=horizon_s
        )
        X = X_anomalous
        y = y_anomalous
        metadata = metadata_anomalous
        st.session_state["anomalies_injected"] = True
        st.session_state["live_df_with_anomalies"] = live_df_with_anomalies

# Make predictions
with st.spinner("Tahminler yapılıyor..."):
    y_pred = predict_soc(model, scaler, X)
    results_df, metrics = evaluate_live_predictions(model, scaler, X, y, metadata)

# Display metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Güvenilirlik Skoru", f"{metrics['reliability_score']:.3f}")
with col2:
    st.metric("MAE", f"{metrics['mae']:.4f}", f"{metrics['mae_percentage']:.2f}%")
with col3:
    st.metric("RMSE", f"{metrics['rmse']:.4f}", f"{metrics['rmse_percentage']:.2f}%")
with col4:
    st.metric("R²", f"{metrics['r2']:.4f}")

# Anomaly detection
if inject_anomalies:
    anomaly_metrics = calculate_anomaly_score(y_pred, y, metadata)
    st.subheader("🔍 Anomali Tespiti")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tespit Edilen Anomaliler", anomaly_metrics["n_anomalies"])
    with col2:
        st.metric("Anomali Oranı", f"{anomaly_metrics['anomaly_rate']*100:.2f}%")
    with col3:
        st.metric("Maksimum Hata", f"{anomaly_metrics['max_error']:.4f}")

# Visualizations
st.subheader("📊 Grafikler")

tab1, tab2, tab3, tab4 = st.tabs(["SOC Karşılaştırması", "Hata Analizi", "Rolling Reliability", "Anomali Gösterimi"])

with tab1:
    fig, ax = plt.subplots(figsize=(12, 6))
    n_points = min(500, len(results_df))
    indices = np.linspace(0, len(results_df) - 1, n_points, dtype=int)
    
    ax.plot(indices, results_df.iloc[indices]["true_soc"], label="Gerçek SOC", alpha=0.7, linewidth=2)
    ax.plot(indices, results_df.iloc[indices]["predicted_soc"], label="Tahmin Edilen SOC", alpha=0.7, linewidth=2)
    ax.fill_between(
        indices,
        results_df.iloc[indices]["true_soc"] - results_df.iloc[indices]["abs_error"],
        results_df.iloc[indices]["true_soc"] + results_df.iloc[indices]["abs_error"],
        alpha=0.2,
        label="Hata Bandı",
    )
    ax.set_xlabel("Örnek İndeksi")
    ax.set_ylabel("SOC")
    ax.set_title("Gerçek vs Tahmin Edilen SOC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab2:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Error over time
    n_points = min(500, len(results_df))
    indices = np.linspace(0, len(results_df) - 1, n_points, dtype=int)
    ax1.plot(indices, results_df.iloc[indices]["error"], alpha=0.7, color="red")
    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Örnek İndeksi")
    ax1.set_ylabel("Hata (Tahmin - Gerçek)")
    ax1.set_title("Tahmin Hatası Zaman Serisi")
    ax1.grid(True, alpha=0.3)
    
    # Error distribution
    ax2.hist(results_df["error"], bins=50, alpha=0.7, edgecolor="black")
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Hata")
    ax2.set_ylabel("Frekans")
    ax2.set_title("Hata Dağılımı")
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)

with tab3:
    rolling_metrics = rolling_reliability(results_df, window_size=50)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_points = min(500, len(rolling_metrics))
    indices = np.linspace(0, len(rolling_metrics) - 1, n_points, dtype=int)
    
    ax.plot(indices, rolling_metrics.iloc[indices]["rolling_reliability"], label="Rolling Reliability", linewidth=2)
    ax.axhline(y=0.8, color="green", linestyle="--", label="İyi Eşik (0.8)")
    ax.axhline(y=0.5, color="orange", linestyle="--", label="Orta Eşik (0.5)")
    ax.set_xlabel("Örnek İndeksi")
    ax.set_ylabel("Güvenilirlik Skoru")
    ax.set_title("Zaman İçinde Rolling Güvenilirlik")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    st.pyplot(fig)

with tab4:
    if inject_anomalies and st.session_state.get("anomalies_injected", False):
        # Detect anomalies
        live_df_with_anomalies = st.session_state.get("live_df_with_anomalies", live_df)
        detected_df = detect_anomalies(live_df_with_anomalies, method="statistical", threshold_std=3.0)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Voltage anomalies
        axes[0, 0].plot(detected_df["voltage_v"], alpha=0.7, label="Voltaj")
        if "is_anomaly" in detected_df.columns:
            anomaly_mask = detected_df["is_anomaly"]
            axes[0, 0].scatter(
                detected_df[anomaly_mask].index,
                detected_df[anomaly_mask]["voltage_v"],
                color="red",
                s=50,
                label="Anomali",
                zorder=5,
            )
        axes[0, 0].set_xlabel("Örnek")
        axes[0, 0].set_ylabel("Voltaj (V)")
        axes[0, 0].set_title("Voltaj Anomalileri")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Current anomalies
        axes[0, 1].plot(detected_df["current_a"], alpha=0.7, label="Akım")
        if "is_anomaly" in detected_df.columns:
            axes[0, 1].scatter(
                detected_df[anomaly_mask].index,
                detected_df[anomaly_mask]["current_a"],
                color="red",
                s=50,
                label="Anomali",
                zorder=5,
            )
        axes[0, 1].set_xlabel("Örnek")
        axes[0, 1].set_ylabel("Akım (A)")
        axes[0, 1].set_title("Akım Anomalileri")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Prediction errors highlighting anomalies
        n_points = min(500, len(results_df))
        indices = np.linspace(0, len(results_df) - 1, n_points, dtype=int)
        errors = results_df.iloc[indices]["abs_error"]
        threshold = np.percentile(errors, 95)  # Top 5% as anomalies
        
        axes[1, 0].plot(indices, errors, alpha=0.7, label="Mutlak Hata")
        axes[1, 0].axhline(y=threshold, color="red", linestyle="--", label=f"Anomali Eşiği ({threshold:.3f})")
        anomaly_indices = indices[errors > threshold]
        if len(anomaly_indices) > 0:
            axes[1, 0].scatter(
                anomaly_indices,
                errors[errors > threshold],
                color="red",
                s=50,
                label="Tespit Edilen Anomaliler",
                zorder=5,
            )
        axes[1, 0].set_xlabel("Örnek İndeksi")
        axes[1, 0].set_ylabel("Mutlak Hata")
        axes[1, 0].set_title("Tahmin Hatalarından Anomali Tespiti")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reliability score with anomalies
        rolling_rel = rolling_reliability(results_df, window_size=50)
        axes[1, 1].plot(rolling_rel["rolling_reliability"], alpha=0.7, label="Güvenilirlik")
        axes[1, 1].axhline(y=0.5, color="red", linestyle="--", label="Düşük Güvenilirlik Eşiği")
        axes[1, 1].set_xlabel("Örnek İndeksi")
        axes[1, 1].set_ylabel("Güvenilirlik Skoru")
        axes[1, 1].set_title("Güvenilirlik ve Anomali İlişkisi")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Anomali gösterimi için sidebar'dan 'Sentetik Anomali Ekle' seçeneğini aktifleştirin.")

# Data table
with st.expander("📋 Detaylı Sonuçlar Tablosu"):
    st.dataframe(results_df.head(100), use_container_width=True)

