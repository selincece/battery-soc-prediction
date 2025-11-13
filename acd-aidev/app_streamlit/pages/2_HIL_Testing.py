import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from soc.hil_model import (
    BatteryDigitalTwin,
    create_load_profile,
    create_temperature_profile,
)
from soc.hil_testing import (
    compare_hil_with_real_data,
    prepare_hil_test_data,
    run_hil_scenario_test,
    test_multiple_scenarios,
    train_hil_from_real_data,
)

# Set page config
st.set_page_config(page_title="HIL Testing", layout="wide")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

st.title("🔬 Hardware-in-the-Loop (HIL) Testi - Dijital İkiz")

st.markdown("""
Bu sayfa, bataryanın dijital ikizi ile HIL testleri yapmanıza olanak sağlar.
Model, zaman ve sıcaklık gibi dış etkenleri input olarak alır ve 
akım, gerilim ve SOC çıktıları üretir. Sonuçlar gerçek veri ile karşılaştırılır.
""")

# Sidebar configuration
st.sidebar.header("Konfigürasyon")

# Battery selection
battery_id = st.sidebar.selectbox("Batarya ID", ["B0005", "B0006", "B0018"], index=0)
cycle_index = st.sidebar.number_input("Cycle Index (0 = tümü)", 0, 200, 1, 1)

# Model parameters
st.sidebar.subheader("Model Parametreleri")
nominal_capacity = st.sidebar.number_input("Nominal Kapasite (Ah)", 1.0, 5.0, 2.0, 0.1)
initial_soc = st.sidebar.slider("Başlangıç SOC", 0.0, 1.0, 1.0, 0.01)

# Training
st.sidebar.subheader("Model Eğitimi")
if st.sidebar.button("HIL Modelini Eğit"):
    with st.spinner("Model eğitiliyor..."):
        try:
            hil_model = train_hil_from_real_data(
                DATA_DIR,
                battery_id,
                cycles_to_use=None,  # Use all cycles
                nominal_capacity_ah=nominal_capacity,
            )
            st.session_state["hil_model"] = hil_model
            st.session_state["hil_trained"] = True
            st.sidebar.success("Model eğitildi!")
        except Exception as e:
            st.error(f"Model eğitim hatası: {e}")

# Check if model is trained
if not st.session_state.get("hil_trained", False):
    st.info("Lütfen sidebar'dan 'HIL Modelini Eğit' butonuna tıklayın.")
    st.stop()

hil_model = st.session_state["hil_model"]

# Load real data for testing
st.sidebar.subheader("Test Verisi")
if st.sidebar.button("Gerçek Veriyi Yükle"):
    with st.spinner("Veri yükleniyor..."):
        try:
            if cycle_index == 0:
                # Use first cycle
                real_data = prepare_hil_test_data(DATA_DIR, battery_id, cycle_index=None)
            else:
                real_data = prepare_hil_test_data(DATA_DIR, battery_id, cycle_index=cycle_index)
            st.session_state["real_data"] = real_data
            st.session_state["data_loaded"] = True
            st.sidebar.success(f"Veri yüklendi: {len(real_data)} örnek")
        except Exception as e:
            st.error(f"Veri yükleme hatası: {e}")

if not st.session_state.get("data_loaded", False):
    st.info("Lütfen sidebar'dan 'Gerçek Veriyi Yükle' butonuna tıklayın.")
    st.stop()

real_data = st.session_state["real_data"]

# Test scenarios
st.header("📋 Test Senaryoları")

tab1, tab2, tab3 = st.tabs(["Gerçek Veri ile Test", "Özel Senaryo", "Çoklu Senaryo Testi"])

with tab1:
    st.subheader("Gerçek Veri ile Karşılaştırma")
    
    if st.button("Testi Çalıştır"):
        with st.spinner("HIL testi çalıştırılıyor..."):
            hil_model.reset(initial_soc=initial_soc)
            
            # Use real temperature and current from data
            hil_predictions, metrics = run_hil_scenario_test(
                hil_model,
                real_data,
                temperature_profile=None,  # Use real temperature
                load_profile=None,  # Use real current
                initial_soc=initial_soc,
            )
            
            st.session_state["hil_predictions"] = hil_predictions
            st.session_state["hil_metrics"] = metrics
    
    if "hil_predictions" in st.session_state:
        hil_predictions = st.session_state["hil_predictions"]
        metrics = st.session_state["hil_metrics"]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Genel Skor", f"{metrics['overall_score']:.3f}")
        with col2:
            st.metric("Voltaj R²", f"{metrics.get('voltage_v_r2', 0):.3f}")
        with col3:
            st.metric("Akım R²", f"{metrics.get('current_a_r2', 0):.3f}")
        with col4:
            st.metric("SOC R²", f"{metrics.get('soc_r2', 0):.3f}")
        
        # Detailed metrics
        with st.expander("Detaylı Metrikler"):
            metrics_df = pd.DataFrame([metrics]).T
            metrics_df.columns = ["Değer"]
            st.dataframe(metrics_df)
        
        # Visualizations
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Voltage comparison
        n_points = min(500, len(hil_predictions))
        indices = np.linspace(0, len(hil_predictions) - 1, n_points, dtype=int)
        
        axes[0].plot(
            real_data.iloc[indices]["time_s"],
            real_data.iloc[indices]["voltage_v"],
            label="Gerçek Voltaj",
            alpha=0.7,
            linewidth=2,
        )
        axes[0].plot(
            hil_predictions.iloc[indices]["time_s"],
            hil_predictions.iloc[indices]["voltage_v"],
            label="HIL Tahmini",
            alpha=0.7,
            linewidth=2,
            linestyle="--",
        )
        axes[0].set_xlabel("Zaman (s)")
        axes[0].set_ylabel("Voltaj (V)")
        axes[0].set_title("Voltaj Karşılaştırması")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Current comparison
        axes[1].plot(
            real_data.iloc[indices]["time_s"],
            real_data.iloc[indices]["current_a"],
            label="Gerçek Akım",
            alpha=0.7,
            linewidth=2,
        )
        axes[1].plot(
            hil_predictions.iloc[indices]["time_s"],
            hil_predictions.iloc[indices]["current_a"],
            label="HIL Tahmini",
            alpha=0.7,
            linewidth=2,
            linestyle="--",
        )
        axes[1].set_xlabel("Zaman (s)")
        axes[1].set_ylabel("Akım (A)")
        axes[1].set_title("Akım Karşılaştırması")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # SOC comparison
        axes[2].plot(
            real_data.iloc[indices]["time_s"],
            real_data.iloc[indices]["soc"],
            label="Gerçek SOC",
            alpha=0.7,
            linewidth=2,
        )
        axes[2].plot(
            hil_predictions.iloc[indices]["time_s"],
            hil_predictions.iloc[indices]["soc"],
            label="HIL Tahmini",
            alpha=0.7,
            linewidth=2,
            linestyle="--",
        )
        axes[2].set_xlabel("Zaman (s)")
        axes[2].set_ylabel("SOC")
        axes[2].set_title("SOC Karşılaştırması")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)
        
        plt.tight_layout()
        st.pyplot(fig)

with tab2:
    st.subheader("Özel Senaryo Testi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sıcaklık Profili")
        temp_type = st.selectbox(
            "Sıcaklık Tipi",
            ["constant", "sinusoidal", "ramp", "step"],
            index=0,
        )
        base_temp = st.number_input("Temel Sıcaklık (°C)", 15.0, 45.0, 25.0, 1.0)
        temp_amplitude = st.number_input("Sıcaklık Genliği", 0.0, 20.0, 5.0, 1.0)
    
    with col2:
        st.subheader("Yük Profili")
        load_type = st.selectbox(
            "Yük Tipi",
            ["constant", "pulse", "sinusoidal", "discharge"],
            index=0,
        )
        base_current = st.number_input("Temel Akım (A)", -5.0, 5.0, -1.0, 0.1)
        load_amplitude = st.number_input("Yük Genliği", 0.0, 2.0, 0.5, 0.1)
    
    duration_s = st.number_input("Süre (saniye)", 100, 10000, int(real_data["time_s"].max() - real_data["time_s"].min()), 100)
    
    if st.button("Özel Senaryoyu Çalıştır"):
        with st.spinner("Senaryo çalıştırılıyor..."):
            # Create profiles
            time_points, temp_profile = create_temperature_profile(
                duration_s=duration_s,
                base_temp=base_temp,
                variation_type=temp_type,
                amplitude=temp_amplitude,
                n_points=len(real_data),
            )
            
            load_profile = create_load_profile(
                duration_s=duration_s,
                profile_type=load_type,
                base_current=base_current,
                amplitude=load_amplitude,
                n_points=len(real_data),
            )
            
            # Run test
            hil_model.reset(initial_soc=initial_soc)
            hil_predictions, metrics = run_hil_scenario_test(
                hil_model,
                real_data,
                temperature_profile=temp_profile,
                load_profile=load_profile,
                initial_soc=initial_soc,
            )
            
            st.session_state["custom_hil_predictions"] = hil_predictions
            st.session_state["custom_hil_metrics"] = metrics
            st.session_state["temp_profile"] = temp_profile
            st.session_state["load_profile"] = load_profile
    
    if "custom_hil_predictions" in st.session_state:
        hil_predictions = st.session_state["custom_hil_predictions"]
        metrics = st.session_state["custom_hil_metrics"]
        temp_profile = st.session_state["temp_profile"]
        load_profile = st.session_state["load_profile"]
        
        # Display metrics
        st.subheader("Sonuçlar")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Genel Skor", f"{metrics['overall_score']:.3f}")
        with col2:
            st.metric("Voltaj MAE", f"{metrics.get('voltage_v_mae', 0):.4f} V")
        with col3:
            st.metric("SOC MAE", f"{metrics.get('soc_mae', 0):.4f}")
        
        # Visualizations
        fig, axes = plt.subplots(4, 1, figsize=(14, 14))
        
        # Temperature profile
        axes[0].plot(temp_profile, label="Sıcaklık Profili", linewidth=2)
        axes[0].set_ylabel("Sıcaklık (°C)")
        axes[0].set_title("Sıcaklık Profili")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Load profile
        axes[1].plot(load_profile, label="Yük Profili", linewidth=2)
        axes[1].set_ylabel("Akım (A)")
        axes[1].set_title("Yük Profili")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Voltage
        n_points = min(500, len(hil_predictions))
        indices = np.linspace(0, len(hil_predictions) - 1, n_points, dtype=int)
        axes[2].plot(
            hil_predictions.iloc[indices]["time_s"],
            hil_predictions.iloc[indices]["voltage_v"],
            label="HIL Voltaj",
            linewidth=2,
        )
        axes[2].set_ylabel("Voltaj (V)")
        axes[2].set_title("HIL Voltaj Tahmini")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # SOC
        axes[3].plot(
            hil_predictions.iloc[indices]["time_s"],
            hil_predictions.iloc[indices]["soc"],
            label="HIL SOC",
            linewidth=2,
        )
        axes[3].set_xlabel("Zaman (s)")
        axes[3].set_ylabel("SOC")
        axes[3].set_title("HIL SOC Tahmini")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim(0, 1)
        
        plt.tight_layout()
        st.pyplot(fig)

with tab3:
    st.subheader("Çoklu Senaryo Testi")
    
    st.markdown("""
    Farklı sıcaklık ve yük profilleri ile çoklu test senaryoları çalıştırın.
    """)
    
    if st.button("Çoklu Senaryo Testini Çalıştır"):
        with st.spinner("Çoklu senaryo testi çalıştırılıyor..."):
            scenarios = [
                {
                    "name": "Sabit Koşullar",
                    "temperature_type": "constant",
                    "load_type": "constant",
                    "base_temp": 25.0,
                    "base_current": -1.0,
                    "duration_s": int(real_data["time_s"].max() - real_data["time_s"].min()),
                },
                {
                    "name": "Sıcaklık Değişimi",
                    "temperature_type": "sinusoidal",
                    "load_type": "constant",
                    "base_temp": 25.0,
                    "base_current": -1.0,
                    "duration_s": int(real_data["time_s"].max() - real_data["time_s"].min()),
                },
                {
                    "name": "Yük Değişimi",
                    "temperature_type": "constant",
                    "load_type": "pulse",
                    "base_temp": 25.0,
                    "base_current": -1.0,
                    "duration_s": int(real_data["time_s"].max() - real_data["time_s"].min()),
                },
                {
                    "name": "Kombine Senaryo",
                    "temperature_type": "ramp",
                    "load_type": "discharge",
                    "base_temp": 25.0,
                    "base_current": -1.0,
                    "duration_s": int(real_data["time_s"].max() - real_data["time_s"].min()),
                },
            ]
            
            results_df = test_multiple_scenarios(hil_model, real_data, scenarios)
            st.session_state["scenario_results"] = results_df
    
    if "scenario_results" in st.session_state:
        results_df = st.session_state["scenario_results"]
        
        st.subheader("Senaryo Sonuçları")
        st.dataframe(results_df, use_container_width=True)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Overall scores
        axes[0, 0].bar(results_df["scenario"], results_df["overall_score"])
        axes[0, 0].set_ylabel("Genel Skor")
        axes[0, 0].set_title("Senaryo Genel Skorları")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis="y")
        
        # Voltage R²
        axes[0, 1].bar(results_df["scenario"], results_df["voltage_v_r2"])
        axes[0, 1].set_ylabel("R²")
        axes[0, 1].set_title("Voltaj R² Skorları")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis="y")
        
        # Current R²
        axes[1, 0].bar(results_df["scenario"], results_df["current_a_r2"])
        axes[1, 0].set_ylabel("R²")
        axes[1, 0].set_title("Akım R² Skorları")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis="y")
        
        # SOC R²
        axes[1, 1].bar(results_df["scenario"], results_df["soc_r2"])
        axes[1, 1].set_ylabel("R²")
        axes[1, 1].set_title("SOC R² Skorları")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        st.pyplot(fig)

