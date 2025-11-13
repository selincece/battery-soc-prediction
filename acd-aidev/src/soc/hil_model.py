from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


class BatteryDigitalTwin:
    """
    Hardware-in-the-Loop (HIL) Digital Twin model for battery.
    Takes external inputs (time, temperature) and predicts battery outputs (current, voltage, SOC).
    """
    
    def __init__(
        self,
        nominal_capacity_ah: float = 2.0,
        initial_soc: float = 1.0,
        initial_voltage: float = 4.2,
    ):
        """
        Initialize digital twin model.
        
        Args:
            nominal_capacity_ah: Nominal battery capacity in Ah
            initial_soc: Initial state of charge (0-1)
            initial_voltage: Initial voltage in V
        """
        self.nominal_capacity_ah = nominal_capacity_ah
        self.current_soc = initial_soc
        self.current_voltage = initial_voltage
        self.current_time = 0.0
        self.current_temperature = 25.0
        
        # Models for predicting outputs from inputs
        self.voltage_model: Optional[GradientBoostingRegressor] = None
        self.current_model: Optional[GradientBoostingRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        
    def train_from_data(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list] = None,
    ):
        """
        Train the digital twin models from historical data.
        
        Args:
            df: DataFrame with columns: time_s, temperature_c, voltage_v, current_a, soc
            feature_cols: Optional list of feature columns to use
        """
        if feature_cols is None:
            feature_cols = ["time_s", "temperature_c", "soc"]
        
        # Prepare features (inputs: time, temperature, current SOC)
        X = df[feature_cols].values
        
        # Prepare targets (outputs: voltage, current)
        y_voltage = df["voltage_v"].values
        y_current = df["current_a"].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train voltage model
        self.voltage_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.voltage_model.fit(X_scaled, y_voltage)
        
        # Train current model
        self.current_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.current_model.fit(X_scaled, y_current)
    
    def predict(
        self,
        time_s: float,
        temperature_c: float,
        load_profile: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Predict battery outputs given external inputs.
        
        Args:
            time_s: Current time in seconds
            temperature_c: Current temperature in Celsius
            load_profile: Optional load profile (current demand) to simulate
        
        Returns:
            Dictionary with predicted voltage, current, and SOC
        """
        if self.voltage_model is None or self.current_model is None:
            raise ValueError("Models not trained. Call train_from_data() first.")
        
        dt = time_s - self.current_time
        self.current_time = time_s
        self.current_temperature = temperature_c
        
        # If load profile is provided, use it; otherwise predict current
        if load_profile is not None:
            # Use provided load profile
            predicted_current = float(load_profile)
        else:
            # Predict current from time, temperature, and current SOC
            X = np.array([[time_s, temperature_c, self.current_soc]])
            X_scaled = self.scaler.transform(X)
            predicted_current = float(self.current_model.predict(X_scaled)[0])
        
        # Predict voltage from time, temperature, and current SOC
        X = np.array([[time_s, temperature_c, self.current_soc]])
        X_scaled = self.scaler.transform(X)
        predicted_voltage = float(self.voltage_model.predict(X_scaled)[0])
        
        # Update SOC using coulomb counting
        # SOC change = (current * dt) / (capacity * 3600)
        # Negative current (discharge) decreases SOC
        dQ_ah = predicted_current * dt / 3600.0
        dSOC = dQ_ah / self.nominal_capacity_ah
        self.current_soc = np.clip(self.current_soc - dSOC, 0.0, 1.0)
        
        # Update voltage based on SOC (simple relationship)
        # Voltage typically decreases as SOC decreases
        soc_factor = self.current_soc
        predicted_voltage = predicted_voltage * (0.7 + 0.3 * soc_factor)
        
        self.current_voltage = predicted_voltage
        
        return {
            "voltage_v": predicted_voltage,
            "current_a": predicted_current,
            "soc": self.current_soc,
            "time_s": time_s,
            "temperature_c": temperature_c,
        }
    
    def simulate_scenario(
        self,
        time_points: np.ndarray,
        temperature_profile: np.ndarray,
        load_profile: Optional[np.ndarray] = None,
        initial_soc: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Simulate a complete scenario with time and temperature profiles.
        
        Args:
            time_points: Array of time points in seconds
            temperature_profile: Array of temperature values at each time point
            load_profile: Optional array of current demand at each time point
            initial_soc: Optional initial SOC (if None, uses current SOC)
        
        Returns:
            DataFrame with predictions for each time point
        """
        if initial_soc is not None:
            self.current_soc = initial_soc
        
        results = []
        prev_time = time_points[0] if len(time_points) > 0 else 0.0
        
        for i, (t, temp) in enumerate(zip(time_points, temperature_profile)):
            load = load_profile[i] if load_profile is not None else None
            pred = self.predict(t, temp, load)
            results.append(pred)
            prev_time = t
        
        return pd.DataFrame(results)
    
    def reset(self, initial_soc: float = 1.0, initial_voltage: float = 4.2):
        """Reset the digital twin to initial state."""
        self.current_soc = initial_soc
        self.current_voltage = initial_voltage
        self.current_time = 0.0
        self.current_temperature = 25.0


def create_temperature_profile(
    duration_s: float,
    base_temp: float = 25.0,
    variation_type: str = "constant",
    amplitude: float = 5.0,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create temperature profile for HIL testing.
    
    Args:
        duration_s: Duration in seconds
        base_temp: Base temperature in Celsius
        variation_type: 'constant', 'sinusoidal', 'ramp', 'step'
        amplitude: Temperature variation amplitude
        n_points: Number of points
    
    Returns:
        Tuple of (time_points, temperature_profile)
    """
    time_points = np.linspace(0, duration_s, n_points)
    
    if variation_type == "constant":
        temperature_profile = np.full(n_points, base_temp)
    elif variation_type == "sinusoidal":
        temperature_profile = base_temp + amplitude * np.sin(2 * np.pi * time_points / duration_s)
    elif variation_type == "ramp":
        temperature_profile = base_temp + amplitude * (time_points / duration_s)
    elif variation_type == "step":
        mid_point = n_points // 2
        temperature_profile = np.concatenate([
            np.full(mid_point, base_temp),
            np.full(n_points - mid_point, base_temp + amplitude)
        ])
    else:
        temperature_profile = np.full(n_points, base_temp)
    
    return time_points, temperature_profile


def create_load_profile(
    duration_s: float,
    profile_type: str = "constant",
    base_current: float = -1.0,
    amplitude: float = 0.5,
    n_points: int = 100,
) -> np.ndarray:
    """
    Create load (current) profile for HIL testing.
    
    Args:
        duration_s: Duration in seconds
        profile_type: 'constant', 'pulse', 'sinusoidal', 'discharge'
        base_current: Base current in A (negative for discharge)
        amplitude: Current variation amplitude
        n_points: Number of points
    
    Returns:
        Array of current values
    """
    if profile_type == "constant":
        return np.full(n_points, base_current)
    elif profile_type == "pulse":
        current_profile = np.full(n_points, base_current)
        # Add pulses
        pulse_indices = np.linspace(0, n_points - 1, 5, dtype=int)
        for idx in pulse_indices:
            if 0 <= idx < n_points:
                current_profile[idx] = base_current - amplitude
        return current_profile
    elif profile_type == "sinusoidal":
        time_points = np.linspace(0, duration_s, n_points)
        return base_current + amplitude * np.sin(2 * np.pi * time_points / duration_s)
    elif profile_type == "discharge":
        # Gradual discharge
        time_points = np.linspace(0, duration_s, n_points)
        return base_current * (1 - 0.3 * time_points / duration_s)
    else:
        return np.full(n_points, base_current)

