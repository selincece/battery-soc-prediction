from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def generate_synthetic_anomalies(
    df: pd.DataFrame,
    anomaly_type: str = "voltage_spike",
    severity: float = 0.3,
    n_anomalies: int = 5,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generates synthetic anomalies in battery data.
    
    Args:
        df: Original DataFrame with battery data
        anomaly_type: Type of anomaly ('voltage_spike', 'voltage_drop', 'current_spike', 
                     'temperature_spike', 'soc_jump', 'sensor_failure')
        severity: Severity factor (0.0 to 1.0)
        n_anomalies: Number of anomalies to inject
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with injected anomalies
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    df_anomalous = df.copy()
    n_samples = len(df_anomalous)
    
    if n_samples == 0:
        return df_anomalous
    
    # Select random indices for anomalies
    anomaly_indices = np.random.choice(n_samples, size=min(n_anomalies, n_samples), replace=False)
    
    for idx in anomaly_indices:
        if anomaly_type == "voltage_spike":
            # Sudden voltage increase
            spike_magnitude = df_anomalous.loc[idx, "voltage_v"] * severity
            df_anomalous.loc[idx, "voltage_v"] += spike_magnitude
            # Clip to reasonable range
            df_anomalous.loc[idx, "voltage_v"] = min(5.0, df_anomalous.loc[idx, "voltage_v"])
            
        elif anomaly_type == "voltage_drop":
            # Sudden voltage drop
            drop_magnitude = df_anomalous.loc[idx, "voltage_v"] * severity
            df_anomalous.loc[idx, "voltage_v"] -= drop_magnitude
            # Clip to reasonable range
            df_anomalous.loc[idx, "voltage_v"] = max(2.0, df_anomalous.loc[idx, "voltage_v"])
            
        elif anomaly_type == "current_spike":
            # Abnormal current spike
            spike_magnitude = abs(df_anomalous.loc[idx, "current_a"]) * severity * 2
            sign = 1 if df_anomalous.loc[idx, "current_a"] >= 0 else -1
            df_anomalous.loc[idx, "current_a"] += sign * spike_magnitude
            
        elif anomaly_type == "temperature_spike":
            # Temperature anomaly
            spike_magnitude = df_anomalous.loc[idx, "temperature_c"] * severity
            df_anomalous.loc[idx, "temperature_c"] += spike_magnitude
            # Clip to reasonable range
            df_anomalous.loc[idx, "temperature_c"] = min(60.0, df_anomalous.loc[idx, "temperature_c"])
            
        elif anomaly_type == "soc_jump":
            # Sudden SOC jump (unrealistic change)
            if "soc" in df_anomalous.columns:
                jump_magnitude = severity * 0.5  # Up to 50% jump
                df_anomalous.loc[idx, "soc"] += jump_magnitude
                df_anomalous.loc[idx, "soc"] = np.clip(df_anomalous.loc[idx, "soc"], 0, 1)
                
        elif anomaly_type == "sensor_failure":
            # Sensor failure: set values to zero or NaN
            if np.random.random() < 0.5:
                df_anomalous.loc[idx, "voltage_v"] = 0.0
            else:
                df_anomalous.loc[idx, "current_a"] = 0.0
    
    return df_anomalous


def detect_anomalies(
    df: pd.DataFrame,
    method: str = "statistical",
    threshold_std: float = 3.0,
) -> pd.DataFrame:
    """
    Detects anomalies in battery data.
    
    Args:
        df: DataFrame with battery data
        method: Detection method ('statistical', 'zscore')
        threshold_std: Standard deviation threshold for statistical methods
    
    Returns:
        DataFrame with 'is_anomaly' column added
    """
    df_result = df.copy()
    df_result["is_anomaly"] = False
    
    if method == "statistical" or method == "zscore":
        # Z-score based detection
        for col in ["voltage_v", "current_a", "temperature_c"]:
            if col not in df_result.columns:
                continue
            mean_val = df_result[col].mean()
            std_val = df_result[col].std()
            if std_val > 0:
                z_scores = np.abs((df_result[col] - mean_val) / std_val)
                df_result.loc[z_scores > threshold_std, "is_anomaly"] = True
        
        # Check for unrealistic SOC changes
        if "soc" in df_result.columns:
            soc_diff = df_result["soc"].diff().abs()
            soc_std = soc_diff.std()
            if soc_std > 0:
                df_result.loc[soc_diff > threshold_std * soc_std, "is_anomaly"] = True
    
    return df_result


def calculate_anomaly_score(
    predicted_soc: np.ndarray,
    true_soc: np.ndarray,
    features: pd.DataFrame,
    anomaly_threshold: float = 0.15,
) -> Dict[str, float]:
    """
    Calculates anomaly score based on prediction errors and feature anomalies.
    
    Args:
        predicted_soc: Predicted SOC values
        true_soc: True SOC values
        features: DataFrame with input features
        anomaly_threshold: Error threshold for anomaly detection
    
    Returns:
        Dictionary with anomaly metrics
    """
    errors = np.abs(predicted_soc - true_soc)
    
    # Anomaly detection based on prediction error
    is_anomaly = errors > anomaly_threshold
    n_anomalies = int(np.sum(is_anomaly))
    anomaly_rate = n_anomalies / len(errors) if len(errors) > 0 else 0.0
    
    # Average error for anomalies
    avg_anomaly_error = float(np.mean(errors[is_anomaly])) if n_anomalies > 0 else 0.0
    
    # Feature-based anomaly detection
    feature_anomalies = 0
    if "voltage_v" in features.columns:
        voltage_std = features["voltage_v"].std()
        voltage_mean = features["voltage_v"].mean()
        if voltage_std > 0:
            voltage_z = np.abs((features["voltage_v"] - voltage_mean) / voltage_std)
            feature_anomalies += np.sum(voltage_z > 3.0)
    
    return {
        "n_anomalies": n_anomalies,
        "anomaly_rate": float(anomaly_rate),
        "avg_anomaly_error": avg_anomaly_error,
        "max_error": float(np.max(errors)),
        "feature_anomalies": int(feature_anomalies),
    }


def inject_anomalies_into_stream(
    stream_df: pd.DataFrame,
    anomaly_start_idx: Optional[int] = None,
    anomaly_types: List[str] = None,
    severity: float = 0.3,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Injects anomalies into a data stream at specified locations.
    
    Args:
        stream_df: DataFrame representing the stream
        anomaly_start_idx: Starting index for anomalies (if None, random)
        anomaly_types: List of anomaly types to inject
        severity: Severity of anomalies
    
    Returns:
        Modified DataFrame and list of anomaly indices
    """
    if anomaly_types is None:
        anomaly_types = ["voltage_spike", "current_spike"]
    
    if anomaly_start_idx is None:
        # Inject in the middle third of the stream
        start = len(stream_df) // 3
        end = 2 * len(stream_df) // 3
        anomaly_start_idx = np.random.randint(start, end) if start < end else len(stream_df) // 2
    
    df_with_anomalies = stream_df.copy()
    anomaly_indices = []
    
    # Inject each anomaly type
    for i, anomaly_type in enumerate(anomaly_types):
        idx = anomaly_start_idx + i * (len(stream_df) // (len(anomaly_types) + 1))
        if idx < len(stream_df):
            df_with_anomalies = generate_synthetic_anomalies(
                df_with_anomalies.iloc[[idx]],
                anomaly_type=anomaly_type,
                severity=severity,
                n_anomalies=1,
            )
            # Update the original dataframe
            stream_df.loc[idx] = df_with_anomalies.iloc[0]
            anomaly_indices.append(idx)
    
    return df_with_anomalies, anomaly_indices

