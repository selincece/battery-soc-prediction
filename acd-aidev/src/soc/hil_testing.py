from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from soc.battery_data import estimate_soc_by_coulomb_counting, read_nasa_mat
from soc.hil_model import (
    BatteryDigitalTwin,
    create_load_profile,
    create_temperature_profile,
)


def prepare_hil_test_data(
    data_dir: str | Path,
    battery_id: str,
    cycle_index: Optional[int] = None,
) -> pd.DataFrame:
    """
    Prepare real battery data for HIL testing.
    
    Args:
        data_dir: Data directory
        battery_id: Battery ID (e.g., 'B0005')
        cycle_index: Specific cycle to use (if None, uses first cycle)
    
    Returns:
        DataFrame with real battery data
    """
    mat_path = Path(data_dir) / "raw" / f"{battery_id}.mat"
    cycles = read_nasa_mat(mat_path)
    
    if cycle_index is None:
        cycle_index = min(cycles.keys())
    
    if cycle_index not in cycles:
        raise ValueError(f"Cycle {cycle_index} not found in {battery_id}")
    
    df = cycles[cycle_index].copy()
    df = df.sort_values("time_s").reset_index(drop=True)
    df["soc"] = estimate_soc_by_coulomb_counting(df)
    
    return df


def compare_hil_with_real_data(
    hil_predictions: pd.DataFrame,
    real_data: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compare HIL model predictions with real battery data.
    
    Args:
        hil_predictions: DataFrame from HIL model with columns: voltage_v, current_a, soc, time_s
        real_data: DataFrame with real data with same columns
    
    Returns:
        Dictionary with comparison metrics
    """
    # Align data by time (interpolate if needed)
    common_time = np.intersect1d(
        hil_predictions["time_s"].values,
        real_data["time_s"].values,
    )
    
    if len(common_time) == 0:
        # Interpolate to common time points
        min_time = max(hil_predictions["time_s"].min(), real_data["time_s"].min())
        max_time = min(hil_predictions["time_s"].max(), real_data["time_s"].max())
        common_time = np.linspace(min_time, max_time, min(len(hil_predictions), len(real_data)))
    
    # Interpolate both datasets to common time
    hil_interp = {}
    real_interp = {}
    
    for col in ["voltage_v", "current_a", "soc"]:
        if col in hil_predictions.columns and col in real_data.columns:
            hil_interp[col] = np.interp(
                common_time,
                hil_predictions["time_s"].values,
                hil_predictions[col].values,
            )
            real_interp[col] = np.interp(
                common_time,
                real_data["time_s"].values,
                real_data[col].values,
            )
    
    metrics = {}
    
    for col in ["voltage_v", "current_a", "soc"]:
        if col in hil_interp and col in real_interp:
            pred = hil_interp[col]
            true = real_interp[col]
            
            mae = mean_absolute_error(true, pred)
            rmse = np.sqrt(mean_squared_error(true, pred))
            r2 = r2_score(true, pred)
            
            # Calculate percentage errors (for voltage and SOC)
            if col in ["voltage_v", "soc"]:
                mae_pct = (mae / (np.max(true) - np.min(true))) * 100 if np.max(true) != np.min(true) else 0
            else:
                mae_pct = (mae / np.abs(np.mean(true))) * 100 if np.mean(true) != 0 else 0
            
            metrics[f"{col}_mae"] = float(mae)
            metrics[f"{col}_rmse"] = float(rmse)
            metrics[f"{col}_r2"] = float(r2)
            metrics[f"{col}_mae_pct"] = float(mae_pct)
    
    # Overall score (weighted average)
    metrics["overall_score"] = float(
        (metrics.get("voltage_v_r2", 0) + metrics.get("current_a_r2", 0) + metrics.get("soc_r2", 0)) / 3
    )
    
    return metrics


def run_hil_scenario_test(
    hil_model: BatteryDigitalTwin,
    real_data: pd.DataFrame,
    temperature_profile: Optional[np.ndarray] = None,
    load_profile: Optional[np.ndarray] = None,
    initial_soc: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run a complete HIL scenario test and compare with real data.
    
    Args:
        hil_model: Trained BatteryDigitalTwin model
        real_data: Real battery data for comparison
        temperature_profile: Optional temperature profile (if None, uses real data temperature)
        load_profile: Optional load profile (if None, uses real data current)
        initial_soc: Initial SOC (if None, uses real data initial SOC)
    
    Returns:
        Tuple of (hil_predictions, comparison_metrics)
    """
    # Extract time points from real data
    time_points = real_data["time_s"].values
    
    # Use real temperature if not provided
    if temperature_profile is None:
        temperature_profile = real_data["temperature_c"].values
    
    # Use real current as load profile if not provided
    if load_profile is None:
        load_profile = real_data["current_a"].values
    
    # Set initial SOC
    if initial_soc is None:
        initial_soc = real_data["soc"].iloc[0] if len(real_data) > 0 else 1.0
    
    # Reset and run simulation
    hil_model.reset(initial_soc=initial_soc)
    hil_predictions = hil_model.simulate_scenario(
        time_points=time_points,
        temperature_profile=temperature_profile,
        load_profile=load_profile,
        initial_soc=initial_soc,
    )
    
    # Compare with real data
    metrics = compare_hil_with_real_data(hil_predictions, real_data)
    
    return hil_predictions, metrics


def test_multiple_scenarios(
    hil_model: BatteryDigitalTwin,
    real_data: pd.DataFrame,
    scenarios: List[Dict],
) -> pd.DataFrame:
    """
    Test multiple HIL scenarios and return comparison results.
    
    Args:
        hil_model: Trained BatteryDigitalTwin model
        real_data: Real battery data
        scenarios: List of scenario dictionaries with keys:
                  - name: Scenario name
                  - temperature_type: Temperature profile type
                  - load_type: Load profile type
                  - duration_s: Duration in seconds
    
    Returns:
        DataFrame with results for each scenario
    """
    results = []
    
    for scenario in scenarios:
        name = scenario.get("name", "Unknown")
        temp_type = scenario.get("temperature_type", "constant")
        load_type = scenario.get("load_type", "constant")
        duration_s = scenario.get("duration_s", real_data["time_s"].max() - real_data["time_s"].min())
        base_temp = scenario.get("base_temp", real_data["temperature_c"].mean())
        base_current = scenario.get("base_current", real_data["current_a"].mean())
        
        # Create profiles
        time_points, temp_profile = create_temperature_profile(
            duration_s=duration_s,
            base_temp=base_temp,
            variation_type=temp_type,
            n_points=len(real_data),
        )
        
        load_profile = create_load_profile(
            duration_s=duration_s,
            profile_type=load_type,
            base_current=base_current,
            n_points=len(real_data),
        )
        
        # Run test
        hil_pred, metrics = run_hil_scenario_test(
            hil_model,
            real_data,
            temperature_profile=temp_profile,
            load_profile=load_profile,
        )
        
        # Store results
        result = {
            "scenario": name,
            "temperature_type": temp_type,
            "load_type": load_type,
            **metrics,
        }
        results.append(result)
    
    return pd.DataFrame(results)


def train_hil_from_real_data(
    data_dir: str | Path,
    battery_id: str,
    cycles_to_use: Optional[List[int]] = None,
    nominal_capacity_ah: float = 2.0,
) -> BatteryDigitalTwin:
    """
    Train HIL digital twin model from real battery data.
    
    Args:
        data_dir: Data directory
        battery_id: Battery ID
        cycles_to_use: List of cycle indices to use for training (if None, uses all)
        nominal_capacity_ah: Nominal capacity
    
    Returns:
        Trained BatteryDigitalTwin model
    """
    mat_path = Path(data_dir) / "raw" / f"{battery_id}.mat"
    cycles = read_nasa_mat(mat_path)
    
    if cycles_to_use is None:
        cycles_to_use = list(cycles.keys())
    
    # Combine cycles
    dfs = []
    for cycle_idx in cycles_to_use:
        if cycle_idx in cycles:
            df = cycles[cycle_idx].copy()
            df = df.sort_values("time_s").reset_index(drop=True)
            df["soc"] = estimate_soc_by_coulomb_counting(df, nominal_capacity_ah)
            dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No valid cycles found for {battery_id}")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values("time_s").reset_index(drop=True)
    
    # Train model
    hil_model = BatteryDigitalTwin(nominal_capacity_ah=nominal_capacity_ah)
    hil_model.train_from_data(combined_df)
    
    return hil_model

