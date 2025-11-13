from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from soc.battery_data import (
    build_supervised_sequences,
    estimate_soc_by_coulomb_counting,
    read_nasa_mat,
)


def split_data_for_live_testing(
    data_dir: str | Path,
    battery_id: str,
    train_ratio: float = 0.8,
    horizon_s: float = 60.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits battery data into training (80%) and live testing (20%) portions.
    
    Returns:
        train_df: DataFrame with training data (all cycles concatenated)
        live_df: DataFrame with live test data (all cycles concatenated)
    """
    mat_path = Path(data_dir) / "raw" / f"{battery_id}.mat"
    cycles = read_nasa_mat(mat_path)
    
    # Convert cycles to list of DataFrames with cycle info
    cycle_dfs: List[pd.DataFrame] = []
    for cycle_idx, df in cycles.items():
        df = df.copy().sort_values("time_s").reset_index(drop=True)
        df["soc"] = estimate_soc_by_coulomb_counting(df)
        df["cycle_index"] = cycle_idx
        cycle_dfs.append(df)
    
    if not cycle_dfs:
        raise ValueError(f"No cycles found in {battery_id}")
    
    # Split cycles (not individual samples) to maintain temporal integrity
    n_cycles = len(cycle_dfs)
    n_train_cycles = int(n_cycles * train_ratio)
    
    train_cycles = cycle_dfs[:n_train_cycles]
    live_cycles = cycle_dfs[n_train_cycles:]
    
    train_df = pd.concat(train_cycles, ignore_index=True)
    live_df = pd.concat(live_cycles, ignore_index=True)
    
    return train_df, live_df


def prepare_live_test_sequences(
    live_df: pd.DataFrame,
    horizon_s: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepares live test data in the same format as training data.
    
    Returns:
        X: Feature matrix [voltage, current, temperature, soc]
        y: Target SOC values
        metadata: DataFrame with time, cycle_index, and original features
    """
    X, y = build_supervised_sequences(live_df, horizon_s=horizon_s)
    
    # Create metadata DataFrame for tracking
    metadata_rows = []
    t = live_df["time_s"].to_numpy()
    target_times = t + horizon_s
    
    j = 0
    for i_idx, t_now in enumerate(t):
        t_target = target_times[i_idx]
        while j < len(t) and t[j] < t_target:
            j += 1
        if j >= len(t):
            break
        metadata_rows.append({
            "time_s": t_now,
            "target_time_s": t[j],
            "cycle_index": live_df.loc[i_idx, "cycle_index"],
            "voltage_v": live_df.loc[i_idx, "voltage_v"],
            "current_a": live_df.loc[i_idx, "current_a"],
            "temperature_c": live_df.loc[i_idx, "temperature_c"],
            "soc_now": live_df.loc[i_idx, "soc"],
            "soc_target": live_df.loc[j, "soc"],
        })
    
    metadata = pd.DataFrame(metadata_rows)
    
    return X, y, metadata

