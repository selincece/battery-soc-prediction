from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat


@dataclass
class BatterySample:
    battery_id: str
    cycle_index: int
    timestamp_s: float
    voltage_v: float
    current_a: float
    temperature_c: float
    soc: float  # 0..1


def _extract_array(d: dict, *candidate_keys: str) -> np.ndarray:
    for key in candidate_keys:
        if key in d:
            arr = np.asarray(d[key]).squeeze()
            return arr
    raise KeyError(f"None of keys {candidate_keys} found in dict")


def _safe_name(x) -> str:
    try:
        return str(x)
    except Exception:
        return "unknown"


def read_nasa_mat(mat_path: str | Path) -> Dict[int, pd.DataFrame]:
    """
    Parses NASA battery .mat file (B0005/B0006/B0018) into per-cycle DataFrames.

    Returns a dict: cycle_index -> DataFrame with columns:
    ['time_s','voltage_v','current_a','temperature_c']
    """
    mat = loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    stem = Path(mat_path).stem
    root_arr = mat.get(stem)
    if root_arr is None:
        candidates = [k for k in mat.keys() if not k.startswith("__")]
        if not candidates:
            raise ValueError("No battery struct found in mat file")
        root_arr = mat[candidates[0]]

    # MATLAB struct at [0,0] with attribute 'cycle'
    root = root_arr[0, 0]
    if not hasattr(root, "cycle"):
        raise ValueError("MAT file root has no 'cycle' field")

    cycle_nd = root.cycle  # shape (1, N)
    n_cycles = int(cycle_nd.shape[1]) if len(cycle_nd.shape) == 2 else int(len(cycle_nd))

    cycles: Dict[int, pd.DataFrame] = {}
    for idx in range(n_cycles):
        c = cycle_nd[0, idx] if cycle_nd.ndim == 2 else cycle_nd[idx]
        # Each cycle is a MATLAB struct with 'data' field of shape (1,1)
        if not hasattr(c, "data"):
            continue
        data = c.data
        data00 = data[0, 0] if isinstance(data, np.ndarray) and data.size == 1 else data

        # Extract arrays; each is numpy array with shape (1, N)
        try:
            time_s = np.asarray(getattr(data00, "Time")).ravel()
            voltage_v = np.asarray(getattr(data00, "Voltage_measured")).ravel()
            current_a = np.asarray(getattr(data00, "Current_measured")).ravel()
            temperature_c = np.asarray(getattr(data00, "Temperature_measured")).ravel()
        except AttributeError:
            # Skip if any field missing
            continue

        # Some arrays come as shape (1, N); ravel() already flattens
        df = pd.DataFrame({
            "time_s": time_s.astype(float),
            "voltage_v": voltage_v.astype(float),
            "current_a": current_a.astype(float),
            "temperature_c": temperature_c.astype(float),
        }).sort_values("time_s").reset_index(drop=True)

        if not df.empty:
            cycles[idx + 1] = df

    return cycles


def estimate_soc_by_coulomb_counting(df: pd.DataFrame, nominal_capacity_ah: float | None = None) -> pd.Series:
    """
    Estimates SOC within a single discharge/charge sequence using coulomb counting.

    Assumptions:
    - df is a single cycle sorted by time and includes 'time_s' and 'current_a'.
    - If nominal_capacity_ah is None, it is inferred per-cycle from average discharge rate and duration
      and clipped to reasonable bounds.
    Returns SOC in [0,1].
    """
    t = df["time_s"].to_numpy()
    i = df["current_a"].to_numpy()
    dt = np.diff(t, prepend=t[0])
    # Charge delta in Ah for each step: I[A] * dt[s] / 3600
    dQ_ah = i * dt / 3600.0
    # Integrate: cumulative charge added (positive current) or removed (negative current)
    q_ah = np.cumsum(dQ_ah)

    # Infer capacity from total absolute discharge within cycle
    total_discharge_ah = float(np.sum(np.clip(-dQ_ah, 0, None)))
    cap_ah = nominal_capacity_ah or max(0.5, min(5.0, total_discharge_ah * 1.2))

    # SOC baseline at start: assume full at start of discharge if current is negative on average
    avg_i = float(np.mean(i))
    if avg_i < 0:  # discharge
        soc = 1.0 - np.clip(np.cumsum(np.clip(-dQ_ah, 0, None)) / cap_ah, 0, 1)
    else:  # charge or mixed
        soc = np.clip(np.cumsum(np.clip(dQ_ah, 0, None)) / cap_ah, 0, 1)

    return pd.Series(soc, index=df.index, name="soc")


def build_supervised_sequences(
    df: pd.DataFrame,
    horizon_s: float = 60.0,
    min_step_s: float | None = None,
    feature_cols: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for next-horizon SOC prediction using current features.
    Uses nearest index at time + horizon_s for target SOC.
    """
    if feature_cols is None:
        feature_cols = ["voltage_v", "current_a", "temperature_c", "soc"]

    if min_step_s is None:
        # Infer median step
        steps = np.diff(df["time_s"].to_numpy())
        min_step_s = float(np.median(steps)) if len(steps) else 1.0

    t = df["time_s"].to_numpy()
    soc = df["soc"].to_numpy()

    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    target_times = t + horizon_s
    # Two-pointer search for nearest future index
    j = 0
    for i_idx, t_now in enumerate(t):
        t_target = target_times[i_idx]
        while j < len(t) and t[j] < t_target:
            j += 1
        if j >= len(t):
            break
        X_list.append(df.loc[i_idx, feature_cols].to_numpy(dtype=float))
        y_list.append(soc[j])

    if not X_list:
        return np.empty((0, len(feature_cols))), np.empty((0,))

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=float)
    return X, y


def load_batteries_to_supervised(
    data_dir: str | Path,
    batteries: List[str],
    nominal_capacity_ah: float | None = None,
    horizon_s: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Loads selected batteries and returns concatenated (X, y), and counts per battery.
    """
    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    counts: Dict[str, int] = {}
    for b in batteries:
        mat_path = Path(data_dir) / "raw" / f"{b}.mat"
        cycles = read_nasa_mat(mat_path)
        n_samples = 0
        for _, df in cycles.items():
            df = df.copy().sort_values("time_s").reset_index(drop=True)
            df["soc"] = estimate_soc_by_coulomb_counting(df, nominal_capacity_ah)
            X, y = build_supervised_sequences(df, horizon_s=horizon_s)
            if len(X) == 0:
                continue
            X_all.append(X)
            y_all.append(y)
            n_samples += len(y)
        counts[b] = n_samples
    if not X_all:
        return np.empty((0, 4)), np.empty((0,)), counts
    return np.vstack(X_all), np.concatenate(y_all), counts
