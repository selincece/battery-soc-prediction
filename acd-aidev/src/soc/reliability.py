from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from soc.modeling import load_artifacts, predict_soc


def calculate_reliability_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_mae: float = 0.05,
    threshold_rmse: float = 0.10,
) -> Dict[str, float]:
    """
    Calculates reliability score based on prediction accuracy.
    
    Args:
        y_true: True SOC values
        y_pred: Predicted SOC values
        threshold_mae: MAE threshold for good predictions (default 5%)
        threshold_rmse: RMSE threshold for good predictions (default 10%)
    
    Returns:
        Dictionary with metrics and reliability score (0-1)
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate percentage errors
    mae_pct = mae * 100  # SOC is 0-1, so MAE * 100 = percentage
    rmse_pct = rmse * 100
    
    # Reliability score: combination of MAE and RMSE performance
    # Score decreases as errors exceed thresholds
    mae_score = max(0.0, 1.0 - (mae / threshold_mae))
    rmse_score = max(0.0, 1.0 - (rmse / threshold_rmse))
    r2_score_norm = max(0.0, (r2 + 1) / 2)  # Normalize R2 from [-1,1] to [0,1]
    
    # Weighted combination
    reliability = 0.4 * mae_score + 0.4 * rmse_score + 0.2 * r2_score_norm
    reliability = max(0.0, min(1.0, reliability))  # Clip to [0, 1]
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mae_percentage": float(mae_pct),
        "rmse_percentage": float(rmse_pct),
        "reliability_score": float(reliability),
        "mae_score": float(mae_score),
        "rmse_score": float(rmse_score),
    }


def compare_predictions_with_ground_truth(
    model,
    scaler,
    X: np.ndarray,
    y_true: np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compares model predictions with ground truth and calculates metrics.
    
    Returns:
        DataFrame with predictions, errors, and metadata
    """
    y_pred = predict_soc(model, scaler, X)
    
    results = pd.DataFrame({
        "predicted_soc": y_pred,
        "true_soc": y_true,
        "error": y_pred - y_true,
        "abs_error": np.abs(y_pred - y_true),
        "error_percentage": np.abs(y_pred - y_true) * 100,
    })
    
    if metadata is not None:
        # Merge metadata if provided
        for col in metadata.columns:
            if col not in results.columns:
                results[col] = metadata[col].values[:len(results)]
    
    return results


def evaluate_live_predictions(
    model,
    scaler,
    X: np.ndarray,
    y_true: np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Evaluates live predictions and returns detailed results with reliability score.
    
    Returns:
        results_df: DataFrame with predictions and errors
        metrics: Dictionary with reliability metrics
    """
    results_df = compare_predictions_with_ground_truth(model, scaler, X, y_true, metadata)
    metrics = calculate_reliability_score(y_true, results_df["predicted_soc"].values)
    
    return results_df, metrics


def rolling_reliability(
    results_df: pd.DataFrame,
    window_size: int = 50,
) -> pd.DataFrame:
    """
    Calculates rolling reliability metrics over a sliding window.
    
    Args:
        results_df: DataFrame with 'predicted_soc' and 'true_soc' columns
        window_size: Size of rolling window
    
    Returns:
        DataFrame with rolling metrics
    """
    rolling_mae = (
        results_df["abs_error"]
        .rolling(window=window_size, min_periods=1)
        .mean()
    )
    rolling_rmse = (
        np.sqrt(
            results_df["error"]
            .rolling(window=window_size, min_periods=1)
            .apply(lambda x: np.mean(x**2))
        )
    )
    
    rolling_reliability = pd.DataFrame({
        "rolling_mae": rolling_mae,
        "rolling_rmse": rolling_rmse,
        "rolling_reliability": 1.0 - np.clip(rolling_mae / 0.05, 0, 1),
    })
    
    return rolling_reliability

