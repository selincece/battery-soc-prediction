from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_model(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[GradientBoostingRegressor, StandardScaler, dict]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(Xs, y, test_size=0.2, random_state=random_state)

    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = float(mean_absolute_error(y_val, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

    metrics = {"mae": mae, "rmse": rmse}
    return model, scaler, metrics


def save_artifacts(model, scaler, artifacts_dir: str | Path = "artifacts") -> None:
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(artifacts_dir) / "model.joblib")
    joblib.dump(scaler, Path(artifacts_dir) / "scaler.joblib")


def load_artifacts(artifacts_dir: str | Path = "artifacts") -> Tuple[GradientBoostingRegressor, StandardScaler]:
    model = joblib.load(Path(artifacts_dir) / "model.joblib")
    scaler = joblib.load(Path(artifacts_dir) / "scaler.joblib")
    return model, scaler


def predict_soc(model, scaler, features: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(features)
    return model.predict(Xs)
