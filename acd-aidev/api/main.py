from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from soc.modeling import load_artifacts, predict_soc

ART_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
MODEL, SCALER = load_artifacts(ART_DIR)

app = FastAPI(title="SOC Prediction API", version="1.0.0")


class PredictRequest(BaseModel):
    # Features order must match training features: [voltage_v, current_a, temperature_c, soc]
    features: List[float] = Field(..., description="[voltage_v, current_a, temperature_c, soc]")


class PredictBatchRequest(BaseModel):
    features: List[List[float]]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    X = np.asarray([req.features], dtype=float)
    y = predict_soc(MODEL, SCALER, X)
    return {"soc_pred": float(y[0])}


@app.post("/predict/batch")
async def predict_batch(req: PredictBatchRequest):
    X = np.asarray(req.features, dtype=float)
    y = predict_soc(MODEL, SCALER, X)
    return {"soc_pred": y.tolist()}
