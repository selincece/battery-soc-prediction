#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from soc.battery_data import load_batteries_to_supervised
from soc.modeling import save_artifacts, train_model


def main():
    parser = argparse.ArgumentParser(description="Train SOC prediction model")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory containing raw/*.mat")
    parser.add_argument("--train_batteries", nargs="*", default=["B0005"], help="Batteries to train on")
    parser.add_argument("--horizon_s", type=float, default=60.0, help="Prediction horizon in seconds")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Output directory for model artifacts")
    args = parser.parse_args()

    X, y, counts = load_batteries_to_supervised(args.data_dir, args.train_batteries, horizon_s=args.horizon_s)
    if X.shape[0] == 0:
        raise SystemExit("No samples were generated. Check data and parameters.")

    model, scaler, metrics = train_model(X, y)
    save_artifacts(model, scaler, args.artifacts_dir)

    print("Samples per battery:", counts)
    print("Dataset shape:", X.shape, y.shape)
    print("Validation metrics:", metrics)


if __name__ == "__main__":
    main()
