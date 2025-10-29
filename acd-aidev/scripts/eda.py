#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from soc.battery_data import estimate_soc_by_coulomb_counting, read_nasa_mat


def main():
    data_dir = Path("data/raw")
    out_dir = Path("outputs/eda")
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for b in ["B0005", "B0006", "B0018"]:
        mat = data_dir / f"{b}.mat"
        if not mat.exists():
            continue
        cycles = read_nasa_mat(mat)
        for idx, df in cycles.items():
            df = df.copy()
            df["soc"] = estimate_soc_by_coulomb_counting(df)
            df["battery"] = b
            df["cycle"] = idx
            dfs.append(df)
    if not dfs:
        print("No data found for EDA.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    corr = df_all[["voltage_v", "current_a", "temperature_c", "soc"]].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Korelasyon Isı Haritası")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=160)
    plt.close()

    g = sns.pairplot(df_all.sample(min(len(df_all), 5000)), vars=["voltage_v", "current_a", "temperature_c", "soc"], hue="battery", corner=True, diag_kind="hist")
    g.fig.suptitle("Özellik Çiftleri (Örneklenmiş)", y=1.02)
    g.savefig(out_dir / "pairplot.png", dpi=130)


if __name__ == "__main__":
    main()
