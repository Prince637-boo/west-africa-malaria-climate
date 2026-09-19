from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def generate_climate_lags(input_csv: str = "togo_climate_monthly.csv", max_lag: int = 3) -> pd.DataFrame:
    """Generate lagged and rolling climate covariates by district."""
    input_path = DATA_PROCESSED / input_csv
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}. Run the climate data extraction first.")

    print(f"Loading climate data from {input_csv}...")
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["district_id", "date"]).reset_index(drop=True)

    climate_vars = ["precipitation_mm", "temp_mean_c", "temp_max_c", "temp_min_c"]
    for var in climate_vars:
        for lag in range(1, max_lag + 1):
            df[f"{var}_lag_{lag}m"] = df.groupby("district_id")[var].shift(lag)
        df[f"{var}_rolling_3m"] = df.groupby("district_id")[var].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    initial_len = len(df)
    df_clean = df.dropna().reset_index(drop=True)
    output_path = DATA_PROCESSED / "togo_features_dataset.csv"
    df_clean.to_csv(output_path, index=False)

    print(f"Feature engineering complete.")
    print(f"Initial rows: {initial_len}")
    print(f"Rows after removing lag NaNs: {len(df_clean)}")
    print(f"Saved to: '{output_path}'")
    return df_clean


if __name__ == "__main__":
    generate_climate_lags(max_lag=3)
