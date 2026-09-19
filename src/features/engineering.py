"""Horizon-aware feature engineering with no same-month nowcast as the target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CLIMATE_LAGS,
    CLIMATE_VARS,
    DATA_PROCESSED,
    FORECAST_HORIZONS,
    INCIDENCE_COL,
    INCIDENCE_LAGS,
    ROLLING_WINDOW_MONTHS,
    TARGET_PREFIX,
)
from src.logging_utils import get_logger

logger = get_logger(__name__)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic month encodings known at the forecast origin."""
    out = df.copy()
    out["month"] = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_climate_lags_and_rolling(df: pd.DataFrame, max_lag: int | None = None) -> pd.DataFrame:
    """Lag climate and compute a trailing rolling mean that ends at the origin month."""
    out = df.copy()
    lags = CLIMATE_LAGS if max_lag is None else tuple(range(1, max_lag + 1))
    grouped = out.groupby("district_id", sort=False)
    for var in CLIMATE_VARS:
        if var not in out.columns:
            continue
        for lag in lags:
            out[f"{var}_lag_{lag}m"] = grouped[var].shift(lag)
        out[f"{var}_rolling_{ROLLING_WINDOW_MONTHS}m"] = grouped[var].transform(
            lambda x: x.rolling(window=ROLLING_WINDOW_MONTHS, min_periods=ROLLING_WINDOW_MONTHS).mean()
        )
    return out


def add_incidence_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add autoregressive incidence features known at the origin (including lag 0 = y_t)."""
    out = df.copy()
    grouped = out.groupby("district_id", sort=False)[INCIDENCE_COL]
    for lag in INCIDENCE_LAGS:
        if lag == 0:
            out["incidence_lag_0m"] = out[INCIDENCE_COL]
        else:
            out[f"incidence_lag_{lag}m"] = grouped.shift(lag)
    return out


def add_forecast_targets(df: pd.DataFrame, horizons: tuple[int, ...] = FORECAST_HORIZONS) -> pd.DataFrame:
    """Create y_{t+h} targets. Features on the same row remain values at origin t."""
    out = df.copy()
    grouped = out.groupby("district_id", sort=False)[INCIDENCE_COL]
    for horizon in horizons:
        out[f"{TARGET_PREFIX}{horizon}"] = grouped.shift(-horizon)
        out[f"target_date_h{horizon}"] = out["date"] + pd.offsets.MonthBegin(horizon)
        out[f"target_month_h{horizon}"] = out[f"target_date_h{horizon}"].dt.month
        # Seasonal naive for y_{t+h} is y_{t+h-12}, which is incidence.shift(12 - horizon) at origin t.
        out[f"seasonal_naive_h{horizon}"] = grouped.shift(12 - horizon)
    return out


def build_forecast_frame(
    df: pd.DataFrame,
    max_lag: int | None = None,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
) -> pd.DataFrame:
    """Return a modeling table with origin-time features and lead-h targets."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["district_id", "date"]).reset_index(drop=True)
    initial_len = len(out)
    out = add_calendar_features(out)
    out = add_climate_lags_and_rolling(out, max_lag=max_lag)
    out = add_incidence_lags(out)
    out = add_forecast_targets(out, horizons=horizons)
    feature_na_cols = [c for c in out.columns if c.startswith(tuple(f"{v}_lag_" for v in CLIMATE_VARS)) or c.endswith(f"_rolling_{ROLLING_WINDOW_MONTHS}m") or c.startswith("incidence_lag_")]
    out = out.dropna(subset=feature_na_cols).reset_index(drop=True)
    logger.info(
        "Feature engineering: %s rows in, %s rows after lag/rolling dropna.",
        initial_len,
        len(out),
    )
    return out


def prepare_horizon_frame(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Keep rows with a finite target and add calendar features of the known forecast month."""
    target_col = f"{TARGET_PREFIX}{horizon}"
    month_col = f"target_month_h{horizon}"
    out = df.dropna(subset=[target_col]).copy()
    out["known_target_month_sin"] = np.sin(2 * np.pi * out[month_col] / 12)
    out["known_target_month_cos"] = np.cos(2 * np.pi * out[month_col] / 12)
    return out


def generate_climate_lags(
    input_csv: str = "togo_merged_monthly_panel.csv",
    max_lag: int = 3,
) -> pd.DataFrame:
    """Load the merged panel, build forecast features, and write the modeling dataset."""
    input_path = DATA_PROCESSED / input_csv
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}. Run merge_datasets.py first.")
    df = pd.read_csv(input_path)
    df_clean = build_forecast_frame(df, max_lag=max_lag)
    output_path = DATA_PROCESSED / "togo_final_modeling_dataset.csv"
    df_clean.to_csv(output_path, index=False)
    logger.info("Saved modeling dataset to %s", output_path)
    return df_clean


if __name__ == "__main__":
    generate_climate_lags(max_lag=3)
