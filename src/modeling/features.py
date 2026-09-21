"""Annual feature preparation for district-level malaria forecasting."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DISTRICT_ID_COL,
    INCIDENCE_COL,
    TARGET_PROXIES,
    YEAR_COL,
)
from src.logging_utils import get_logger

logger = get_logger(__name__)

# Whitelist pattern: matches explicit lag features and annual rolling features
_ALLOWED_FEATURE_PATTERN = re.compile(r".+_(lag_\d+|roll_\d+y)$")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Strictly select predictive features using a whitelist pattern.
    
    Only features explicitly matching allowed patterns (_lag_X, _roll_Xy) 
    or explicit interaction indices are included.
    Excludes all raw metadata, targets, target proxies, and unverified merged columns.
    """
    valid_cols = [col for col in frame.columns if _ALLOWED_FEATURE_PATTERN.match(col)]
    
    # Explicitly include verified domain features if present
    if "hydro_thermal_index" in frame.columns:
        valid_cols.append("hydro_thermal_index")
    if "pf_incidence_rate_drift" in frame.columns:
        valid_cols.append("pf_incidence_rate_drift")

    # Safety check: ensure target, target proxies, and non-feature columns never enter feature matrix
    banned = {
        INCIDENCE_COL,
        "malaria_incidence",
        "target_delta",
        "iso3",
        "district_name",
        "region",
        *TARGET_PROXIES,
    }
    return sorted([col for col in set(valid_cols) if col not in banned])


def _lag_feature_columns(df: pd.DataFrame) -> list[str]:
    base_cols = [
        "precipitation_sum_mm",
        "rainy_season_precip_mm",
        "precip_anomaly_mm",
        "temp_mean_c",
        "temp_max_c",
        "temp_min_c",
        "relative_humidity_mean",
    ]
    return [col for col in base_cols if col in df.columns]


def prepare_annual_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create annual lag and rolling features at district level without future leakage.
    
    Features at year t strictly use information from years t-1, t-2, etc.
    """
    if df is None:
        dataset_path = DATA_PROCESSED / "merged_dataset.csv"
        if not dataset_path.exists():
            dataset_path = DATA_PROCESSED / "final_malaria_climate_dataset.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Processed dataset not found in {DATA_PROCESSED}")
        df = pd.read_csv(dataset_path)

    working = df.copy()
    if YEAR_COL not in working.columns:
        raise KeyError(f"The dataset must contain the '{YEAR_COL}' column.")
    if DISTRICT_ID_COL not in working.columns:
        raise KeyError(f"The dataset must contain the '{DISTRICT_ID_COL}' column.")

    working = working.sort_values([DISTRICT_ID_COL, YEAR_COL]).reset_index(drop=True)

    # 1. Target incidence lags (t-1, t-2, t-3)
    if INCIDENCE_COL in working.columns:
        for lag in [1, 2, 3]:
            working[f"{INCIDENCE_COL}_lag_{lag}"] = working.groupby(DISTRICT_ID_COL)[
                INCIDENCE_COL
            ].shift(lag)
        
        # Historical target drift: (y_{t-1} - y_{t-2})
        working["pf_incidence_rate_drift"] = (
            working[f"{INCIDENCE_COL}_lag_1"] - working[f"{INCIDENCE_COL}_lag_2"]
        )

    # 2. Climate predictor lags and shifted rolling stats (strict t-1 base)
    feature_cols = _lag_feature_columns(working)
    for col in feature_cols:
        for lag in [1, 2]:
            working[f"{col}_lag_{lag}"] = working.groupby(DISTRICT_ID_COL)[col].shift(lag)
            
        # Shift rolling window by 1 year with exact min_periods to prevent partial windows
        for window in [2, 3]:
            working[f"{col}_roll_{window}y"] = (
                working.groupby(DISTRICT_ID_COL)[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=window).mean())
            )

    # 3. Hydro-thermal index derived from t-1 climate
    if (
        "precipitation_sum_mm_lag_1" in working.columns
        and "relative_humidity_mean_lag_1" in working.columns
        and "temp_mean_c_lag_1" in working.columns
    ):
        working["hydro_thermal_index"] = (
            working["precipitation_sum_mm_lag_1"] * working["relative_humidity_mean_lag_1"]
        ) / (working["temp_mean_c_lag_1"] + 1e-5)

    # Compute explicit target delta: Delta_t = y_t - y_{t-1}
    if INCIDENCE_COL in working.columns and f"{INCIDENCE_COL}_lag_1" in working.columns:
        working["target_delta"] = working[INCIDENCE_COL] - working[f"{INCIDENCE_COL}_lag_1"]

    # Drop rows lacking required historical lags
    required_cols = [INCIDENCE_COL]
    if f"{INCIDENCE_COL}_lag_1" in working.columns:
        required_cols.append(f"{INCIDENCE_COL}_lag_1")
    if f"{INCIDENCE_COL}_lag_2" in working.columns:
        required_cols.append(f"{INCIDENCE_COL}_lag_2")

    working = working.dropna(subset=required_cols).reset_index(drop=True)
    logger.info("Prepared annual feature panel with %s rows.", len(working))
    return working


def prepare_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compatibility wrapper for prepare_annual_features."""
    return prepare_annual_features(df)


def ablation_feature_sets(columns: Iterable[str]) -> dict[str, list[str]]:
    """Create feature subsets for ablation studies."""
    column_list = list(columns)
    
    # Climate-only features exclude all incidence-derived variables (lags and drift)
    climate_cols = [
        col for col in column_list 
        if not col.startswith(INCIDENCE_COL) and col != "pf_incidence_rate_drift"
    ]
    
    return {
        "all": column_list,
        "climate_only": climate_cols,
        "lag_only": [col for col in column_list if "lag_" in col],
        "rolling_only": [col for col in column_list if "roll_" in col],
    }


if __name__ == "__main__":
    prepare_annual_features().to_csv(
        DATA_PROCESSED / "model_ready_dataset.csv", index=False
    )