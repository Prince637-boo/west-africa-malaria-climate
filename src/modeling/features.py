"""Central definition of model feature groups (used by training and evaluation)."""

from __future__ import annotations

import pandas as pd

from src.config import CLIMATE_VARS, ID_COLUMNS, INCIDENCE_COL, TARGET_PREFIX

NON_FEATURE_PREFIXES = (
    TARGET_PREFIX,
    "target_date_",
    "target_month_",
    "seasonal_naive_",
)

NON_FEATURE_COLUMNS = set(ID_COLUMNS) | {
    INCIDENCE_COL,
    "target_is_simulated",
    "extract_latitude",
    "extract_longitude",
    "month",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric origin-time covariates, excluding identifiers and all lead targets."""
    cols = []
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if any(col.startswith(prefix) for prefix in NON_FEATURE_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        cols.append(col)
    return cols


def climate_feature_columns(columns: list[str]) -> list[str]:
    climate_prefixes = tuple(CLIMATE_VARS)
    return [c for c in columns if c.startswith(climate_prefixes)]


def season_feature_columns(columns: list[str]) -> list[str]:
    return [
        c
        for c in columns
        if c in {"month_sin", "month_cos", "known_target_month_sin", "known_target_month_cos"}
    ]


def autoregressive_feature_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith("incidence_lag_")]


def spatial_feature_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if c in {"latitude", "longitude"}]


def ablation_feature_sets(columns: list[str]) -> dict[str, list[str]]:
    """Named subsets for climate-only / season-only / AR-only / full comparisons."""
    climate = climate_feature_columns(columns)
    season = season_feature_columns(columns)
    ar = autoregressive_feature_columns(columns)
    spatial = spatial_feature_columns(columns)
    return {
        "season_only": season,
        "climate_only": climate + spatial,
        "autoregressive_only": ar + spatial,
        "full": columns,
    }
