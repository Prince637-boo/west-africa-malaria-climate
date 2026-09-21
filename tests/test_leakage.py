"""Extended data leakage tests for feature engineering and temporal splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DISTRICT_ID_COL, INCIDENCE_COL, YEAR_COL
from src.modeling.features import feature_columns, prepare_annual_features


def test_no_future_climate_or_incidence_leakage() -> None:
    """Ensure features at year t only depend on historical values up to year t-1."""
    # Synthetic multi-year district panel
    data = []
    districts = ["DIST_01", "DIST_02"]
    years = list(range(2000, 2010))

    for d in districts:
        for y in years:
            data.append(
                {
                    DISTRICT_ID_COL: d,
                    YEAR_COL: y,
                    INCIDENCE_COL: float(y * 10),
                    "precipitation_sum_mm": float(y * 100),
                    "temp_mean_c": float(y + 5),
                }
            )

    df_raw = pd.DataFrame(data)
    df_feat = prepare_annual_features(df_raw)

    # 1. Check feature columns selection strictly excludes target & current-year climate
    cols = feature_columns(df_feat)
    assert INCIDENCE_COL not in cols
    assert "precipitation_sum_mm" not in cols
    assert "temp_mean_c" not in cols

    # 2. Check value alignment for year 2005 (must strictly match values from 2004 or earlier)
    row_2005 = df_feat[(df_feat[DISTRICT_ID_COL] == "DIST_01") & (df_feat[YEAR_COL] == 2005)].iloc[0]

    # Incidence lag 1 in 2005 must be incidence of 2004 (2004 * 10 = 20040)
    assert row_2005[f"{INCIDENCE_COL}_lag_1"] == 20040.0
    # Climate lag 1 in 2005 must be precipitation of 2004 (2004 * 100 = 200400)
    assert row_2005["precipitation_sum_mm_lag_1"] == 200400.0


def test_expanding_window_split_integrity() -> None:
    """Ensure no overlap between training set and test year in walk-forward evaluation."""
    data = []
    for y in range(2000, 2010):
        data.append({DISTRICT_ID_COL: "D1", YEAR_COL: y, INCIDENCE_COL: 100.0, "precipitation_sum_mm": 500.0})

    df = prepare_annual_features(pd.DataFrame(data))
    
    test_year = 2008
    train_df = df[df[YEAR_COL] < test_year]
    test_df = df[df[YEAR_COL] == test_year]

    assert train_df[YEAR_COL].max() < test_year
    assert test_year not in train_df[YEAR_COL].values
    assert len(test_df) == 1