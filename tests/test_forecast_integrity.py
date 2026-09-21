"""Unit tests for forecast integrity and feature panel preparation."""

from __future__ import annotations

import pandas as pd
from src.modeling.features import prepare_annual_features, feature_columns


def test_annual_feature_target_isolation():
    """Verify target and metadata columns are strictly excluded from features."""
    synthetic_df = pd.DataFrame({
        "district_id": ["D1", "D1", "D1"],
        "year": [2020, 2021, 2022],
        "pf_incidence_rate": [0.15, 0.14, 0.13],
        "temp_mean": [26.5, 27.0, 26.8],
        "precip_sum": [1200, 1150, 1300]
    })
    
    panel = prepare_annual_features(synthetic_df)
    cols = feature_columns(panel)
    
    assert "pf_incidence_rate" not in cols
    assert "district_id" not in cols
    assert "year" not in cols
    assert "pf_incidence_rate_lag_1" in cols