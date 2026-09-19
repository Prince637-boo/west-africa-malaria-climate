from __future__ import annotations

import pandas as pd

from src.data.resample import resample_monthly
from src.features.engineering import build_forecast_frame
from src.modeling.evaluation import walk_forward_splits
from src.modeling.features import feature_columns
from tests.helpers import make_synthetic_panel


def test_resample_monthly_sums_precipitation():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-02-01"]),
            "precipitation_sum": [1.0, 3.0, 5.0],
            "temperature_2m_mean": [20.0, 22.0, 24.0],
            "temperature_2m_max": [25.0, 26.0, 28.0],
            "temperature_2m_min": [15.0, 16.0, 18.0],
        }
    )
    monthly = resample_monthly(df)
    january = monthly[monthly["time"].dt.month == 1].iloc[0]
    assert january["precipitation_sum"] == 4.0


def test_horizon_target_is_future_incidence_not_same_month():
    panel = make_synthetic_panel(n_districts=2, n_months=36)
    frame = build_forecast_frame(panel)
    row = frame[(frame["district_id"] == "D0") & (frame["date"] == "2019-06-01")].iloc[0]
    origin = panel[(panel["district_id"] == "D0") & (panel["date"] == "2019-06-01")].iloc[0]
    nxt = panel[(panel["district_id"] == "D0") & (panel["date"] == "2019-07-01")].iloc[0]
    assert row["precipitation_mm"] == origin["precipitation_mm"]
    assert row["target_h1"] == nxt["malaria_incidence"]
    assert row["malaria_incidence"] == origin["malaria_incidence"]


def test_features_exclude_lead_targets_and_same_month_is_origin_only():
    frame = build_forecast_frame(make_synthetic_panel())
    cols = feature_columns(frame)
    assert not any(c.startswith("target_h") for c in cols)
    assert "malaria_incidence" not in cols
    assert "precipitation_mm" in cols
    assert "incidence_lag_0m" in cols


def test_climate_lag_1_matches_previous_month():
    panel = make_synthetic_panel(n_districts=1, n_months=24, seed=1)
    frame = build_forecast_frame(panel)
    row = frame[frame["date"] == "2019-03-01"].iloc[0]
    prev = panel[panel["date"] == "2019-02-01"].iloc[0]
    assert row["precipitation_mm_lag_1m"] == prev["precipitation_mm"]


def test_walk_forward_has_no_future_target_dates_in_train():
    frame = build_forecast_frame(make_synthetic_panel(n_months=60, start="2016-01-01"))
    splits = list(walk_forward_splits(frame, horizon=1, test_years=(2019, 2020)))
    assert splits
    for _, train, test in splits:
        assert train["target_date_h1"].max() < test["target_date_h1"].min()


def test_merge_inner_join_on_district_and_date(tmp_path, monkeypatch):
    from src.data import merge_datasets

    malaria = make_synthetic_panel(n_districts=2, n_months=6)[
        ["district_id", "district_name", "region", "date", "malaria_incidence", "target_is_simulated"]
    ]
    climate = make_synthetic_panel(n_districts=2, n_months=6)[
        ["district_id", "district_name", "date", "precipitation_mm", "temp_mean_c", "temp_max_c", "temp_min_c"]
    ]
    malaria.loc[0, "date"] = pd.Timestamp("1999-01-01")
    processed = tmp_path / "processed"
    processed.mkdir()
    malaria.to_csv(processed / "togo_malaria_incidence_simulated.csv", index=False)
    climate.to_csv(processed / "togo_climate_monthly.csv", index=False)
    monkeypatch.setattr(merge_datasets, "DATA_PROCESSED", processed)
    merged = merge_datasets.merge_epidemiological_and_climate_data()
    assert merged["date"].min() >= pd.Timestamp("2018-01-01")
    assert set(["precipitation_mm", "malaria_incidence"]).issubset(merged.columns)
