"""Calendar aggregation helpers with pandas 2.1 / 2.2 frequency names."""

from __future__ import annotations

import pandas as pd


def resample_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily Open-Meteo series to calendar months."""
    try:
        aggregated = df_daily.resample("ME", on="time")
    except ValueError:
        aggregated = df_daily.resample("M", on="time")
    return aggregated.agg(
        {
            "precipitation_sum": "sum",
            "temperature_2m_mean": "mean",
            "temperature_2m_max": "mean",
            "temperature_2m_min": "mean",
        }
    ).reset_index()
