from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_panel(
    n_districts: int = 3,
    n_months: int = 48,
    start: str = "2018-01-01",
    seed: int = 0,
) -> pd.DataFrame:
    """District-month panel with known climate and a lagged precipitation signal."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_months, freq="MS")
    rows = []
    for d in range(n_districts):
        lat = 6.2 + d * 0.8
        precip = rng.uniform(20, 180, size=n_months)
        temp = 26 + 2 * np.sin(np.arange(n_months) / 12 * 2 * np.pi) + 0.1 * d
        precip_lag = np.concatenate([[np.nan], precip[:-1]])
        for i, date in enumerate(dates):
            month = date.month
            seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * (month - 5) / 12)
            base = 12 + (lat - 6) * 3
            climate_term = 0.02 * (precip_lag[i] if np.isfinite(precip_lag[i]) else 80)
            y = max(0.5, base * seasonal + climate_term + rng.normal(0, 0.8))
            rows.append(
                {
                    "district_id": f"D{d}",
                    "district_name": f"District {d}",
                    "region": "Test",
                    "latitude": lat,
                    "longitude": 1.2 + 0.1 * d,
                    "date": date,
                    "precipitation_mm": float(precip[i]),
                    "temp_mean_c": float(temp[i]),
                    "temp_max_c": float(temp[i] + 4),
                    "temp_min_c": float(temp[i] - 3),
                    "malaria_incidence": float(y),
                    "target_is_simulated": True,
                }
            )
    return pd.DataFrame(rows)
