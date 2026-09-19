"""Simulate monthly malaria incidence from a documented data-generating process.

This is a methods dataset. It is not MAP incidence, not DHIS2, and must not be
presented as observed malaria in Togo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CLIMATE_END,
    CLIMATE_START,
    DATA_PROCESSED,
    INCIDENCE_COL,
    RANDOM_SEED,
    ensure_data_dirs,
)
from src.data.download_data import download_togo_boundaries
from src.data.provenance import write_data_manifest
from src.logging_utils import get_logger

logger = get_logger(__name__)

# Documented DGP coefficients (methods paper / software note).
DGP = {
    "spatial_base_at_lat_6": 15.0,
    "spatial_slope_per_degree_north": 3.5,
    "seasonal_amplitude": 0.35,
    "seasonal_phase_shift_months": 5,
    "precip_lag1_coef_per_mm": 0.012,
    "temp_mean_centered_coef": 0.15,
    "relative_trend_per_year": -0.015,
    "noise_sigma": 2.0,
    "floor": 0.5,
}


def _representative_latlon(gdf):
    """Return an in-polygon point in WGS84 for each feature."""
    projected = gdf.to_crs(epsg=32631)
    points = projected.geometry.representative_point().to_crs(epsg=4326)
    return points.x.to_numpy(), points.y.to_numpy()


def simulate_malaria_incidence(
    climate: pd.DataFrame | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Build a noisy panel whose mean structure is known by construction.

    If monthly climate is provided, lagged precipitation and temperature enter
    the mean. Otherwise only latitude, season, trend, and noise are used.
    """
    ensure_data_dirs()
    gdf = download_togo_boundaries().reset_index(drop=True)
    lon, lat = _representative_latlon(gdf)
    rng = np.random.default_rng(seed)

    start = pd.Timestamp(f"{CLIMATE_START[:7]}-01")
    end = pd.Timestamp(f"{CLIMATE_END[:7]}-01")
    dates = pd.date_range(start=start, end=end, freq="MS")

    climate_lookup = None
    if climate is not None and not climate.empty:
        climate_frame = climate.copy()
        climate_frame["date"] = pd.to_datetime(climate_frame["date"])
        climate_frame = climate_frame.sort_values(["district_id", "date"])
        climate_frame["precipitation_mm_lag_1m"] = climate_frame.groupby("district_id")[
            "precipitation_mm"
        ].shift(1)
        climate_lookup = climate_frame.drop_duplicates(["district_id", "date"]).set_index(
            ["district_id", "date"]
        )

    records = []
    for pos, row in gdf.iterrows():
        dist_id = row["GID_2"]
        dist_name = row["NAME_2"]
        region = row["NAME_1"]
        centroid_lat = float(lat[pos])
        centroid_lon = float(lon[pos])
        base_rate = DGP["spatial_base_at_lat_6"] + (centroid_lat - 6.0) * DGP[
            "spatial_slope_per_degree_north"
        ]

        for date in dates:
            month = int(date.month)
            seasonal_factor = 1.0 + DGP["seasonal_amplitude"] * np.sin(
                2 * np.pi * (month - DGP["seasonal_phase_shift_months"]) / 12
            )
            yearly_trend = 1.0 + (date.year - start.year) * DGP["relative_trend_per_year"]
            climate_effect = 0.0
            if climate_lookup is not None:
                key = (dist_id, date)
                if key in climate_lookup.index:
                    rec = climate_lookup.loc[key]
                    precip_lag = rec["precipitation_mm_lag_1m"]
                    temp_mean = rec["temp_mean_c"]
                    if pd.notna(precip_lag):
                        climate_effect += DGP["precip_lag1_coef_per_mm"] * float(precip_lag)
                    if pd.notna(temp_mean):
                        climate_effect += DGP["temp_mean_centered_coef"] * (float(temp_mean) - 27.0)

            noise = float(rng.normal(0.0, DGP["noise_sigma"]))
            estimated = max(
                DGP["floor"],
                base_rate * seasonal_factor * yearly_trend + climate_effect + noise,
            )
            records.append(
                {
                    "district_id": dist_id,
                    "district_name": dist_name,
                    "region": region,
                    "latitude": round(centroid_lat, 5),
                    "longitude": round(centroid_lon, 5),
                    "date": date.strftime("%Y-%m-%d"),
                    INCIDENCE_COL: round(float(estimated), 3),
                    "target_is_simulated": True,
                }
            )

    df_malaria = pd.DataFrame(records)
    csv_out = DATA_PROCESSED / "togo_malaria_incidence_simulated.csv"
    df_malaria.to_csv(csv_out, index=False)
    write_data_manifest({"incidence_dgp": DGP, "n_simulated_rows": int(len(df_malaria))})
    logger.info(
        "Wrote %s simulated incidence rows to %s. These are not observational MAP/DHIS2 data.",
        len(df_malaria),
        csv_out,
    )
    return df_malaria


if __name__ == "__main__":
    climate_path = DATA_PROCESSED / "togo_climate_monthly.csv"
    climate_df = pd.read_csv(climate_path) if climate_path.exists() else None
    if climate_df is None:
        logger.warning("No climate file found; simulating incidence without a climate term.")
    simulate_malaria_incidence(climate=climate_df)
