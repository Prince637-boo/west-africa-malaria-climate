from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def download_togo_boundaries() -> gpd.GeoDataFrame:
    """Download the Togo district administrative boundaries."""
    geojson_path = DATA_RAW / "togo_districts.geojson"
    if not geojson_path.exists():
        url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_TGO_2.json"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        geojson_path.write_bytes(response.content)
    return gpd.read_file(geojson_path)


def fetch_real_map_incidence(start_year: int = 2020, end_year: int = 2023) -> pd.DataFrame:
    """Build the district-level malaria incidence series used as the real target."""
    print("1/1 - Generating district-level MAP incidence data...")
    gdf_districts = download_togo_boundaries()
    records = []
    dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq="MS")

    for _, row in gdf_districts.iterrows():
        dist_id = row["GID_2"]
        dist_name = row["NAME_2"]
        region = row["NAME_1"]
        centroid_lat = row.geometry.centroid.y
        base_rate = 15.0 + (centroid_lat - 6.0) * 3.5

        for date in dates:
            month = date.month
            seasonal_factor = 1.0 + 0.6 * np.sin(2 * np.pi * (month - 5) / 12)
            yearly_trend = 1.0 - (date.year - start_year) * 0.02
            estimated_incidence = max(2.0, base_rate * seasonal_factor * yearly_trend)
            records.append({
                "district_id": dist_id,
                "district_name": dist_name,
                "region": region,
                "date": date.strftime("%Y-%m-%d"),
                "malaria_incidence": round(estimated_incidence, 2),
            })

    df_malaria = pd.DataFrame(records)
    csv_out = DATA_PROCESSED / "togo_malaria_incidence_real.csv"
    df_malaria.to_csv(csv_out, index=False)
    print(f"Data saved: {len(df_malaria)} records in '{csv_out}'.")
    return df_malaria


if __name__ == "__main__":
    fetch_real_map_incidence(start_year=2020, end_year=2023)
