"""Download monthly climate series from the Open-Meteo ERA5 archive API."""

from __future__ import annotations

import time

import geopandas as gpd
import pandas as pd
import requests

from src.config import (
    CLIMATE_API_URL,
    CLIMATE_END,
    CLIMATE_START,
    CLIMATE_TIMEZONE,
    DATA_PROCESSED,
    DATA_RAW,
    ensure_data_dirs,
)
from src.data.provenance import write_data_manifest
from src.data.resample import resample_monthly
from src.logging_utils import get_logger

logger = get_logger(__name__)


def _request_with_backoff(
    url: str,
    params=None,
    timeout: int = 30,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_retries:
                raise
            wait_time = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                "Request failed on attempt %s/%s. Retrying in %.1f seconds...",
                attempt,
                max_retries,
                wait_time,
            )
            time.sleep(wait_time)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Request failed without returning a usable response.")


def fetch_openmeteo_climate(
    start_date: str = CLIMATE_START,
    end_date: str = CLIMATE_END,
    request_delay_seconds: float = 1.0,
    checkpoint_every_districts: int = 5,
) -> pd.DataFrame:
    """Download precipitation and 2 m temperature at one in-polygon point per unit."""
    ensure_data_dirs()
    logger.info(
        "Downloading Open-Meteo ERA5 archive climate from %s to %s (point extraction, not zonal mean).",
        start_date,
        end_date,
    )
    geojson_path = DATA_RAW / "togo_districts.geojson"
    if not geojson_path.exists():
        raise FileNotFoundError("Run src/data/download_data.py first to download GADM polygons.")

    gdf_districts = gpd.read_file(geojson_path).reset_index(drop=True)
    gdf_projected = gdf_districts.to_crs(epsg=32631)
    points_latlon = gdf_projected.geometry.representative_point().to_crs(epsg=4326)

    csv_out = DATA_PROCESSED / "togo_climate_monthly.csv"
    seen_keys = set()
    if csv_out.exists():
        existing = pd.read_csv(csv_out)
        if not existing.empty:
            seen_keys = set(zip(existing["district_id"].astype(str), existing["date"].astype(str)))

    records: list[dict] = []

    for pos, row in gdf_districts.iterrows():
        dist_id = row["GID_2"]
        dist_name = row["NAME_2"]
        lat = float(points_latlon.iloc[pos].y)
        lon = float(points_latlon.iloc[pos].x)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "precipitation_sum",
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
            ],
            "timezone": CLIMATE_TIMEZONE,
        }

        try:
            response = _request_with_backoff(
                CLIMATE_API_URL, params=params, timeout=30, max_retries=5, base_delay=1.0, max_delay=30.0
            )
            data = response.json()
            daily_data = data.get("daily", {})
            if not daily_data or "time" not in daily_data:
                logger.warning("No climate data returned for %s.", dist_name)
                continue

            df_daily = pd.DataFrame(daily_data)
            df_daily["time"] = pd.to_datetime(df_daily["time"])
            df_monthly = resample_monthly(df_daily)

            for _, month_row in df_monthly.iterrows():
                record = {
                    "district_id": dist_id,
                    "district_name": dist_name,
                    "date": pd.Timestamp(month_row["time"]).to_period("M").to_timestamp().strftime("%Y-%m-01"),
                    "precipitation_mm": round(float(month_row["precipitation_sum"]), 2),
                    "temp_mean_c": round(float(month_row["temperature_2m_mean"]), 2),
                    "temp_max_c": round(float(month_row["temperature_2m_max"]), 2),
                    "temp_min_c": round(float(month_row["temperature_2m_min"]), 2),
                    "extract_latitude": round(lat, 5),
                    "extract_longitude": round(lon, 5),
                }
                key = (str(record["district_id"]), record["date"])
                if key in seen_keys:
                    continue
                records.append(record)
                seen_keys.add(key)

            if len(records) >= 250 or (pos + 1) % checkpoint_every_districts == 0:
                df_batch = pd.DataFrame(records)
                if not df_batch.empty:
                    file_exists = csv_out.exists()
                    df_batch.to_csv(csv_out, mode="a" if file_exists else "w", header=not file_exists, index=False)
                    records = []

            time.sleep(request_delay_seconds)

        except Exception as exc:  # pragma: no cover - network failure path
            logger.error("Error while processing %s: %s", dist_name, exc)
            continue

    if records:
        df_batch = pd.DataFrame(records)
        file_exists = csv_out.exists()
        df_batch.to_csv(csv_out, mode="a" if file_exists else "w", header=not file_exists, index=False)

    final_df = pd.read_csv(csv_out) if csv_out.exists() else pd.DataFrame()
    write_data_manifest()
    logger.info("Climate extraction complete: %s rows in %s.", len(final_df), csv_out)
    return final_df


if __name__ == "__main__":
    fetch_openmeteo_climate()
