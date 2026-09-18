import os
import time
import requests
import geopandas as gpd
import pandas as pd

DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)


def _request_with_backoff(url, params=None, timeout=30, max_retries=5, base_delay=1.0, max_delay=30.0):
    """Perform an HTTP request with exponential backoff and retry logic."""
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
            print(
                f"Request failed on attempt {attempt}/{max_retries}. "
                f"Retrying in {wait_time:.1f} seconds..."
            )
            time.sleep(wait_time)

    if last_error is not None:
        raise last_error

    raise RuntimeError("Request failed without returning a usable error.")


def fetch_openmeteo_climate(
    start_date="2020-01-01",
    end_date="2023-12-31",
    request_delay_seconds=1.0,
    checkpoint_every_districts=5,
):
    """
    Fetch historical precipitation and temperature data for each district centroid in Togo
    using the Open-Meteo Archive API, with retry handling and incremental saving.
    """
    print(
        f"1/1 - Extracting ERA5 climate data via Open-Meteo "
        f"({start_date} to {end_date})..."
    )

    geojson_path = os.path.join(DATA_RAW, "togo_districts.geojson")
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(
            "Run src/fetch_malaria_data.py first to download the district boundaries."
        )

    gdf_districts = gpd.read_file(geojson_path)

    # Reproject district polygons to UTM Zone 31N to compute centroid coordinates precisely.
    gdf_projected = gdf_districts.to_crs(epsg=32631)
    centroids_projected = gdf_projected.geometry.centroid
    centroids_latlon = centroids_projected.to_crs(epsg=4326)

    csv_out = os.path.join(DATA_PROCESSED, "togo_climate_monthly.csv")
    seen_keys = set()
    if os.path.exists(csv_out):
        existing = pd.read_csv(csv_out)
        if not existing.empty:
            seen_keys = set(
                zip(existing["district_id"].astype(str), existing["date"].astype(str))
            )

    records = []
    base_url = "https://archive-api.open-meteo.com/v1/archive"

    for idx, row in gdf_districts.iterrows():
        dist_id = row["GID_2"]
        dist_name = row["NAME_2"]

        lat = centroids_latlon.iloc[idx].y
        lon = centroids_latlon.iloc[idx].x

        print(
            f"  --> Processing district: {dist_name} "
            f"({idx + 1}/{len(gdf_districts)})..."
        )

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
            "timezone": "Africa/Lome",
        }

        try:
            response = _request_with_backoff(
                base_url,
                params=params,
                timeout=30,
                max_retries=5,
                base_delay=1.0,
                max_delay=30.0,
            )
            data = response.json()

            daily_data = data.get("daily", {})
            if not daily_data or "time" not in daily_data:
                print(f"Warning: no climate data returned for {dist_name}.")
                continue

            df_daily = pd.DataFrame(daily_data)
            df_daily["time"] = pd.to_datetime(df_daily["time"])

            df_monthly = (
                df_daily.resample("ME", on="time")
                .agg(
                    {
                        "precipitation_sum": "sum",
                        "temperature_2m_mean": "mean",
                        "temperature_2m_max": "mean",
                        "temperature_2m_min": "mean",
                    }
                )
                .reset_index()
            )

            for _, m_row in df_monthly.iterrows():
                record = {
                    "district_id": dist_id,
                    "district_name": dist_name,
                    "date": m_row["time"].strftime("%Y-%m-01"),
                    "precipitation_mm": round(float(m_row["precipitation_sum"]), 2),
                    "temp_mean_c": round(float(m_row["temperature_2m_mean"]), 2),
                    "temp_max_c": round(float(m_row["temperature_2m_max"]), 2),
                    "temp_min_c": round(float(m_row["temperature_2m_min"]), 2),
                }
                key = (str(record["district_id"]), record["date"])
                if key in seen_keys:
                    continue
                records.append(record)
                seen_keys.add(key)

            if len(records) >= 250 or (idx + 1) % checkpoint_every_districts == 0:
                df_batch = pd.DataFrame(records)
                if not df_batch.empty:
                    file_exists = os.path.exists(csv_out)
                    df_batch.to_csv(
                        csv_out,
                        mode="a" if file_exists else "w",
                        header=not file_exists,
                        index=False,
                    )
                    records = []

            time.sleep(request_delay_seconds)

        except Exception as exc:
            print(f"Error while processing {dist_name}: {exc}")
            continue

    if records:
        df_batch = pd.DataFrame(records)
        file_exists = os.path.exists(csv_out)
        df_batch.to_csv(
            csv_out,
            mode="a" if file_exists else "w",
            header=not file_exists,
            index=False,
        )

    final_df = pd.read_csv(csv_out) if os.path.exists(csv_out) else pd.DataFrame()
    print(
        f"\nClimate extraction complete: {len(final_df)} rows saved in '{csv_out}'."
    )
    return final_df


if __name__ == "__main__":
    df_climate = fetch_openmeteo_climate(start_date="2020-01-01", end_date="2023-12-31")
    print("Climate data extraction finished.")