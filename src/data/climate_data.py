"""Module d'extraction et d'agrégation des données climatiques ERA5 via Open-Meteo API (Mode Batch optimisé)."""

from __future__ import annotations

import time
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

from src.config import (
    BOUNDARIES_DIR,
    CLIMATE_API_URL,
    COUNTRIES,
    DATA_PROCESSED,
    END_YEAR,
    START_YEAR,
)
from src.logging_utils import get_logger

logger = get_logger(__name__)

CLIMATE_OUTPUT_CSV = DATA_PROCESSED / "west_africa_climate_2000_2025.csv"
CACHE_DIR = DATA_PROCESSED / "climate_cache"

# Configuration du client Open-Meteo avec cache et réessais
cache_session = requests_cache.CachedSession('.openmeteo_cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=2)
openmeteo = openmeteo_requests.Client(session=retry_session)


def fetch_country_boundaries(iso3: str, url: str) -> gpd.GeoDataFrame:
    """Charge le GeoJSON GADM Admin-2 d'un pays depuis le cache local ou le télécharge."""
    BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
    local_path = BOUNDARIES_DIR / f"gadm41_{iso3}_2.json"

    if not local_path.exists():
        logger.info(f"Téléchargement des frontières GADM pour {iso3}...")
        import requests
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        local_path.write_bytes(response.content)

    gdf = gpd.read_file(local_path)
    gdf["iso3"] = iso3
    return gdf


def get_all_districts_centroids() -> gpd.GeoDataFrame:
    """Rassemble les districts des 4 pays et calcule leurs centroïdes (lat, lon)."""
    gdfs = []
    for iso3, url in COUNTRIES.items():
        try:
            gdf = fetch_country_boundaries(iso3, url)

            name_col = next(
                (col for col in ["NAME_2", "NAME_1", "GID_2"] if col in gdf.columns),
                gdf.columns[0],
            )
            region_col = "NAME_1" if "NAME_1" in gdf.columns else name_col

            gdf["district_name"] = gdf[name_col]
            gdf["region"] = gdf[region_col]
            gdf["district_id"] = (
                gdf["iso3"] + "_" + gdf["GID_2"]
                if "GID_2" in gdf.columns
                else gdf["iso3"] + "_" + gdf.index.astype(str)
            )

            centroids = gdf.geometry.to_crs(epsg=3857).centroid.to_crs(epsg=4326)
            gdf["latitude"] = centroids.y
            gdf["longitude"] = centroids.x

            gdfs.append(
                gdf[
                    [
                        "iso3",
                        "district_id",
                        "district_name",
                        "region",
                        "latitude",
                        "longitude",
                    ]
                ]
            )
        except Exception as e:
            logger.error(f"Erreur lors du traitement des limites pour {iso3}: {e}")

    combined_gdf = pd.concat(gdfs, ignore_index=True)
    logger.info(f"Total de {len(combined_gdf)} districts enregistrés.")
    return combined_gdf


def extract_variable_array(daily_obj, var_idx: int) -> np.ndarray:
    """Extrait le tableau numpy de la variable donnée de façon robuste selon le SDK."""
    var = daily_obj.Variables(var_idx)
    if hasattr(var, "ValuesAsNparray"):
        res = var.ValuesAsNparray()
        return res() if callable(res) else res
    elif hasattr(var, "Values"):
        return np.array([var.Values(i) for i in range(var.ValuesLength())])
    else:
        raise AttributeError(f"Impossible d'extraire les données pour la variable {var_idx}")


def aggregate_daily_response_to_annual(response) -> pd.DataFrame:
    """Transforme la réponse binaire Open-Meteo d'un district en métriques annuelles."""
    daily = response.Daily()
    
    # Génération des dates
    times = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )

    df_daily = pd.DataFrame({
        "time": times,
        "precipitation_sum": extract_variable_array(daily, 0),
        "temperature_2m_mean": extract_variable_array(daily, 1),
        "temperature_2m_max": extract_variable_array(daily, 2),
        "temperature_2m_min": extract_variable_array(daily, 3),
        "relative_humidity_2m_mean": extract_variable_array(daily, 4),
    })

    df_daily["year"] = df_daily["time"].dt.year
    df_daily["month"] = df_daily["time"].dt.month
    df_daily["is_wet_season"] = df_daily["month"].isin([5, 6, 7, 8, 9, 10])

    annual_records = []
    for year, group in df_daily.groupby("year"):
        precip_total = group["precipitation_sum"].sum()
        precip_wet_season = group[group["is_wet_season"]]["precipitation_sum"].sum()
        temp_mean = group["temperature_2m_mean"].mean()
        temp_max = group["temperature_2m_max"].mean()
        temp_min = group["temperature_2m_min"].mean()
        rh_mean = group["relative_humidity_2m_mean"].mean()

        annual_records.append({
            "year": int(year),
            "precipitation_sum_mm": round(float(precip_total), 2),
            "rainy_season_precip_mm": round(float(precip_wet_season), 2),
            "temp_mean_c": round(float(temp_mean), 2),
            "temp_max_c": round(float(temp_max), 2),
            "temp_min_c": round(float(temp_min), 2),
            "relative_humidity_mean": round(float(rh_mean), 2),
        })

    df_annual = pd.DataFrame(annual_records)
    mean_precip = df_annual["precipitation_sum_mm"].mean()
    df_annual["precip_anomaly_mm"] = round(df_annual["precipitation_sum_mm"] - mean_precip, 2)

    return df_annual


def fetch_climate_data(batch_size: int = 10) -> pd.DataFrame:
    """Fonction principale d'extraction en batch."""
    if CLIMATE_OUTPUT_CSV.exists():
        logger.info(f"Fichier climat déjà existant : {CLIMATE_OUTPUT_CSV}")
        return pd.read_csv(CLIMATE_OUTPUT_CSV)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    districts_df = get_all_districts_centroids()
    total_districts = len(districts_df)

    all_climate_records = []
    missing_districts = []

    # 1. Vérification du cache local
    for idx, row in districts_df.iterrows():
        cache_file = CACHE_DIR / f"{row['district_id']}.csv"
        if cache_file.exists():
            df_annual = pd.read_csv(cache_file)
            for key in ["iso3", "district_id", "district_name", "region"]:
                df_annual[key] = row[key]
            all_climate_records.append(df_annual)
        else:
            missing_districts.append(row)

    logger.info(f"Districts déjà en cache : {len(all_climate_records)}/{total_districts}")
    logger.info(f"Districts à télécharger : {len(missing_districts)}/{total_districts}")

    if missing_districts:
        missing_df = pd.DataFrame(missing_districts)
        total_batches = (len(missing_df) - 1) // batch_size + 1

        batch_idx = 0
        while batch_idx < total_batches:
            batch = missing_df.iloc[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            lats = batch["latitude"].tolist()
            lons = batch["longitude"].tolist()

            logger.info(
                f"Téléchargement du lot {batch_idx + 1}/{total_batches} "
                f"({len(batch)} districts) : {batch['district_name'].tolist()}..."
            )

            params = {
                "latitude": lats,
                "longitude": lons,
                "start_date": f"{START_YEAR}-01-01",
                "end_date": f"{END_YEAR}-12-31",
                "daily": [
                    "precipitation_sum",
                    "temperature_2m_mean",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "relative_humidity_2m_mean",
                ],
                "timezone": "Africa/Lome",
            }

            try:
                responses = openmeteo.weather_api(CLIMATE_API_URL, params=params)
                
                for idx_in_batch, response in enumerate(responses):
                    district_row = batch.iloc[idx_in_batch]
                    df_annual = aggregate_daily_response_to_annual(response)

                    # Sauvegarde cache individuel
                    cache_file = CACHE_DIR / f"{district_row['district_id']}.csv"
                    df_annual.to_csv(cache_file, index=False)

                    for key in ["iso3", "district_id", "district_name", "region"]:
                        df_annual[key] = district_row[key]

                    all_climate_records.append(df_annual)

                batch_idx += 1
                time.sleep(3)  # Pause raisonnable entre deux lots

            except Exception as e:
                err_msg = str(e)
                logger.error(f"Erreur sur le lot {batch_idx + 1} : {err_msg}")

                if "Minutely API request limit exceeded" in err_msg:
                    logger.warning("Quota par minute atteint. Pause automatique de 65 secondes...")
                    time.sleep(65)
                elif "Hourly API request limit exceeded" in err_msg:
                    logger.warning("Quota horaire atteint. Si vous utilisez un VPN, changez de serveur, sinon attendez.")
                    time.sleep(60)
                else:
                    time.sleep(10)

    if not all_climate_records:
        raise RuntimeError("Aucune donnée climatique n'a pu être récupérée.")

    final_climate_df = pd.concat(all_climate_records, ignore_index=True)
    cols_order = [
        "iso3",
        "district_id",
        "district_name",
        "region",
        "year",
        "precipitation_sum_mm",
        "rainy_season_precip_mm",
        "precip_anomaly_mm",
        "temp_mean_c",
        "temp_max_c",
        "temp_min_c",
        "relative_humidity_mean",
    ]
    final_climate_df = final_climate_df[cols_order]
    final_climate_df.to_csv(CLIMATE_OUTPUT_CSV, index=False)

    logger.info(
        f"Données climatiques enregistrées avec succès : {CLIMATE_OUTPUT_CSV} "
        f"({len(final_climate_df)} lignes)"
    )
    return final_climate_df


if __name__ == "__main__":
    fetch_climate_data()