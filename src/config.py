"""Project-wide configuration for the annual malaria and climate forecasting pipeline."""

from __future__ import annotations

from pathlib import Path

# Project directory structure
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
RASTERS_DIR = DATA_RAW / "map_rasters"
BOUNDARIES_DIR = DATA_RAW / "boundaries"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = PROJECT_ROOT / "figures"

RANDOM_SEED = 42

# Country spatial boundaries sources
COUNTRIES = {
    "TGO": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_TGO_2.json",
    "BEN": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BEN_2.json",
    "GHA": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_GHA_2.json",
    "BFA": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BFA_2.json",
}

# Temporal bounds
START_YEAR = 2000
END_YEAR = 2025

# Climate provider settings
CLIMATE_PROVIDER = "Open-Meteo Archive API"
CLIMATE_REANALYSIS = "ERA5"
CLIMATE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
CLIMATE_TIMEZONE = "Africa/Lome"

# Dataset column naming conventions
YEAR_COL = "year"
DISTRICT_COL = "district_id"
DISTRICT_ID_COL = DISTRICT_COL
ISO3_COL = "iso3"
REGION_COL = "region"
DISTRICT_NAME_COL = "district_name"
ID_COLUMNS = (ISO3_COL, DISTRICT_COL, DISTRICT_NAME_COL, REGION_COL, YEAR_COL)

INCIDENCE_COL = "pf_incidence_rate"
MODEL_TARGET_COL = INCIDENCE_COL
TARGET_PREFIX = "target_h"
TARGET_PROXIES = {"pf_incidence_min", "pf_incidence_max"}

CLIMATE_VARS = (
    "precipitation_sum_mm",
    "rainy_season_precip_mm",
    "precip_anomaly_mm",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "relative_humidity_mean",
)

WALK_FORWARD_TEST_YEARS = 8


def ensure_data_dirs() -> None:
    """Create required project directories if they do not exist."""
    for path in (
        DATA_RAW,
        DATA_PROCESSED,
        RASTERS_DIR,
        BOUNDARIES_DIR,
        REPORTS_DIR,
        FIGURE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)