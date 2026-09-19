"""Project-wide configuration: paths, seeds, horizons, and modeling defaults."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = PROJECT_ROOT / "notebooks" / "figures"

RANDOM_SEED = 42

# GADM 4.1 Admin-2 units are prefectures, not Togo Ministry of Health districts.
GADM_VERSION = "4.1"
GADM_LAYER = "TGO_2"
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_TGO_2.json"

CLIMATE_PROVIDER = "Open-Meteo Archive API"
CLIMATE_REANALYSIS = "ERA5 (served by Open-Meteo; not a direct Copernicus CDS / cdsapi download)"
CLIMATE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
CLIMATE_TIMEZONE = "Africa/Lome"
CLIMATE_START = "2015-01-01"
CLIMATE_END = "2023-12-31"

FORECAST_HORIZONS = (1, 2, 3)
CLIMATE_LAGS = (1, 2, 3)
INCIDENCE_LAGS = (0, 1, 2, 3, 12)
ROLLING_WINDOW_MONTHS = 3

ID_COLUMNS = ("district_id", "district_name", "region", "date")
TARGET_PREFIX = "target_h"
INCIDENCE_COL = "malaria_incidence"

CLIMATE_VARS = (
    "precipitation_mm",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
)

N_BOOTSTRAP = 400
WALK_FORWARD_TEST_MONTHS = 12


def ensure_data_dirs() -> None:
    """Create raw, processed, report, and figure directories if they are missing."""
    for path in (DATA_RAW, DATA_PROCESSED, REPORTS_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)
