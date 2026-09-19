"""Download GADM Admin-2 boundaries for Togo (prefectures, not health districts)."""

from __future__ import annotations

from src.config import DATA_RAW, GADM_URL, ensure_data_dirs
from src.data.provenance import write_data_manifest
from src.logging_utils import get_logger

logger = get_logger(__name__)


def download_togo_boundaries():
    """Download GADM 4.1 Admin-2 GeoJSON if it is not already cached."""
    import geopandas as gpd
    import requests

    ensure_data_dirs()
    geojson_path = DATA_RAW / "togo_districts.geojson"
    if not geojson_path.exists():
        logger.info("Downloading GADM 4.1 Admin-2 polygons from %s", GADM_URL)
        response = requests.get(GADM_URL, timeout=60)
        response.raise_for_status()
        geojson_path.write_bytes(response.content)
    gdf = gpd.read_file(geojson_path)
    logger.info(
        "Loaded %s GADM Admin-2 polygons. These are prefectures, not official health districts.",
        len(gdf),
    )
    write_data_manifest()
    return gdf


def fetch_real_map_incidence(*_args, **_kwargs):
    """Removed: this project never downloaded MAP monthly district incidence."""
    raise RuntimeError(
        "fetch_real_map_incidence has been removed. Monthly district incidence is "
        "simulated (see src.data.simulate_incidence). MAP rasters are annual and "
        "are not used as a prediction target."
    )


if __name__ == "__main__":
    download_togo_boundaries()
