from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

geopandas = pytest.importorskip("geopandas")
from shapely.geometry import box

from src.visualization.spatial_prediction_map import generate_spatial_prediction_figure


def test_generate_spatial_prediction_figure_creates_png(tmp_path):
    geo = geopandas.GeoDataFrame(
        {
            "GID_2": ["A", "B"],
            "NAME_2": ["Alpha", "Beta"],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        },
        crs="EPSG:4326",
    )
    geo_path = tmp_path / "tiny.geojson"
    geo.to_file(geo_path, driver="GeoJSON")
    comparison = pd.DataFrame(
        {
            "district_id": ["A", "B"],
            "observed_incidence": [10.0, 12.0],
            "predicted_incidence": [9.5, 12.5],
            "mean_absolute_error": [0.5, 0.5],
        }
    )
    output_path = tmp_path / "spatial_prediction_and_error_map.png"
    result = generate_spatial_prediction_figure(
        output_path=str(output_path),
        comparison=comparison,
        geo_path=geo_path,
    )
    assert Path(result).exists()
    assert output_path.exists()
