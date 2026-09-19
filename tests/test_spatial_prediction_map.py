from pathlib import Path

from src.visualization.spatial_prediction_map import generate_spatial_prediction_figure


def test_generate_spatial_prediction_figure_creates_png(tmp_path):
    output_path = tmp_path / "figure_1_spatial_prediction_map.png"

    result = generate_spatial_prediction_figure(output_path=str(output_path))

    assert output_path.exists()
    assert result == str(output_path)
