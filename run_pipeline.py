"""End-to-end local pipeline after climate and GADM files exist."""

from __future__ import annotations

import pandas as pd

from src.config import DATA_PROCESSED
from src.data.merge_datasets import merge_epidemiological_and_climate_data
from src.data.simulate_incidence import simulate_malaria_incidence
from src.features.engineering import generate_climate_lags
from src.logging_utils import get_logger
from src.modeling.evaluation import evaluate_walk_forward
from src.modeling.interpretability import permutation_importance_table
from src.visualization.spatial_prediction_map import (
    generate_district_timeseries_figure,
    generate_spatial_prediction_figure,
)

logger = get_logger(__name__)


def main() -> None:
    climate_path = DATA_PROCESSED / "togo_climate_monthly.csv"
    if not climate_path.exists():
        logger.error(
            "Missing climate file. Run: python -m src.data.download_data && python -m src.data.climate_data"
        )
        raise SystemExit(1)
    climate = pd.read_csv(climate_path)
    simulate_malaria_incidence(climate=climate)
    merge_epidemiological_and_climate_data()
    generate_climate_lags()
    evaluate_walk_forward()
    permutation_importance_table()
    generate_spatial_prediction_figure()
    generate_district_timeseries_figure()


if __name__ == "__main__":
    main()
