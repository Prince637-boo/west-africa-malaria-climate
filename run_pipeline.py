"""Production entry point for the annual malaria and climate forecasting pipeline."""

from __future__ import annotations

from src.config import DATA_PROCESSED, ensure_data_dirs
from src.logging_utils import get_logger
from src.modeling.training import run_pipeline

logger = get_logger(__name__)


def main() -> None:
    """Execute the production pipeline end-to-end without swallowing errors."""
    ensure_data_dirs()

    incidence_path = DATA_PROCESSED / "west_africa_malaria_incidence_2000_2025.csv"
    if not incidence_path.exists():
        raise FileNotFoundError(
            f"The malaria incidence dataset is missing: {incidence_path}. "
            "Run the data preparation workflow before modeling."
        )

    climate_path = DATA_PROCESSED / "west_africa_climate_2000_2025.csv"
    if not climate_path.exists():
        raise FileNotFoundError(
            f"The climate dataset is missing: {climate_path}. "
            "Run the climate extraction workflow before modeling."
        )

    logger.info("Starting the annual forecasting pipeline.")
    results = run_pipeline(split_year=2020, gap_years=1)
    logger.info("Pipeline completed successfully with %s model evaluations.", len(results))


if __name__ == "__main__":
    main()