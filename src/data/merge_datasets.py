"""Merge simulated incidence with monthly climate into a district-month panel."""

from __future__ import annotations

import pandas as pd

from src.config import DATA_PROCESSED, INCIDENCE_COL
from src.logging_utils import get_logger

logger = get_logger(__name__)


def merge_epidemiological_and_climate_data() -> pd.DataFrame:
    """Inner-join epidemiological and climate data on district and date."""
    path_malaria = DATA_PROCESSED / "togo_malaria_incidence_simulated.csv"
    path_climate = DATA_PROCESSED / "togo_climate_monthly.csv"

    if not path_malaria.exists():
        raise FileNotFoundError(
            f"Missing file: {path_malaria}. Run src/data/simulate_incidence.py first."
        )
    if not path_climate.exists():
        raise FileNotFoundError(
            f"Missing file: {path_climate}. Run src/data/climate_data.py first."
        )

    df_malaria = pd.read_csv(path_malaria)
    df_climate = pd.read_csv(path_climate)
    df_malaria["date"] = pd.to_datetime(df_malaria["date"])
    df_climate["date"] = pd.to_datetime(df_climate["date"])

    climate_drop = {"district_name"}
    climate_cols = [col for col in df_climate.columns if col not in climate_drop]
    n_before = len(df_malaria)
    df_merged = pd.merge(df_malaria, df_climate[climate_cols], on=["district_id", "date"], how="inner")
    df_merged = df_merged.sort_values(by=["district_id", "date"]).reset_index(drop=True)

    dropped = n_before - len(df_merged)
    if dropped > 0:
        logger.warning(
            "Inner merge dropped %s incidence rows without a matching climate month.",
            dropped,
        )

    null_counts = int(df_merged.isnull().sum().sum())
    if null_counts > 0:
        logger.warning("Detected %s missing values after merging; dropping incomplete rows.", null_counts)
        df_merged = df_merged.dropna().reset_index(drop=True)

    incidence_col = None
    for candidate in (INCIDENCE_COL, "malaria_incidence"):
        if candidate in df_merged.columns:
            incidence_col = candidate
            break
    if incidence_col is None:
        raise ValueError("Merged panel is missing the incidence column.")

    if incidence_col != "malaria_incidence" and "malaria_incidence" not in df_merged.columns:
        df_merged["malaria_incidence"] = df_merged[incidence_col]
    if incidence_col != INCIDENCE_COL and INCIDENCE_COL not in df_merged.columns:
        df_merged[INCIDENCE_COL] = df_merged["malaria_incidence"]

    output_path = DATA_PROCESSED / "togo_merged_monthly_panel.csv"
    df_merged.to_csv(output_path, index=False)
    logger.info("Merged panel saved: %s (%s rows, %s columns).", output_path, len(df_merged), df_merged.shape[1])
    return df_merged


if __name__ == "__main__":
    merge_epidemiological_and_climate_data()
