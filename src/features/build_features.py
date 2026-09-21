"""Load and merge annual climate and incidence data by district_id and year."""

from __future__ import annotations

import pandas as pd

from src.config import DATA_PROCESSED
from src.logging_utils import get_logger

logger = get_logger(__name__)

MALARIA_CSV_NAME = "west_africa_malaria_incidence_2000_2025.csv"
CLIMATE_CSV_NAME = "west_africa_climate_2000_2025.csv"
FINAL_OUTPUT_CSV = DATA_PROCESSED / "final_malaria_climate_dataset.csv"


def build_features() -> pd.DataFrame:
    """Load and merge the district-level annual records without silent name-based fallback."""
    malaria_path = DATA_PROCESSED / MALARIA_CSV_NAME
    climate_path = DATA_PROCESSED / CLIMATE_CSV_NAME

    if not malaria_path.exists() or not climate_path.exists():
        raise FileNotFoundError("One or more source CSV files are missing.")

    logger.info("Loading annual malaria and climate data...")
    df_malaria = pd.read_csv(malaria_path)
    df_climate = pd.read_csv(climate_path)

    df_malaria["year"] = pd.to_numeric(df_malaria["year"], errors="raise").astype(int)
    df_climate["year"] = pd.to_numeric(df_climate["year"], errors="raise").astype(int)

    for df in [df_malaria, df_climate]:
        for col in ["iso3", "district_id", "district_name", "region"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

    df_climate["district_id"] = df_climate["district_id"].str.replace(r"^[A-Z]{3}_", "", regex=True)
    df_malaria["district_id"] = df_malaria["district_id"].str.replace(r"^[A-Z]{3}_", "", regex=True)

    # Clean redundant metadata columns in climate before merge to avoid _malaria / _climate duplicates
    overlap_cols = [c for c in ["iso3", "district_name", "region"] if c in df_climate.columns and c in df_malaria.columns]
    df_climate_clean = df_climate.drop(columns=overlap_cols)

    logger.info("Merging annual records on ['district_id', 'year']...")
    df_merged = pd.merge(
        df_malaria,
        df_climate_clean,
        on=["district_id", "year"],
        how="inner",
        validate="one_to_one",
    )

    if df_merged.empty:
        raise ValueError("No exact district_id/year matches were found between malaria and climate datasets.")

    logger.info("Merged dataset contains %s rows.", len(df_merged))

    FINAL_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(FINAL_OUTPUT_CSV, index=False)
    logger.info("Final dataset saved to %s", FINAL_OUTPUT_CSV)
    return df_merged


# Alias for compatibility with execution modules
main = build_features


if __name__ == "__main__":
    build_features()