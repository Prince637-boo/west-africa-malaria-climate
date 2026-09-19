from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def merge_epidemiological_and_climate_data() -> pd.DataFrame:
    """Merge the malaria incidence series with climate and lagged features."""
    print("Loading datasets...")

    path_malaria = DATA_PROCESSED / "togo_malaria_incidence_real.csv"
    path_climate = DATA_PROCESSED / "togo_features_dataset.csv"

    if not path_malaria.exists():
        raise FileNotFoundError(f"Missing file: {path_malaria}. Run data acquisition first.")
    if not path_climate.exists():
        raise FileNotFoundError(f"Missing file: {path_climate}. Run feature engineering first.")

    df_malaria = pd.read_csv(path_malaria)
    df_climate = pd.read_csv(path_climate)
    df_malaria["date"] = pd.to_datetime(df_malaria["date"])
    df_climate["date"] = pd.to_datetime(df_climate["date"])

    climate_cols = [col for col in df_climate.columns if col not in ["district_name", "region"]]
    df_merged = pd.merge(df_malaria, df_climate[climate_cols], on=["district_id", "date"], how="inner")
    df_merged = df_merged.sort_values(by=["district_id", "date"]).reset_index(drop=True)

    null_counts = df_merged.isnull().sum().sum()
    if null_counts > 0:
        print(f"Detected {null_counts} missing values after merging; dropping them.")
        df_merged = df_merged.dropna().reset_index(drop=True)

    output_path = DATA_PROCESSED / "togo_final_modeling_dataset.csv"
    df_merged.to_csv(output_path, index=False)
    print(f"Final dataset saved: {output_path}")
    print(f"Rows: {len(df_merged)} | columns: {len(df_merged.columns)}")
    return df_merged


if __name__ == "__main__":
    merge_epidemiological_and_climate_data()
