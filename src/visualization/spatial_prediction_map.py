from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "notebooks" / "figures"


def _load_district_prediction_frame() -> pd.DataFrame:
    dataset = pd.read_csv(DATA_PROCESSED / "togo_final_modeling_dataset.csv", parse_dates=["date"])
    test_df = dataset[dataset["date"] >= "2023-01-01"].copy()

    from src.modeling.training import train_and_predict

    _, predictions_df, _ = train_and_predict(test_df)

    real_by_district = (
        test_df.groupby("district_id", as_index=False)["malaria_incidence"].mean().rename(columns={"malaria_incidence": "observed_incidence"})
    )
    predicted_by_district = (
        predictions_df.groupby("district_id", as_index=False)["predicted_incidence"].mean()
    )

    comparison = real_by_district.merge(predicted_by_district, on="district_id", how="inner")
    return comparison


def generate_spatial_prediction_figure(output_path: str | None = None) -> str:
    """Create Figure 1: district-level map comparing observed and predicted malaria incidence."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    comparison = _load_district_prediction_frame()
    district_geo = gpd.read_file(DATA_RAW / "togo_districts.geojson")
    district_geo = district_geo.rename(columns={"GID_2": "district_id", "NAME_2": "district_name"})
    district_geo["district_id"] = district_geo["district_id"].astype(str)

    spatial_df = district_geo.merge(comparison, on="district_id", how="left")

    if spatial_df["observed_incidence"].isna().any() or spatial_df["predicted_incidence"].isna().any():
        raise ValueError("Some districts are missing observed or predicted incidence values.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    fig.suptitle("Figure 1. Geographic comparison of observed and predicted malaria incidence by district", fontsize=15, fontweight="bold")

    for ax, column, title, cmap in [
        (axes[0], "observed_incidence", "Observed incidence (cases per 1,000 inhabitants)", "YlOrRd"),
        (axes[1], "predicted_incidence", "Predicted incidence (cases per 1,000 inhabitants)", "YlOrRd"),
    ]:
        spatial_df.plot(
            column=column,
            cmap=cmap,
            linewidth=0.6,
            edgecolor="black",
            ax=ax,
            legend=True,
            legend_kwds={"label": "Incidence", "orientation": "horizontal"},
        )
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()

    if output_path is None:
        output_path = str(FIGURE_DIR / "figure_1_spatial_prediction_map.png")

    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(final_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(final_path)


if __name__ == "__main__":
    generate_spatial_prediction_figure()
