"""Maps and district time series for forecast error analysis.

Figure numbers belong in captions (README or manuscript), not in the image title.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from src.config import BOUNDARIES_DIR, DATA_PROCESSED, DATA_RAW, FIGURE_DIR, ensure_data_dirs
from src.logging_utils import get_logger
from src.modeling.features import feature_columns, prepare_annual_features

logger = get_logger(__name__)


def load_final_dataset() -> pd.DataFrame:
    """Load processed merged dataset from storage."""
    dataset_path = DATA_PROCESSED / "merged_dataset.csv"
    if not dataset_path.exists():
        dataset_path = DATA_PROCESSED / "final_malaria_climate_dataset.csv"
    if not dataset_path.exists():
        processed_files = list(DATA_PROCESSED.glob("*.csv"))
        if not processed_files:
            raise FileNotFoundError(f"No processed CSV files found in {DATA_PROCESSED}")
        dataset_path = processed_files[0]

    return pd.read_csv(dataset_path)


def resolve_geojson_path(geo_path: str | Path | None = None) -> Path:
    """Resolve the location of spatial GeoJSON or JSON files across project boundaries."""
    if geo_path is not None:
        p = Path(geo_path)
        if p.exists():
            return p

    candidates = [
        BOUNDARIES_DIR / "togo_districts.geojson",
        BOUNDARIES_DIR / "gadm41_TGO_2.json",
        DATA_RAW / "togo_districts.geojson",
        DATA_RAW / "gadm41_TGO_2.json",
        DATA_RAW / "boundaries" / "gadm41_TGO_2.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No valid spatial boundaries GeoJSON file found. Checked paths: {[str(c) for c in candidates]}"
    )


def _last_fold_xgb_predictions(dataset: pd.DataFrame | None = None) -> pd.DataFrame:
    """Generate last-fold test predictions using XGBoost."""
    df = dataset if dataset is not None else load_final_dataset()
    ready = prepare_annual_features(df)

    # Split train (up to 2024) / test (2025)
    train_df = ready[ready["year"] < 2025].copy()
    test_df = ready[ready["year"] == 2025].copy()

    if train_df.empty or test_df.empty:
        raise RuntimeError("No temporal split available for year 2025 forecast.")

    target_col = "pf_incidence_rate"
    cols = feature_columns(ready)

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    model.fit(train_df[cols], train_df[target_col])
    preds = model.predict(test_df[cols])

    out = test_df[["district_id", "district_name", "year", target_col]].copy()
    out["observed_incidence"] = test_df[target_col].to_numpy()
    out["predicted_incidence"] = preds
    out["absolute_error"] = (out["predicted_incidence"] - out["observed_incidence"]).abs()
    return out


def _load_district_prediction_frame() -> pd.DataFrame:
    """Compute mean observed, predicted, and error values over the test year 2025."""
    annual = _last_fold_xgb_predictions()
    return (
        annual.groupby("district_id", as_index=False)
        .agg(
            observed_incidence=("observed_incidence", "mean"),
            predicted_incidence=("predicted_incidence", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
        )
    )


def generate_spatial_prediction_figure(
    output_path: str | Path | None = None,
    comparison: pd.DataFrame | None = None,
    geo_path: str | Path | None = None,
) -> str:
    """Generate a choropleth map comparing observed, predicted incidence, and absolute error."""
    ensure_data_dirs()
    if comparison is None:
        comparison = _load_district_prediction_frame()

    target_geo_path = resolve_geojson_path(geo_path)
    district_geo = gpd.read_file(target_geo_path)

    # Normalize administrative columns
    for col in ["GID_2", "NAME_2", "district_id", "district_name"]:
        if col in district_geo.columns:
            if col == "GID_2":
                district_geo = district_geo.rename(columns={"GID_2": "district_id"})
            elif col == "NAME_2":
                district_geo = district_geo.rename(columns={"NAME_2": "district_name"})

    district_geo["district_id"] = district_geo["district_id"].astype(str)
    comparison = comparison.copy()
    comparison["district_id"] = comparison["district_id"].astype(str)

    spatial_df = district_geo.merge(comparison, on="district_id", how="left")

    if spatial_df["observed_incidence"].isna().any() or spatial_df["predicted_incidence"].isna().any():
        logger.warning("Some spatial district regions failed to match the prediction frame.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    panels = [
        (axes[0], "observed_incidence", "Observed Incidence (2025)", "YlOrRd"),
        (axes[1], "predicted_incidence", "Predicted Incidence (2025)", "YlOrRd"),
        (axes[2], "mean_absolute_error", "Mean Absolute Error", "PuRd"),
    ]
    for ax, column, title, cmap in panels:
        spatial_df.plot(
            column=column,
            cmap=cmap,
            linewidth=0.6,
            edgecolor="black",
            ax=ax,
            legend=True,
            legend_kwds={"label": column.replace("_", " "), "orientation": "horizontal"},
        )
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()

    if output_path is None:
        output_path = FIGURE_DIR / "spatial_prediction_and_error_map.png"
    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(final_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved spatial map to %s", final_path)
    return str(final_path)


def generate_district_timeseries_figure(
    output_path: str | Path | None = None,
    n_districts: int = 4,
) -> str:
    """Generate annual observed vs predicted time-series plots for selected district units."""
    ensure_data_dirs()
    annual = _last_fold_xgb_predictions()
    ids = list(annual["district_id"].drop_duplicates().head(n_districts))

    fig, axes = plt.subplots(len(ids), 1, figsize=(10, 2.4 * len(ids)), sharex=True)
    if len(ids) == 1:
        axes = [axes]

    for ax, dist_id in zip(axes, ids):
        part = annual[annual["district_id"] == dist_id].sort_values("year")
        name = part["district_name"].iloc[0] if "district_name" in part.columns else str(dist_id)
        ax.plot(part["year"], part["observed_incidence"], label="Observed", marker="o")
        ax.plot(part["year"], part["predicted_incidence"], label="Predicted", linestyle="--", marker="x")
        ax.set_ylabel("Incidence Rate")
        ax.set_title(str(name))
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Year")
    if output_path is None:
        output_path = FIGURE_DIR / "district_forecast_timeseries.png"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved district time-series figure to %s", path)
    return str(path)


if __name__ == "__main__":
    generate_spatial_prediction_figure()
    generate_district_timeseries_figure()