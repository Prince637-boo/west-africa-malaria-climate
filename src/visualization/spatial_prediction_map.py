"""Maps and district time series for forecast error analysis.

Figure numbers belong in captions (README or manuscript), not in the image title.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from src.config import DATA_RAW, FIGURE_DIR, TARGET_PREFIX, ensure_data_dirs
from src.features.engineering import prepare_horizon_frame
from src.logging_utils import get_logger
from src.modeling.features import feature_columns
from src.modeling.training import fit_xgboost, load_final_dataset
from src.modeling.evaluation import walk_forward_splits

logger = get_logger(__name__)


def _last_fold_xgb_predictions(dataset: pd.DataFrame | None = None, horizon: int = 1) -> pd.DataFrame:
    df = dataset if dataset is not None else load_final_dataset()
    ready = prepare_horizon_frame(df, horizon)
    splits = list(walk_forward_splits(ready, horizon))
    if not splits:
        raise RuntimeError("No walk-forward fold available for mapping.")
    _, train_df, test_df = splits[-1]
    target_col = f"{TARGET_PREFIX}{horizon}"
    cols = feature_columns(ready)
    model = fit_xgboost(train_df, cols, target_col)
    preds = model.predict(test_df[cols])
    out = test_df[["district_id", "district_name", "date", target_col, f"target_date_h{horizon}"]].copy()
    out["observed_incidence"] = test_df[target_col].to_numpy()
    out["predicted_incidence"] = preds
    out["absolute_error"] = (out["predicted_incidence"] - out["observed_incidence"]).abs()
    return out


def _load_district_prediction_frame() -> pd.DataFrame:
    """Mean observed/predicted/error over the last walk-forward test year (h=1)."""
    monthly = _last_fold_xgb_predictions(horizon=1)
    return (
        monthly.groupby("district_id", as_index=False)
        .agg(
            observed_incidence=("observed_incidence", "mean"),
            predicted_incidence=("predicted_incidence", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
        )
    )


def generate_spatial_prediction_figure(
    output_path: str | None = None,
    comparison: pd.DataFrame | None = None,
    geo_path: str | Path | None = None,
) -> str:
    """Choropleth of mean observed incidence, prediction, and absolute error."""
    ensure_data_dirs()
    if comparison is None:
        comparison = _load_district_prediction_frame()
    district_geo = gpd.read_file(geo_path or (DATA_RAW / "togo_districts.geojson"))
    district_geo = district_geo.rename(columns={"GID_2": "district_id", "NAME_2": "district_name"})
    district_geo["district_id"] = district_geo["district_id"].astype(str)
    comparison = comparison.copy()
    comparison["district_id"] = comparison["district_id"].astype(str)
    spatial_df = district_geo.merge(comparison, on="district_id", how="left")

    if spatial_df["observed_incidence"].isna().any() or spatial_df["predicted_incidence"].isna().any():
        raise ValueError("Some districts are missing observed or predicted incidence values.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    panels = [
        (axes[0], "observed_incidence", "Observed (simulated) mean incidence", "YlOrRd"),
        (axes[1], "predicted_incidence", "Predicted mean incidence (h=1)", "YlOrRd"),
        (axes[2], "mean_absolute_error", "Mean absolute error", "PuRd"),
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
        output_path = str(FIGURE_DIR / "spatial_prediction_and_error_map.png")
    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(final_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote map to %s", final_path)
    return str(final_path)


def generate_district_timeseries_figure(
    output_path: str | None = None,
    n_districts: int = 4,
) -> str:
    """Monthly observed vs predicted series for a few units on the last test year."""
    ensure_data_dirs()
    monthly = _last_fold_xgb_predictions(horizon=1)
    ids = list(monthly["district_id"].drop_duplicates().head(n_districts))
    fig, axes = plt.subplots(len(ids), 1, figsize=(10, 2.4 * len(ids)), sharex=True)
    if len(ids) == 1:
        axes = [axes]
    for ax, dist_id in zip(axes, ids):
        part = monthly[monthly["district_id"] == dist_id].sort_values("target_date_h1")
        name = part["district_name"].iloc[0]
        ax.plot(part["target_date_h1"], part["observed_incidence"], label="Observed (simulated)")
        ax.plot(part["target_date_h1"], part["predicted_incidence"], label="Predicted (h=1)", linestyle="--")
        ax.set_ylabel("Incidence")
        ax.set_title(str(name))
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Target month")
    if output_path is None:
        output_path = str(FIGURE_DIR / "district_forecast_timeseries.png")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote time series figure to %s", path)
    return str(path)


if __name__ == "__main__":
    generate_spatial_prediction_figure()
    generate_district_timeseries_figure()
