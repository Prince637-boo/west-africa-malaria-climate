"""Walk-forward temporal validation and leave-one-district-out spatial validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FORECAST_HORIZONS, REPORTS_DIR, TARGET_PREFIX, ensure_data_dirs
from src.features.engineering import prepare_horizon_frame
from src.logging_utils import get_logger
from src.modeling.baselines import (
    fit_autoregressive_ridge,
    predict_autoregressive_ridge,
    predict_climatology,
    predict_global_mean,
    predict_seasonal_naive,
)
from src.modeling.features import ablation_feature_sets, feature_columns
from src.modeling.metrics import bootstrap_mae_interval, grouped_mae, regression_metrics, skill_score
from src.modeling.training import fit_lightgbm, fit_random_forest, fit_xgboost, load_final_dataset

logger = get_logger(__name__)


def walk_forward_splits(
    frame: pd.DataFrame,
    horizon: int,
    test_years: tuple[int, ...] | None = None,
):
    """Yield expanding-window folds keyed by the calendar year of the *target* month."""
    target_date = frame[f"target_date_h{horizon}"]
    years = test_years
    if years is None:
        all_years = sorted(target_date.dt.year.unique())
        years = tuple(y for y in all_years if y >= all_years[0] + 2)
    for year in years:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-01")
        train = frame[target_date < start].copy()
        test = frame[(target_date >= start) & (target_date <= end)].copy()
        if train.empty or test.empty:
            logger.warning("Skipping walk-forward year %s (empty train or test).", year)
            continue
        yield year, train, test


def evaluate_horizon(frame: pd.DataFrame, horizon: int) -> dict[str, pd.DataFrame]:
    """Compare baselines and tree models under walk-forward validation for one horizon."""
    target_col = f"{TARGET_PREFIX}{horizon}"
    month_col = f"target_month_h{horizon}"
    ready = prepare_horizon_frame(frame, horizon)
    feature_cols = feature_columns(ready)

    fold_rows = []
    prediction_rows = []

    for year, train_df, test_df in walk_forward_splits(ready, horizon):
        y_test = test_df[target_col].to_numpy()
        climatology = predict_climatology(train_df, test_df, month_col, target_col)
        seasonal = predict_seasonal_naive(test_df, horizon)
        global_mean = predict_global_mean(train_df, len(test_df))
        ar_model = fit_autoregressive_ridge(train_df, target_col)
        ar_pred = predict_autoregressive_ridge(ar_model, test_df)

        rf = fit_random_forest(train_df, feature_cols, target_col)
        xgb_model = fit_xgboost(train_df, feature_cols, target_col)
        lgb_model = fit_lightgbm(train_df, feature_cols, target_col)

        candidates = {
            "Global mean (weak)": global_mean,
            "District-month climatology": climatology,
            "Seasonal naive": seasonal,
            "Autoregressive Ridge": ar_pred,
            "Random Forest": rf.predict(test_df[feature_cols]),
            "XGBoost": xgb_model.predict(test_df[feature_cols]),
            "LightGBM": lgb_model.predict(test_df[feature_cols]),
        }

        clim_mae = regression_metrics(y_test, climatology)["mae"]
        for name, preds in candidates.items():
            if preds is None or np.isnan(preds).any():
                continue
            metrics = regression_metrics(y_test, preds)
            mae_lo, mae_hi = bootstrap_mae_interval(y_test, preds)
            fold_rows.append(
                {
                    "horizon": horizon,
                    "test_year": year,
                    "model": name,
                    "mae": metrics["mae"],
                    "mae_ci_low": mae_lo,
                    "mae_ci_high": mae_hi,
                    "rmse": metrics["rmse"],
                    "r2_pooled": metrics["r2"],
                    "skill_vs_climatology": skill_score(metrics["mae"], clim_mae),
                    "n_test": int(len(y_test)),
                }
            )
            part = test_df[["district_id", "district_name", "date"]].copy()
            part["horizon"] = horizon
            part["test_year"] = year
            part["model"] = name
            part["observed_incidence"] = y_test
            part["predicted_incidence"] = preds
            part["target_date"] = test_df[f"target_date_h{horizon}"].to_numpy()
            part["target_month"] = test_df[month_col].to_numpy()
            prediction_rows.append(part)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    summary = (
        fold_df.groupby(["horizon", "model"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2_pooled", "mean"),
            skill_mean=("skill_vs_climatology", "mean"),
        )
        if not fold_df.empty
        else pd.DataFrame()
    )
    return {"folds": fold_df, "summary": summary, "predictions": pred_df}


def evaluate_ablations(frame: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Retrain XGBoost on feature subsets using the last walk-forward fold."""
    target_col = f"{TARGET_PREFIX}{horizon}"
    ready = prepare_horizon_frame(frame, horizon)
    splits = list(walk_forward_splits(ready, horizon))
    if not splits:
        return pd.DataFrame()
    year, train_df, test_df = splits[-1]
    sets = ablation_feature_sets(feature_columns(ready))
    rows = []
    y_test = test_df[target_col].to_numpy()
    for name, cols in sets.items():
        if not cols:
            continue
        model = fit_xgboost(train_df, cols, target_col)
        preds = model.predict(test_df[cols])
        metrics = regression_metrics(y_test, preds)
        rows.append({"horizon": horizon, "test_year": year, "ablation": name, **metrics, "n_features": len(cols)})
    return pd.DataFrame(rows)


def evaluate_leave_one_district_out(frame: pd.DataFrame, horizon: int = 1, max_districts: int | None = None) -> pd.DataFrame:
    """Spatial block validation: train on all other units, test the held-out unit (last year)."""
    target_col = f"{TARGET_PREFIX}{horizon}"
    ready = prepare_horizon_frame(frame, horizon)
    splits = list(walk_forward_splits(ready, horizon))
    if not splits:
        return pd.DataFrame()
    _, train_all, test_all = splits[-1]
    feature_cols = feature_columns(ready)
    districts = list(test_all["district_id"].unique())
    if max_districts is not None:
        districts = districts[:max_districts]
    rows = []
    for district_id in districts:
        train_df = train_all[train_all["district_id"] != district_id]
        test_df = test_all[test_all["district_id"] == district_id]
        if train_df.empty or test_df.empty:
            continue
        model = fit_xgboost(train_df, feature_cols, target_col)
        preds = model.predict(test_df[feature_cols])
        metrics = regression_metrics(test_df[target_col].to_numpy(), preds)
        rows.append({"horizon": horizon, "held_out_district": district_id, **metrics, "n_test": int(len(test_df))})
    return pd.DataFrame(rows)


def evaluate_walk_forward(df: pd.DataFrame | None = None, horizons: tuple[int, ...] = FORECAST_HORIZONS):
    """Run walk-forward comparison for each forecast horizon and write reports."""
    ensure_data_dirs()
    dataset = df if df is not None else load_final_dataset()
    all_folds = []
    all_summary = []
    last_predictions = None
    for horizon in horizons:
        logger.info("Walk-forward evaluation for horizon h=%s", horizon)
        result = evaluate_horizon(dataset, horizon)
        all_folds.append(result["folds"])
        all_summary.append(result["summary"])
        if horizon == 1:
            last_predictions = result["predictions"]
            ablations = evaluate_ablations(dataset, horizon=1)
            spatial = evaluate_leave_one_district_out(dataset, horizon=1)
            if last_predictions is not None and not last_predictions.empty:
                xgb_last = last_predictions[
                    (last_predictions["model"] == "XGBoost")
                    & (last_predictions["test_year"] == last_predictions["test_year"].max())
                ]
                if not xgb_last.empty:
                    by_district = grouped_mae(xgb_last, "district_id", "observed_incidence", "predicted_incidence")
                    by_month = grouped_mae(xgb_last, "target_month", "observed_incidence", "predicted_incidence")
                    by_district.to_csv(REPORTS_DIR / "mae_by_district_h1.csv", index=False)
                    by_month.to_csv(REPORTS_DIR / "mae_by_month_h1.csv", index=False)
            if not ablations.empty:
                ablations.to_csv(REPORTS_DIR / "ablation_h1.csv", index=False)
                logger.info("Ablation (last fold):\n%s", ablations.to_string(index=False))
            if not spatial.empty:
                spatial.to_csv(REPORTS_DIR / "leave_one_district_out_h1.csv", index=False)
                logger.info(
                    "Leave-one-district-out MAE mean=%.3f",
                    spatial["mae"].mean(),
                )

    folds_df = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    if not folds_df.empty:
        folds_df.to_csv(REPORTS_DIR / "walk_forward_folds.csv", index=False)
        summary_df.to_csv(REPORTS_DIR / "walk_forward_summary.csv", index=False)
        logger.info("Walk-forward summary:\n%s", summary_df.to_string(index=False))
    if last_predictions is not None and not last_predictions.empty:
        last_predictions.to_csv(REPORTS_DIR / "walk_forward_predictions_h1.csv", index=False)
    return summary_df, folds_df


if __name__ == "__main__":
    evaluate_walk_forward()
