"""Train tree models for a single forecast horizon with nested time-series tuning."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.config import DATA_PROCESSED, FORECAST_HORIZONS, RANDOM_SEED, TARGET_PREFIX
from src.features.engineering import prepare_horizon_frame
from src.logging_utils import get_logger
from src.modeling.features import feature_columns
from src.modeling.metrics import regression_metrics

logger = get_logger(__name__)

XGB_PARAM_GRID = {
    "model__max_depth": [3, 6],
    "model__n_estimators": [100, 200],
}


def load_final_dataset() -> pd.DataFrame:
    """Load the merged forecast-ready modeling dataset."""
    path = DATA_PROCESSED / "togo_final_modeling_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run the preprocessing pipeline first.")
    return pd.read_csv(path, parse_dates=["date"])


def _sorted_time_split(train_df: pd.DataFrame, n_splits: int = 3) -> TimeSeriesSplit:
    n_splits = max(2, min(n_splits, max(2, train_df["date"].nunique() - 1)))
    return TimeSeriesSplit(n_splits=min(3, n_splits))


def fit_xgboost(train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> xgb.XGBRegressor:
    """Tune XGBoost on an inner time-series split of the training origins only."""
    ordered = train_df.sort_values("date")
    pipe = Pipeline(
        [
            (
                "model",
                xgb.XGBRegressor(
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                ),
            )
        ]
    )
    n_unique = ordered["date"].nunique()
    if n_unique < 6:
        model = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        model.fit(ordered[feature_cols], ordered[target_col])
        return model

    search = GridSearchCV(
        pipe,
        XGB_PARAM_GRID,
        cv=_sorted_time_split(ordered),
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        refit=True,
    )
    search.fit(ordered[feature_cols], ordered[target_col])
    logger.info("XGBoost inner CV best params: %s", search.best_params_)
    return search.best_estimator_.named_steps["model"]


def fit_random_forest(train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    model.fit(train_df[feature_cols], train_df[target_col])
    return model


def fit_lightgbm(train_df: pd.DataFrame, feature_cols: list[str], target_col: str):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(train_df[feature_cols], train_df[target_col])
    return model


def train_and_predict(
    test_df: pd.DataFrame | None = None,
    horizon: int = 1,
    dataset: pd.DataFrame | None = None,
):
    """Fit the tuned XGBoost model for one horizon and predict a provided or last-year test set."""
    df = dataset if dataset is not None else load_final_dataset()
    frame = prepare_horizon_frame(df, horizon)
    target_col = f"{TARGET_PREFIX}{horizon}"
    feature_cols = feature_columns(frame)
    cutoff = frame[f"target_date_h{horizon}"].max() - pd.offsets.MonthBegin(11)
    train_df = frame[frame[f"target_date_h{horizon}"] < cutoff].copy()
    eval_df = test_df if test_df is not None else frame[frame[f"target_date_h{horizon}"] >= cutoff].copy()
    if test_df is not None:
        eval_df = prepare_horizon_frame(test_df, horizon) if target_col not in test_df.columns or "known_target_month_sin" not in test_df.columns else test_df.copy()

    model = fit_xgboost(train_df, feature_cols, target_col)
    predictions = model.predict(eval_df[feature_cols])
    metrics = regression_metrics(eval_df[target_col].to_numpy(), predictions)
    pred_frame = eval_df[["district_id", "district_name", "date"]].copy()
    pred_frame["horizon"] = horizon
    pred_frame["predicted_incidence"] = predictions
    pred_frame["observed_incidence"] = eval_df[target_col].to_numpy()
    pred_frame["target_date"] = eval_df[f"target_date_h{horizon}"]
    return model, pred_frame, metrics, feature_cols


if __name__ == "__main__":
    for horizon in FORECAST_HORIZONS:
        _, predictions, metrics, _ = train_and_predict(horizon=horizon)
        logger.info("Horizon %s metrics: %s", horizon, metrics)
        logger.info("Preview:\n%s", predictions.head())
