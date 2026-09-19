from __future__ import annotations

import pandas as pd

from src.modeling.training import load_final_dataset


def evaluate_walk_forward(df: pd.DataFrame | None = None):
    """Evaluate the model using a strict temporal split with no leakage."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb
    import numpy as np

    dataset = df if df is not None else load_final_dataset()
    target_col = "malaria_incidence"
    exclude_cols = ["district_id", "district_name", "region", "date", target_col]
    feature_cols = [col for col in dataset.columns if col not in exclude_cols]

    train_df = dataset[dataset["date"] < "2023-01-01"].copy()
    test_df = dataset[dataset["date"] >= "2023-01-01"].copy()

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    models = {
        "Baseline (Historical Mean)": None,
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "XGBoost Regressor": xgb.XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=6, random_state=42),
    }

    results = []
    for name, model in models.items():
        if model is None:
            preds = np.full(shape=len(y_test), fill_value=y_train.mean())
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append({
            "Model": name,
            "MAE (Cases/1,000 inhabitants)": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R² Score": round(r2, 4),
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return results_df, models, feature_cols


if __name__ == "__main__":
    evaluate_walk_forward()
