import json
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ARTIFACTS_DIR = "artifacts"


def load_model(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["features"]


def save_model(model, features, path):
    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": features}, f)


def warm_start_update(existing_model, features, new_data, n_estimators_increment=50):
    X_new = new_data[features]
    y_new = new_data["demand"]

    new_model = XGBRegressor(
        n_estimators=n_estimators_increment,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    new_model.fit(X_new, y_new, xgb_model=existing_model.get_booster())
    return new_model


def adaptive_reestimate(new_disrupted_data_path, model_path=None, features_path=None):
    model_path = model_path or f"{ARTIFACTS_DIR}/causal_xgb.pkl"
    model, features = load_model(model_path)

    new_data = pd.read_csv(new_disrupted_data_path, parse_dates=["date"])
    new_data["day_of_week"] = new_data["date"].dt.dayofweek
    new_data["month"] = new_data["date"].dt.month
    new_data["lag_1"] = new_data.groupby("sku_id")["demand"].shift(1)
    new_data["lag_7"] = new_data.groupby("sku_id")["demand"].shift(7)
    new_data["rolling_mean_7"] = (
        new_data.groupby("sku_id")["demand"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    )
    new_data = new_data.dropna().reset_index(drop=True)

    missing = [f for f in features if f not in new_data.columns]
    if missing:
        raise ValueError(f"Missing required features in new data: {missing}")

    updated_model = warm_start_update(model, features, new_data)
    save_model(updated_model, features, model_path)
    return updated_model


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python adaptive_update.py <path_to_new_disrupted_data.csv>")
    else:
        adaptive_reestimate(sys.argv[1])
        print("Model updated with warm start.")