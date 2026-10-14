import json
import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

DATA_PATH = "data/demand_timeseries.csv"
CAUSAL_FEATURES_PATH = "artifacts/causal_features.json"
ARTIFACTS_DIR = "artifacts"

BASE_FEATURES = [
    "price",
    "promotion_flag",
    "lead_time",
    "inventory_level",
    "spurious_noise_feature",
    "marketing_spend_proxy",
    "competitor_price_index",
    "disruption_indicator",
    "severity_level",
    "day_of_week",
    "month",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
]


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    return df


def engineer_features(df):
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["lag_1"] = df.groupby("sku_id")["demand"].shift(1)
    df["lag_7"] = df.groupby("sku_id")["demand"].shift(7)
    df["rolling_mean_7"] = (
        df.groupby("sku_id")["demand"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    )
    df = df.dropna().reset_index(drop=True)
    return df


def chronological_split(df, train_frac=0.7, val_frac=0.15):
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def recovery_speed(y_true, y_pred, disruption_indicator, pre_disruption_error, threshold_ratio=1.05):
    disruption_indicator = np.array(disruption_indicator)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    abs_errors = np.abs(y_true - y_pred)

    disruption_starts = np.where(
        (disruption_indicator[1:] == 1) & (disruption_indicator[:-1] == 0)
    )[0] + 1

    if len(disruption_starts) == 0:
        return None

    recovery_times = []
    threshold = pre_disruption_error * threshold_ratio

    for start in disruption_starts:
        recovered_at = None
        for offset, err in enumerate(abs_errors[start:]):
            if err <= threshold:
                recovered_at = offset
                break
        if recovered_at is not None:
            recovery_times.append(recovered_at)

    if not recovery_times:
        return None
    return float(np.mean(recovery_times))


def train_xgb(train, val, features):
    X_train = train[features]
    y_train = train["demand"]
    X_val = val[features]
    y_val = val["demand"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def recursive_forecast_eval_single_sku(model, sku_df, features, warmup=7):
    sku_df = sku_df.reset_index(drop=True)
    n = len(sku_df)
    if n <= warmup:
        return np.array([]), np.array([]), np.array([])

    demand_history = list(sku_df["demand"].values[:warmup])
    y_true_all = sku_df["demand"].values
    y_pred_all = np.full(n, np.nan)

    for i in range(warmup, n):
        row = sku_df.iloc[i:i + 1].copy()
        row["lag_1"] = demand_history[-1]
        row["lag_7"] = demand_history[-7] if len(demand_history) >= 7 else demand_history[0]
        row["rolling_mean_7"] = np.mean(demand_history[-7:])

        pred = float(model.predict(row[features])[0])
        y_pred_all[i] = pred
        demand_history.append(pred)

    valid_mask = ~np.isnan(y_pred_all)
    return (
        y_true_all[valid_mask],
        y_pred_all[valid_mask],
        sku_df["disruption_indicator"].values[valid_mask],
    )


def recursive_forecast_eval(model, df_split, features, warmup=7):
    all_true, all_pred, all_flags = [], [], []

    for sku_id, sku_df in df_split.groupby("sku_id"):
        sku_df = sku_df.sort_values("date")
        y_true, y_pred, flags = recursive_forecast_eval_single_sku(model, sku_df, features, warmup)
        if len(y_true) == 0:
            continue
        all_true.append(y_true)
        all_pred.append(y_pred)
        all_flags.append(flags)

    if not all_true:
        return np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_flags),
    )


def evaluate_model(model, df_split, features):
    y_true, y_pred, disruption_flags = recursive_forecast_eval(model, df_split, features)

    normal_mask = disruption_flags == 0
    disruption_mask = disruption_flags == 1

    normal_mape = mape(y_true[normal_mask], y_pred[normal_mask]) if normal_mask.sum() > 0 else None
    normal_rmse = rmse(y_true[normal_mask], y_pred[normal_mask]) if normal_mask.sum() > 0 else None
    disruption_mape = mape(y_true[disruption_mask], y_pred[disruption_mask]) if disruption_mask.sum() > 0 else None
    disruption_rmse = rmse(y_true[disruption_mask], y_pred[disruption_mask]) if disruption_mask.sum() > 0 else None

    pre_disruption_error = normal_rmse if normal_rmse is not None else rmse(y_true, y_pred)
    recovery = recovery_speed(
        y_true, y_pred, disruption_flags, pre_disruption_error,
        threshold_ratio=1.05
    )

    return {
        "overall_mape": mape(y_true, y_pred),
        "overall_rmse": rmse(y_true, y_pred),
        "normal_period_mape": normal_mape,
        "normal_period_rmse": normal_rmse,
        "disruption_period_mape": disruption_mape,
        "disruption_period_rmse": disruption_rmse,
        "recovery_speed_timesteps": recovery,
    }


def main():
    df = load_data()
    df = engineer_features(df)

    with open(CAUSAL_FEATURES_PATH, "r") as f:
        causal_info = json.load(f)
    causal_features = [f for f in causal_info["causal_features"] if f in BASE_FEATURES]
    engineered_extra = ["day_of_week", "month", "lag_1", "lag_7", "rolling_mean_7"]
    causal_feature_set = list(dict.fromkeys(causal_features + engineered_extra))
    baseline_feature_set = BASE_FEATURES

    train, val, test = chronological_split(df)

    baseline_model = train_xgb(train, val, baseline_feature_set)
    causal_model = train_xgb(train, val, causal_feature_set)

    print("Evaluating baseline model (recursive forecast, may take a moment)...")
    baseline_test_metrics = evaluate_model(baseline_model, test, baseline_feature_set)

    print("Evaluating causal model (recursive forecast, may take a moment)...")
    causal_test_metrics = evaluate_model(causal_model, test, causal_feature_set)

    metrics = {
        "baseline_correlational": {
            "features_used": baseline_feature_set,
            "test": baseline_test_metrics,
        },
        "craft_causal": {
            "features_used": causal_feature_set,
            "test": causal_test_metrics,
        },
    }

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACTS_DIR, "correlational_xgb.pkl"), "wb") as f:
        pickle.dump({"model": baseline_model, "features": baseline_feature_set}, f)
    with open(os.path.join(ARTIFACTS_DIR, "causal_xgb.pkl"), "wb") as f:
        pickle.dump({"model": causal_model, "features": causal_feature_set}, f)
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()