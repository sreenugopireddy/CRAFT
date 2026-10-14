import json
import os
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_PATH = os.path.join(BASE_DIR, "data", "demand_timeseries.csv")

DISRUPTION_TYPES = ["demand_spike", "demand_drop", "supply_delay"]


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_models():
    baseline_payload = _load_pickle(os.path.join(ARTIFACTS_DIR, "correlational_xgb.pkl"))
    causal_payload = _load_pickle(os.path.join(ARTIFACTS_DIR, "causal_xgb.pkl"))
    return baseline_payload, causal_payload


def load_metrics():
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "r") as f:
        return json.load(f)


def load_causal_features():
    with open(os.path.join(ARTIFACTS_DIR, "causal_features.json"), "r") as f:
        return json.load(f)


def load_history(sku_id=None):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    if sku_id:
        df = df[df["sku_id"] == sku_id]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def engineer_features(df):
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["lag_1"] = df["demand"].shift(1)
    df["lag_7"] = df["demand"].shift(7)
    df["rolling_mean_7"] = df["demand"].shift(1).rolling(7).mean()
    df = df.dropna().reset_index(drop=True)
    return df


def inject_disruption(demand, start_idx, duration, disruption_type, magnitude):
    demand = demand.copy()
    end_idx = min(start_idx + duration, len(demand))
    idx = np.arange(start_idx, end_idx)
    decay = np.exp(-np.linspace(0, 3, len(idx)))

    if disruption_type == "demand_spike":
        demand[idx] = demand[idx] * (1 + magnitude * decay)
    elif disruption_type == "demand_drop":
        demand[idx] = demand[idx] * (1 - magnitude * decay)
        demand[idx] = np.clip(demand[idx], 0.5, None)
    elif disruption_type == "supply_delay":
        lag = int(duration * 0.4)
        shifted = np.roll(demand[idx], lag)
        shifted[:lag] = demand[idx][:lag] * (1 - magnitude)
        demand[idx] = shifted * (1 - magnitude * decay * 0.5)

    return demand


def get_available_skus():
    df = pd.read_csv(DATA_PATH, usecols=["sku_id"])
    return sorted(df["sku_id"].unique().tolist())


def forecast_next_n(sku_id, horizon=30):
    baseline_payload, causal_payload = load_models()
    baseline_model, baseline_features = baseline_payload["model"], baseline_payload["features"]
    causal_model, causal_features = causal_payload["model"], causal_payload["features"]

    history = load_history(sku_id)
    history_feat = engineer_features(history)

    last_rows = history_feat.tail(60).copy().reset_index(drop=True)
    working = last_rows.copy()

    dates = []
    baseline_preds = []
    causal_preds = []
    actuals = []
    disruption_flags = []

    last_date = working["date"].iloc[-1]
    last_demand_series = list(working["demand"].values)

    for step in range(horizon):
        next_date = last_date + pd.Timedelta(days=step + 1)
        row = working.iloc[-1:].copy()
        row["date"] = next_date
        row["day_of_week"] = next_date.dayofweek
        row["month"] = next_date.month
        row["lag_1"] = last_demand_series[-1]
        row["lag_7"] = last_demand_series[-7] if len(last_demand_series) >= 7 else last_demand_series[0]
        row["rolling_mean_7"] = np.mean(last_demand_series[-7:])

        X_baseline = row[baseline_features]
        X_causal = row[causal_features]

        base_pred = float(baseline_model.predict(X_baseline)[0])
        causal_pred = float(causal_model.predict(X_causal)[0])

        dates.append(next_date.strftime("%Y-%m-%d"))
        baseline_preds.append(base_pred)
        causal_preds.append(causal_pred)
        disruption_flags.append(int(row["disruption_indicator"].iloc[0]))

        new_row = row.copy()
        new_row["demand"] = causal_pred
        working = pd.concat([working, new_row], ignore_index=True)
        last_demand_series.append(causal_pred)

    history_out = {
        "dates": history["date"].dt.strftime("%Y-%m-%d").tolist()[-90:],
        "actual_demand": history["demand"].tolist()[-90:],
    }

    return {
        "sku_id": sku_id,
        "horizon": horizon,
        "history": history_out,
        "forecast_dates": dates,
        "baseline_forecast": baseline_preds,
        "causal_forecast": causal_preds,
        "disruption_flags": disruption_flags,
    }


def simulate_disruption(sku_id, disruption_type, start_offset, duration, magnitude, horizon=30):
    if disruption_type not in DISRUPTION_TYPES:
        raise ValueError(f"Invalid disruption_type. Choose from {DISRUPTION_TYPES}")

    baseline_payload, causal_payload = load_models()
    baseline_model, baseline_features = baseline_payload["model"], baseline_payload["features"]
    causal_model, causal_features = causal_payload["model"], causal_payload["features"]

    history = load_history(sku_id)
    history_feat = engineer_features(history)
    working = history_feat.tail(60).copy().reset_index(drop=True)

    last_date = working["date"].iloc[-1]
    last_demand_series = list(working["demand"].values)

    dates = []
    baseline_preds = []
    causal_preds = []
    disruption_flags = []

    for step in range(horizon):
        next_date = last_date + pd.Timedelta(days=step + 1)
        row = working.iloc[-1:].copy()
        row["date"] = next_date
        row["day_of_week"] = next_date.dayofweek
        row["month"] = next_date.month
        row["lag_1"] = last_demand_series[-1]
        row["lag_7"] = last_demand_series[-7] if len(last_demand_series) >= 7 else last_demand_series[0]
        row["rolling_mean_7"] = np.mean(last_demand_series[-7:])

        is_disrupted = start_offset <= step < (start_offset + duration)
        row["disruption_indicator"] = 1 if is_disrupted else 0
        if is_disrupted:
            row["severity_level"] = 3
            row["lead_time"] = row["lead_time"] * 1.4
            row["inventory_level"] = row["inventory_level"] * 0.8
        else:
            row["severity_level"] = 0

        X_baseline = row[baseline_features]
        X_causal = row[causal_features]

        base_pred = float(baseline_model.predict(X_baseline)[0])
        causal_pred = float(causal_model.predict(X_causal)[0])

        if is_disrupted:
            offset_in_disruption = step - start_offset
            simulated = inject_disruption(
                np.array([base_pred]), 0, 1, disruption_type, magnitude
            )[0]
            base_pred = float(simulated)
            simulated_causal = inject_disruption(
                np.array([causal_pred]), 0, 1, disruption_type, magnitude * 0.6
            )[0]
            causal_pred = float(simulated_causal)

        dates.append(next_date.strftime("%Y-%m-%d"))
        baseline_preds.append(base_pred)
        causal_preds.append(causal_pred)
        disruption_flags.append(1 if is_disrupted else 0)

        new_row = row.copy()
        new_row["demand"] = causal_pred
        working = pd.concat([working, new_row], ignore_index=True)
        last_demand_series.append(causal_pred)

    return {
        "sku_id": sku_id,
        "disruption_type": disruption_type,
        "start_offset": start_offset,
        "duration": duration,
        "magnitude": magnitude,
        "forecast_dates": dates,
        "baseline_forecast": baseline_preds,
        "causal_forecast": causal_preds,
        "disruption_flags": disruption_flags,
    }


def get_causal_graph_json():
    graph_edges = [
        {"source": "price", "target": "demand"},
        {"source": "promotion_flag", "target": "demand"},
        {"source": "lead_time", "target": "demand"},
        {"source": "inventory_level", "target": "demand"},
        {"source": "disruption_indicator", "target": "demand"},
        {"source": "severity_level", "target": "demand"},
        {"source": "disruption_indicator", "target": "lead_time"},
        {"source": "disruption_indicator", "target": "inventory_level"},
        {"source": "severity_level", "target": "lead_time"},
        {"source": "severity_level", "target": "inventory_level"},
    ]
    causal_features_info = load_causal_features()
    return {
        "edges": graph_edges,
        "causal_features": causal_features_info["causal_features"],
        "candidate_features": causal_features_info["candidate_features"],
    }