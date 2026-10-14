import numpy as np
import pandas as pd
import os

np.random.seed(42)

SKUS = [f"SKU_{i:03d}" for i in range(1, 11)]
COUNTRIES = ["USA", "India", "Germany", "Brazil", "China"]
EVENT_TYPES = ["natural_disaster", "geopolitical", "pandemic", "port_closure", "supplier_failure", "none"]
CAUSES = ["weather", "conflict", "logistics", "labor", "regulatory", "none"]

START_DATE = "2021-01-01"
END_DATE = "2024-12-31"
DATES = pd.date_range(START_DATE, END_DATE, freq="D")

DISRUPTION_TYPES = ["demand_spike", "demand_drop", "supply_delay"]


def base_demand_components(n, sku_seed):
    rng = np.random.default_rng(sku_seed)
    t = np.arange(n)
    trend = 50 + 0.01 * t
    weekly = 8 * np.sin(2 * np.pi * t / 7)
    yearly = 20 * np.sin(2 * np.pi * t / 365.25)
    noise = rng.normal(0, 4, n)
    seasonal_index = weekly + yearly
    demand = trend + seasonal_index + noise
    demand = np.clip(demand, 1, None)
    return demand, seasonal_index


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


def generate_exogenous(n, sku_seed):
    rng = np.random.default_rng(sku_seed + 1000)
    price = 20 + rng.normal(0, 1.5, n).cumsum() * 0.01
    price = np.clip(price, 10, None)
    promotion_flag = rng.binomial(1, 0.08, n)
    lead_time = np.clip(rng.normal(7, 2, n), 1, None)
    inventory_level = np.clip(rng.normal(500, 80, n), 0, None)
    random_noise_feature = rng.normal(0, 1, n)
    return price, promotion_flag, lead_time, inventory_level, random_noise_feature


def generate_confounded_features(seasonal_index, sku_seed):
    n = len(seasonal_index)
    rng = np.random.default_rng(sku_seed + 2000)

    marketing_noise = rng.normal(0, 3, n)
    marketing_spend_proxy = 40 + 1.2 * seasonal_index + marketing_noise
    marketing_spend_proxy = np.clip(marketing_spend_proxy, 0, None)

    competitor_noise = rng.normal(0, 2, n)
    competitor_price_index = 60 - 0.8 * seasonal_index + competitor_noise
    competitor_price_index = np.clip(competitor_price_index, 5, None)

    return marketing_spend_proxy, competitor_price_index


def build_sku_frame(sku, sku_idx):
    n = len(DATES)
    demand, seasonal_index = base_demand_components(n, sku_idx)
    price, promotion_flag, lead_time, inventory_level, spurious_noise = generate_exogenous(n, sku_idx)
    marketing_spend_proxy, competitor_price_index = generate_confounded_features(seasonal_index, sku_idx)

    disruption_indicator = np.zeros(n, dtype=int)
    event_type = np.array(["none"] * n, dtype=object)
    cause = np.array(["none"] * n, dtype=object)
    severity_level = np.zeros(n, dtype=int)
    disruption_type_col = np.array(["none"] * n, dtype=object)

    rng = np.random.default_rng(sku_idx + 5000)
    n_events = rng.integers(3, 7)
    for _ in range(n_events):
        start_idx = rng.integers(60, n - 60)
        duration = rng.integers(10, 45)
        d_type = rng.choice(DISRUPTION_TYPES)
        magnitude = rng.uniform(0.3, 0.9)
        severity = rng.integers(1, 5)
        e_type = rng.choice([e for e in EVENT_TYPES if e != "none"])
        c = rng.choice([c for c in CAUSES if c != "none"])

        # NOTE: demand is perturbed here, but marketing_spend_proxy and
        # competitor_price_index are NOT — they keep following their normal
        # seasonal path, which is what breaks the naive correlation during
        # disruption windows.
        demand = inject_disruption(demand, start_idx, duration, d_type, magnitude)

        end_idx = min(start_idx + duration, n)
        disruption_indicator[start_idx:end_idx] = 1
        event_type[start_idx:end_idx] = e_type
        cause[start_idx:end_idx] = c
        severity_level[start_idx:end_idx] = severity
        disruption_type_col[start_idx:end_idx] = d_type

        lead_time[start_idx:end_idx] = lead_time[start_idx:end_idx] * (1 + 0.5 * severity / 4)
        inventory_level[start_idx:end_idx] = inventory_level[start_idx:end_idx] * (1 - 0.15 * severity / 4)

    country = rng.choice(COUNTRIES)

    df = pd.DataFrame({
        "date": DATES,
        "sku_id": sku,
        "demand": np.clip(demand, 0, None),
        "price": price,
        "promotion_flag": promotion_flag,
        "lead_time": lead_time,
        "inventory_level": inventory_level,
        "spurious_noise_feature": spurious_noise,
        "seasonal_index": seasonal_index,
        "marketing_spend_proxy": marketing_spend_proxy,
        "competitor_price_index": competitor_price_index,
        "disruption_indicator": disruption_indicator,
        "event_type": event_type,
        "severity_level": severity_level,
        "cause": cause,
        "country": country,
        "disruption_type": disruption_type_col,
    })
    return df


def main():
    frames = [build_sku_frame(sku, idx) for idx, sku in enumerate(SKUS)]
    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "demand_timeseries.csv")
    full_df.to_csv(out_path, index=False)
    print(f"Saved {len(full_df)} rows to {out_path}")


if __name__ == "__main__":
    main()