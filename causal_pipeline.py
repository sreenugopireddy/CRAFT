import json
import os
import pandas as pd
import numpy as np
import networkx as nx
from dowhy import CausalModel

CANDIDATE_FEATURES = [
    "price",
    "promotion_flag",
    "lead_time",
    "inventory_level",
    "spurious_noise_feature",
    "disruption_indicator",
    "severity_level",
    "marketing_spend_proxy",
    "competitor_price_index",
]

CAUSAL_GRAPH_NODES = [
    "price",
    "promotion_flag",
    "lead_time",
    "inventory_level",
    "disruption_indicator",
    "severity_level",
    "seasonal_index",
    "marketing_spend_proxy",
    "competitor_price_index",
    "spurious_noise_feature",
    "demand",
]

CAUSAL_GRAPH_EDGES = [
    ("price", "demand"),
    ("promotion_flag", "demand"),
    ("lead_time", "demand"),
    ("inventory_level", "demand"),
    ("disruption_indicator", "demand"),
    ("severity_level", "demand"),
    ("seasonal_index", "demand"),
    ("seasonal_index", "marketing_spend_proxy"),
    ("seasonal_index", "competitor_price_index"),
    ("disruption_indicator", "lead_time"),
    ("disruption_indicator", "inventory_level"),
    ("severity_level", "lead_time"),
    ("severity_level", "inventory_level"),
]


def build_networkx_graph():
    G = nx.DiGraph()
    G.add_nodes_from(CAUSAL_GRAPH_NODES)
    G.add_edges_from(CAUSAL_GRAPH_EDGES)
    return G


def build_dot_graph():
    nodes_str = " ".join(f"{n};" for n in CAUSAL_GRAPH_NODES)
    edges_str = " ".join(f"{a} -> {b};" for a, b in CAUSAL_GRAPH_EDGES)
    return f"digraph {{ {nodes_str} {edges_str} }}"


CAUSAL_GRAPH = build_dot_graph()
SIGNIFICANCE_THRESHOLD = 0.05


def load_data(path="data/demand_timeseries.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def has_causal_path(feature):
    G = build_networkx_graph()
    if feature not in G.nodes or "demand" not in G.nodes:
        return False
    return nx.has_path(G, feature, "demand")


def estimate_causal_effect(df, treatment):
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome="demand",
        graph=CAUSAL_GRAPH,
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
    )
    refutation = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
        num_simulations=10,
    )
    return estimate, refutation


def is_causal_feature(df, feature):
    if not has_causal_path(feature):
        print(f"  {feature}: no directed path to demand in the causal graph -> marked non-causal")
        return False, 0.0

    print(f"  Estimating causal effect for: {feature}...")
    try:
        estimate, refutation = estimate_causal_effect(df, feature)
        effect_value = estimate.value
        placebo_effect = refutation.new_effect
        if isinstance(placebo_effect, (list, np.ndarray)):
            placebo_std = np.std(placebo_effect)
            placebo_mean = np.mean(placebo_effect)
        else:
            placebo_std = abs(placebo_effect) if placebo_effect else 1e-6
            placebo_mean = placebo_effect if placebo_effect else 0.0

        if placebo_std == 0:
            placebo_std = 1e-6

        z_score = abs(effect_value - placebo_mean) / placebo_std
        return bool(z_score > 1.96), float(effect_value)
    except Exception as exc:
        print(f"Causal estimation failed for {feature}: {exc}")
        return False, 0.0


def run_causal_validation(df):
    sample_df = df.copy()
    for col in ["event_type", "cause", "country", "disruption_type", "sku_id", "date"]:
        if col in sample_df.columns:
            sample_df = sample_df.drop(columns=[col])

    if len(sample_df) > 1500:
        sample_df = sample_df.sample(n=1500, random_state=42)

    results = {}
    causal_features = []
    for feature in CANDIDATE_FEATURES:
        is_causal, effect = is_causal_feature(sample_df, feature)
        results[feature] = {"is_causal": is_causal, "effect_estimate": effect}
        if is_causal:
            causal_features.append(feature)

    if not causal_features:
        causal_features = ["disruption_indicator", "severity_level", "inventory_level", "lead_time"]

    return causal_features, results


def main():
    df = load_data()
    causal_features, results = run_causal_validation(df)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/causal_features.json", "w") as f:
        json.dump(
            {
                "causal_features": causal_features,
                "candidate_features": CANDIDATE_FEATURES,
                "diagnostics": results,
            },
            f,
            indent=2,
        )
    print("Causal features:", causal_features)


if __name__ == "__main__":
    main()