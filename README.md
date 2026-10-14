# CRAFT — Causal-Resilient Adaptive Forecasting Technology

**Disruption-invariant demand forecasting for supply chains, built on a causal-ML + XGBoost pipeline.**

This project began as a fork of an AI-Powered Supply Chain Disruption Prediction platform
(binary event classification) and was rebuilt around a different problem: forecasting future
demand under real-world disruptions, using causal feature validation to separate genuine
demand drivers from spurious correlations that break down during disruption events.

## What this actually does

- Generates a synthetic multi-SKU daily demand time series with trend, seasonality, and
  controlled disruption injection (demand spikes, demand drops, supply delays).
- Runs DoWhy-based causal feature validation against a declared structural causal graph,
  screening out features with no causal path to demand (including deliberately-injected
  spurious/confounded features).
- Trains two XGBoost regressors — one on all candidate features ("correlational baseline"),
  one on only the causally-validated feature set ("CRAFT causal") — and evaluates both with
  chronological (non-shuffled) train/val/test splits.
- Evaluates both models with MAPE, RMSE, and a recovery-speed metric (timesteps to return
  within a tolerance of pre-disruption error), computed via **recursive multi-step forecasting**
  per SKU rather than one-step lookups, to avoid leaking future ground truth into lag features.
- Supports adaptive model updates via XGBoost warm-starting (`adaptive_update.py`) — new
  disrupted data can be incorporated without retraining from scratch.
- Serves forecasts, disruption simulations, and performance metrics through a Flask web app
  (`/forecast`, `/simulate-disruption`, `/performance`, `/causal-graph`).

## Pipeline (run in this exact order)

```bash
python generate_data.py       # writes data/demand_timeseries.csv
python causal_pipeline.py     # writes artifacts/causal_features.json
python train_model.py         # writes artifacts/*.pkl and artifacts/metrics.json
flask run                     # serves the web app
```

## Local setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

Set `.env` for local development:
```dotenv
FLASK_ENV=development
FLASK_DEBUG=true
SECRET_KEY=dev-local-secret-not-for-production
DATABASE_URL=sqlite:///local.db
TALISMAN_FORCE_HTTPS=false
CORS_ORIGINS=http://localhost:5000
RATELIMIT_STORAGE_URI=memory://
```

Then:
```bash
flask db upgrade
flask run
```

## Key results (current run)

Causal validation correctly excludes a deliberately-injected pure-noise feature and two
seasonally-confounded features (spurious correlation with demand via a shared seasonality
confounder, no real causal effect), keeping `{price, lead_time, inventory_level,
disruption_indicator, severity_level}` plus engineered lag/calendar features.

On a held-out chronological test split with recursive multi-step evaluation:

| Metric | Correlational baseline | CRAFT (causal) |
|---|---|---|
| Overall MAPE | 13.67% | 12.77% |
| Overall RMSE | 9.04 | 8.93 |
| Normal-period MAPE | 12.31% | 11.18% |
| Disruption-period MAPE | 35.66% | 38.46% |
| Recovery speed (timesteps) | 23.0 | 21.5 |

The causal model generalizes better overall and recovers marginally faster, but shows higher
error specifically within disruption windows — likely because causal filtering reduces the
feature count available for reacting to acute, short-term shocks even as it removes reliance
on spurious correlations. This nuance is discussed further in the project report.

## Legacy code

`app/artifacts.py`, `app/validators.py`, `data/supply_chain_events.csv`, and the commented-out
`/api/v1/predict` route in `app/routes/api.py` are remnants of the original classification-based
fork and are not part of the CRAFT forecasting pipeline. They remain functional (via `/api/v1/metrics`,
`/api/v1/analytics/trends`) but report on the old dataset/task, not this project's results.

## Requirements

See `requirements.txt`. Core additions over the original fork: `xgboost`, `dowhy`, `networkx`.