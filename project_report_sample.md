**What We Built:**
- ✅ Synthetic multi-SKU demand time series with trend, seasonality, and controlled disruption injection
- ✅ DoWhy-based causal feature validation against a declared structural causal graph
- ✅ Two XGBoost regressors trained and compared: correlational baseline vs. causally-validated feature set
- ✅ Chronological (non-shuffled) train/val/test splits, appropriate for time series
- ✅ Recursive multi-step forecast evaluation (avoids leaking future ground truth via lag features)
- ✅ MAPE, RMSE, and recovery-speed metrics, broken out by normal vs. disruption periods
- ✅ Adaptive model updating via XGBoost warm-start, without full retraining
- ✅ Flask web app serving forecasts, disruption simulation, and performance comparison
- ✅ Retained from the original fork: authentication, CSRF/CSP/rate-limiting, database migrations