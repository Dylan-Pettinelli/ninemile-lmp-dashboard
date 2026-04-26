"""
model.py
--------
Builds a forecasting model that predicts electricity demand for the next 24 hours.

This is the core "quant" piece of the project. The same logic — feature engineering
on time-series data, training a regression model, evaluating it on holdout data —
is what Constellation's Risk Analytics team does with price/position data.

What the model learns:
  - Hour of day (demand peaks mid-afternoon, troughs at 3am)
  - Day of week (weekends are lower)
  - Month of year (summer AC and winter heating spike demand)
  - Recent trend (yesterday's demand predicts today's)
  - Lag features (what happened 1hr, 24hr, 168hr ago)

What you'll learn building this:
  - Feature engineering on time-series data
  - Train/test splits (how to evaluate a model honestly)
  - MAE / RMSE — the error metrics energy traders actually use
  - How to save and reload a model (pickle)
"""

import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

DB_PATH = "energy_data.db"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"


# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw timestamps + demand into a feature matrix for the model.

    This is the most important function in the project — feature engineering
    is where most of the model's predictive power comes from.

    Args:
        df: DataFrame with 'timestamp' and 'demand_mwh' columns

    Returns:
        DataFrame with features + target, ready for sklearn
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Time features ──
    # These capture the cyclical patterns in electricity demand
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_business_hours"] = ((df["hour"] >= 8) & (df["hour"] <= 18)).astype(int)

    # Encode hour cyclically so 23 and 0 are "close" to each other
    # (this is a common trick in time-series ML)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Lag features ──
    # "What was demand 1 hour ago? 24 hours ago? Same time last week?"
    # These are usually the strongest predictors in energy time-series.
    df["lag_1h"]   = df["demand_mwh"].shift(1)    # 1 hour ago
    df["lag_24h"]  = df["demand_mwh"].shift(24)   # same hour yesterday
    df["lag_168h"] = df["demand_mwh"].shift(168)  # same hour last week

    # ── Rolling statistics ──
    # Captures recent trend / volatility
    df["rolling_mean_24h"] = df["demand_mwh"].shift(1).rolling(24).mean()
    df["rolling_std_24h"]  = df["demand_mwh"].shift(1).rolling(24).std()
    df["rolling_mean_7d"]  = df["demand_mwh"].shift(1).rolling(168).mean()

    # Drop rows where lags are NaN (first 168 hours)
    df = df.dropna().reset_index(drop=True)

    return df


FEATURE_COLS = [
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "day_of_week", "is_weekend", "is_business_hours",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_24h", "rolling_std_24h", "rolling_mean_7d",
]
TARGET_COL = "demand_mwh"


# ─── MODEL TRAINING ───────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame) -> dict:
    """
    Train a forecasting model and return evaluation metrics.

    We use an 80/20 train/test split — training on older data,
    evaluating on the most recent 20%. This simulates how you'd
    actually deploy a model: you can't train on the future.

    Args:
        df: Featured DataFrame (output of build_features)

    Returns:
        dict with model, scaler, and evaluation metrics
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Chronological split — critical for time-series
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"[MODEL] Training on {len(X_train)} hours, testing on {len(X_test)} hours")

    # Scale features (important for linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Try multiple models, pick the best
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        ),
    }

    results = {}
    best_model = None
    best_mae = float("inf")

    for name, model in models.items():
        if name == "Ridge Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            # Tree models don't need scaling
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100  # % error

        results[name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "model": model}
        print(f"  {name:25s} | MAE: {mae:6.1f} MWh | RMSE: {rmse:6.1f} | MAPE: {mape:.2f}%")

        if mae < best_mae:
            best_mae = mae
            best_model = (name, model)

    print(f"\n[MODEL] Best model: {best_model[0]}")

    # Save model and scaler
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"name": best_model[0], "model": best_model[1]}, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[MODEL] Saved to {MODEL_PATH} and {SCALER_PATH}")

    return {
        "best_model_name": best_model[0],
        "best_model": best_model[1],
        "scaler": scaler,
        "metrics": results,
        "test_actuals": y_test,
        "test_preds": results[best_model[0]]["model"].predict(
            X_test_scaled if best_model[0] == "Ridge Regression" else X_test
        ),
        "test_timestamps": df["timestamp"].iloc[split_idx:].values,
        "feature_cols": FEATURE_COLS,
    }


# ─── FORECASTING ──────────────────────────────────────────────────────────────
def forecast_next_24h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a 24-hour ahead forecast using the saved model.

    This is the "live" use case: you have data up to now,
    and you want to predict what demand will be for the next day.

    Args:
        df: Full historical DataFrame (needs at least 168 hours)

    Returns:
        DataFrame with forecasted timestamps and demand values
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No trained model found. Run train_model() first.")

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = model_data["model"]
    model_name = model_data["name"]

    # Build features on historical data
    featured = build_features(df)

    # Iteratively predict the next 24 hours
    last_ts = pd.to_datetime(featured["timestamp"].iloc[-1])
    history = featured["demand_mwh"].tolist()
    forecasts = []

    for h in range(1, 25):
        next_ts = last_ts + timedelta(hours=h)

        # Build a single-row feature vector for this future hour
        row = {
            "hour":              next_ts.hour,
            "day_of_week":       next_ts.dayofweek,
            "month":             next_ts.month,
            "is_weekend":        int(next_ts.dayofweek >= 5),
            "is_business_hours": int(8 <= next_ts.hour <= 18),
            "hour_sin":          np.sin(2 * np.pi * next_ts.hour / 24),
            "hour_cos":          np.cos(2 * np.pi * next_ts.hour / 24),
            "month_sin":         np.sin(2 * np.pi * next_ts.month / 12),
            "month_cos":         np.cos(2 * np.pi * next_ts.month / 12),
            "lag_1h":            history[-1],
            "lag_24h":           history[-24] if len(history) >= 24 else history[-1],
            "lag_168h":          history[-168] if len(history) >= 168 else history[-1],
            "rolling_mean_24h":  np.mean(history[-24:]),
            "rolling_std_24h":   np.std(history[-24:]),
            "rolling_mean_7d":   np.mean(history[-168:]) if len(history) >= 168 else np.mean(history),
        }

        X_row = np.array([[row[f] for f in FEATURE_COLS]])

        if model_name == "Ridge Regression":
            X_row = scaler.transform(X_row)

        pred = model.predict(X_row)[0]
        history.append(pred)
        forecasts.append({"timestamp": next_ts, "forecast_mwh": round(pred, 1)})

    return pd.DataFrame(forecasts)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PJM/EIA Energy Dashboard — Model Trainer")
    print("=" * 60)

    # Load data from DB
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT datetime_utc AS timestamp, total_lmp_rt AS demand_mwh FROM lmp_data ORDER BY datetime_utc ASC",
        conn
    )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"[DATA] Loaded {len(df)} records from database")
    print(f"       Range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    if len(df) < 200:
        print("[!] Not enough data to train. Run fetch_data.py first.")
        return

    # Build features
    print("\n[FEATURES] Engineering features...")
    featured = build_features(df)
    print(f"  Feature matrix shape: {featured[FEATURE_COLS].shape}")
    print(f"  Features: {FEATURE_COLS}")

    # Train
    print("\n[TRAINING] Comparing models...")
    results = train_model(featured)

    # Sample forecast
    print("\n[FORECAST] Next 24 hours:")
    forecast = forecast_next_24h(df)
    for _, row in forecast.iterrows():
        print(f"  {row['timestamp'].strftime('%a %b %d %I%p'):20s} → {row['forecast_mwh']:,.0f} MWh")

    print("\nDone. Next step: run  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
