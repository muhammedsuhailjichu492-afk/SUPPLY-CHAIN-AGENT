"""
XGBoost demand forecasting.

Approach: supervised regression on engineered time-series features
(lags, rolling means, day-of-week, day-of-year) rather than a plain
autoregressive walk — this is the standard, reliable way to get XGBoost
to forecast a time series, and it degrades gracefully for
new/low-history products.
"""
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LAGS = [1, 2, 3, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 30]


def _model_path(product_id: int) -> str:
    return os.path.join(MODEL_DIR, f"xgb_product_{product_id}.joblib")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns ['date', 'units_sold'] sorted ascending by date."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for lag in LAGS:
        df[f"lag_{lag}"] = df["units_sold"].shift(lag)
    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}"] = df["units_sold"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df["units_sold"].shift(1).rolling(w).std()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


FEATURE_COLS = (
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
    + ["day_of_week", "day_of_year", "is_weekend"]
)


def train(product_id: int, history_df: pd.DataFrame) -> dict:
    """Trains and persists an XGBoost model for one product. Returns train metrics."""
    feats = build_features(history_df).dropna().reset_index(drop=True)
    if len(feats) < 30:
        raise ValueError("Not enough history to train a reliable model (need 30+ days).")

    X = feats[FEATURE_COLS]
    y = feats["units_sold"]

    # last 14 days held out to report a rough validation MAE
    split = max(1, len(X) - 14)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    mae = None
    if len(X_val) > 0:
        preds = model.predict(X_val)
        mae = float(np.mean(np.abs(preds - y_val.values)))

    joblib.dump(model, _model_path(product_id))
    return {"mae": mae, "n_samples": len(X)}


def forecast(product_id: int, history_df: pd.DataFrame, horizon_days: int = 30) -> dict:
    """
    Iteratively forecasts `horizon_days` ahead using the trained model,
    feeding each prediction back in as a new lag value (recursive forecasting).
    """
    path = _model_path(product_id)
    if not os.path.exists(path):
        train(product_id, history_df)
    model: XGBRegressor = joblib.load(path)

    working = history_df.copy()
    working["date"] = pd.to_datetime(working["date"])
    working = working.sort_values("date").reset_index(drop=True)

    last_date = working["date"].max()
    predictions = []

    for step in range(1, horizon_days + 1):
        next_date = last_date + pd.Timedelta(days=step)
        temp = pd.concat([
            working[["date", "units_sold"]],
            pd.DataFrame({"date": [next_date], "units_sold": [np.nan]})
        ], ignore_index=True)
        feats = build_features(temp)
        row = feats.iloc[[-1]][FEATURE_COLS]
        pred = float(max(0, model.predict(row)[0]))
        predictions.append({"date": next_date.date().isoformat(), "predicted_units": round(pred, 1)})
        working = pd.concat([
            working, pd.DataFrame({"date": [next_date], "units_sold": [pred]})
        ], ignore_index=True)

    total = sum(p["predicted_units"] for p in predictions)
    return {
        "model_used": "xgboost",
        "horizon_days": horizon_days,
        "daily_predictions": predictions,
        "predicted_daily_demand": round(total / horizon_days, 2),
        "predicted_total_demand": round(total, 2),
    }
