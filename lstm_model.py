"""
LSTM demand forecasting (deep-learning alternative to the XGBoost model).

This module is optional: if PyTorch is not installed, `is_available()`
returns False and the orchestrator/API transparently falls back to the
XGBoost model. This keeps the project runnable on lightweight machines
while still demonstrating an LSTM forecasting component for the
portfolio/demo.
"""
import os
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LEN = 21  # days of history used to predict the next day


def is_available() -> bool:
    return TORCH_AVAILABLE


if TORCH_AVAILABLE:

    class DemandLSTM(nn.Module):
        def __init__(self, hidden_size=32, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                                 num_layers=num_layers, batch_first=True,
                                 dropout=0.1)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    def _model_path(product_id: int) -> str:
        return os.path.join(MODEL_DIR, f"lstm_product_{product_id}.pt")

    def _make_sequences(series: np.ndarray, seq_len: int):
        X, y = [], []
        for i in range(len(series) - seq_len):
            X.append(series[i:i + seq_len])
            y.append(series[i + seq_len])
        return np.array(X), np.array(y)

    def _normalize(series: np.ndarray):
        mean, std = series.mean(), series.std() + 1e-6
        return (series - mean) / std, mean, std

    def train(product_id: int, history_df: pd.DataFrame, epochs: int = 60) -> dict:
        series = history_df.sort_values("date")["units_sold"].values.astype(np.float32)
        if len(series) < SEQ_LEN + 10:
            raise ValueError("Not enough history to train the LSTM (need 30+ days).")

        norm_series, mean, std = _normalize(series)
        X, y = _make_sequences(norm_series, SEQ_LEN)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        model = DemandLSTM()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        torch.save({
            "state_dict": model.state_dict(),
            "mean": float(mean),
            "std": float(std),
        }, _model_path(product_id))
        return {"final_loss": float(loss.item()), "n_samples": len(X)}

    def forecast(product_id: int, history_df: pd.DataFrame, horizon_days: int = 30) -> dict:
        path = _model_path(product_id)
        if not os.path.exists(path):
            train(product_id, history_df)
        checkpoint = torch.load(path, weights_only=False)
        mean, std = checkpoint["mean"], checkpoint["std"]

        model = DemandLSTM()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        series = history_df.sort_values("date")["units_sold"].values.astype(np.float32)
        window = list((series[-SEQ_LEN:] - mean) / (std + 1e-6))

        last_date = pd.to_datetime(history_df["date"]).max()
        predictions = []
        with torch.no_grad():
            for step in range(1, horizon_days + 1):
                x = torch.tensor(window[-SEQ_LEN:], dtype=torch.float32).view(1, SEQ_LEN, 1)
                pred_norm = model(x).item()
                pred = max(0.0, pred_norm * std + mean)
                next_date = last_date + pd.Timedelta(days=step)
                predictions.append({"date": next_date.date().isoformat(),
                                     "predicted_units": round(pred, 1)})
                window.append(pred_norm)

        total = sum(p["predicted_units"] for p in predictions)
        return {
            "model_used": "lstm",
            "horizon_days": horizon_days,
            "daily_predictions": predictions,
            "predicted_daily_demand": round(total / horizon_days, 2),
            "predicted_total_demand": round(total, 2),
        }

else:
    def train(*args, **kwargs):
        raise RuntimeError("PyTorch is not installed — LSTM forecasting is unavailable. "
                            "Install it with `pip install torch`, or use the XGBoost model.")

    def forecast(*args, **kwargs):
        raise RuntimeError("PyTorch is not installed — LSTM forecasting is unavailable. "
                            "Install it with `pip install torch`, or use the XGBoost model.")
