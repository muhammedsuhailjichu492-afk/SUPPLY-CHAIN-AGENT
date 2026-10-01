"""
One-shot setup script: generates synthetic data and pretrains every model
so the dashboard is instantly responsive on first use.

Usage:
    python train_models.py
"""
from backend.database import SessionLocal
from backend import data_generator, models
from backend.forecasting import xgboost_model, lstm_model
from backend.risk import risk_model


def main():
    print("=" * 60)
    print("STEP 1/3 — Generating synthetic supply-chain data")
    print("=" * 60)
    data_generator.generate()

    db = SessionLocal()

    print("\n" + "=" * 60)
    print("STEP 2/3 — Training demand forecasting models (XGBoost)")
    print("=" * 60)
    products = db.query(models.Product).all()
    for p in products:
        history = (
            db.query(models.DemandHistory)
            .filter_by(product_id=p.id)
            .order_by(models.DemandHistory.date)
            .all()
        )
        import pandas as pd
        df = pd.DataFrame([{"date": h.date, "units_sold": h.units_sold} for h in history])
        metrics = xgboost_model.train(p.id, df)
        print(f"  {p.sku:10s} {p.name:28s} val_MAE={metrics['mae']:.2f}" if metrics['mae'] else
              f"  {p.sku:10s} {p.name:28s} trained")

    if lstm_model.is_available():
        print("\n(PyTorch detected — LSTM models will train lazily on first request.)")
    else:
        print("\n(PyTorch not installed — skipping LSTM; XGBoost forecasting is fully functional.)")

    print("\n" + "=" * 60)
    print("STEP 3/3 — Training supplier risk model")
    print("=" * 60)
    metrics = risk_model.train(db)
    print(f"  Trained on {metrics['n_suppliers']} suppliers")
    print("  Top risk factors:",
          sorted(metrics["feature_importances"].items(), key=lambda kv: kv[1], reverse=True)[:3])

    db.close()
    print("\nAll done! Start the API with:")
    print("    uvicorn backend.main:app --reload --port 8000")
    print("Then open http://localhost:8000/ in your browser.")


if __name__ == "__main__":
    main()
