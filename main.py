
import datetime as dt
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import pandas as pd

from . import models, schemas
from .database import get_db, init_db
from .forecasting import xgboost_model, lstm_model
from .inventory import analyzer as inventory_analyzer
from .supplier_rag.retriever import SupplierRetriever
from .risk import risk_model
from .agent import orchestrator

app = FastAPI(
    title="AI Supply Chain & Procurement Intelligence Agent",
    description="Forecasting -> Inventory Analysis -> Supplier RAG -> Risk Prediction "
                "-> Procurement Recommendation -> Approval",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------------------------------
# Products & Inventory
# ---------------------------------------------------------------------------

@app.get("/api/products", response_model=list[schemas.ProductOut])
def list_products(db: Session = Depends(get_db)):
    return db.query(models.Product).all()


@app.get("/api/inventory", response_model=list[schemas.InventoryOut])
def list_inventory(db: Session = Depends(get_db)):
    return db.query(models.InventoryRecord).all()


@app.get("/api/inventory/{product_id}/analysis")
def get_inventory_analysis(product_id: int, db: Session = Depends(get_db)):
    inv = db.query(models.InventoryRecord).filter_by(product_id=product_id).first()
    product = db.query(models.Product).get(product_id)
    if inv is None or product is None:
        raise HTTPException(404, "Product/inventory not found")

    history = (
        db.query(models.DemandHistory)
        .filter_by(product_id=product_id)
        .order_by(models.DemandHistory.date.desc())
        .limit(60)
        .all()
    )
    demand_std = pd.Series([h.units_sold for h in history]).std() if history else 0.0
    avg_daily = pd.Series([h.units_sold for h in history]).mean() if history else 0.0

    links = db.query(models.SupplierProduct).filter_by(product_id=product_id).all()
    avg_lead_time = int(sum(l.lead_time_days for l in links) / len(links)) if links else 7

    result = inventory_analyzer.analyze(
        on_hand_qty=inv.on_hand_qty,
        predicted_daily_demand=float(avg_daily),
        demand_std_dev=float(demand_std) if demand_std == demand_std else 0.0,  # NaN guard
        avg_lead_time_days=avg_lead_time,
        product_id=product_id,
    )
    return result.__dict__


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

@app.post("/api/forecast", response_model=None)
def run_forecast(req: schemas.ForecastRequest, db: Session = Depends(get_db)):
    history_rows = (
        db.query(models.DemandHistory)
        .filter_by(product_id=req.product_id)
        .order_by(models.DemandHistory.date)
        .all()
    )
    if not history_rows:
        raise HTTPException(404, "No demand history for this product")
    history_df = pd.DataFrame([{"date": r.date, "units_sold": r.units_sold} for r in history_rows])

    if req.model_type == "lstm":
        if not lstm_model.is_available():
            raise HTTPException(400, "PyTorch is not installed — LSTM forecasting unavailable. "
                                      "Run `pip install torch` or use model_type='xgboost'.")
        result = lstm_model.forecast(req.product_id, history_df, req.horizon_days)
    else:
        result = xgboost_model.forecast(req.product_id, history_df, req.horizon_days)

    forecast_row = models.DemandForecast(
        product_id=req.product_id,
        horizon_days=req.horizon_days,
        predicted_daily_demand=result["predicted_daily_demand"],
        predicted_total_demand=result["predicted_total_demand"],
        model_used=result["model_used"],
    )
    db.add(forecast_row)
    db.commit()
    result["forecast_id"] = forecast_row.id
    return result


@app.get("/api/forecast/history/{product_id}", response_model=list[schemas.ForecastOut])
def forecast_history(product_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.DemandForecast)
        .filter_by(product_id=product_id)
        .order_by(models.DemandForecast.created_at.desc())
        .limit(10)
        .all()
    )


@app.get("/api/demand-history/{product_id}")
def get_demand_history(product_id: int, days: int = 90, db: Session = Depends(get_db)):
    rows = (
        db.query(models.DemandHistory)
        .filter_by(product_id=product_id)
        .order_by(models.DemandHistory.date.desc())
        .limit(days)
        .all()
    )
    return [{"date": r.date.isoformat(), "units_sold": r.units_sold} for r in reversed(rows)]


# ---------------------------------------------------------------------------
# Suppliers & RAG search
# ---------------------------------------------------------------------------

@app.get("/api/suppliers", response_model=list[schemas.SupplierOut])
def list_suppliers(db: Session = Depends(get_db)):
    return db.query(models.Supplier).all()


@app.get("/api/suppliers/search")
def search_suppliers(query: str, product_id: Optional[int] = None,
                      top_k: int = 5, db: Session = Depends(get_db)):
    retriever = SupplierRetriever().index(db.query(models.Supplier).all())
    candidate_ids = None
    if product_id is not None:
        links = db.query(models.SupplierProduct).filter_by(product_id=product_id).all()
        candidate_ids = [l.supplier_id for l in links]
    results = retriever.search(query, top_k=top_k, candidate_ids=candidate_ids)
    return [
        {"supplier": schemas.SupplierOut.model_validate(s).model_dump(),
         "relevance_score": round(float(score), 4)}
        for s, score in results
    ]


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

@app.get("/api/risk/{supplier_id}")
def get_supplier_risk(supplier_id: int, db: Session = Depends(get_db)):
    try:
        result = risk_model.predict(db, supplier_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    row = models.RiskAssessment(
        supplier_id=supplier_id, risk_score=result["risk_score"],
        risk_label=result["risk_label"], top_factors=result["top_factors"],
    )
    db.add(row)
    db.commit()
    return result


@app.get("/api/risk", response_model=list[schemas.RiskOut])
def list_all_risk(db: Session = Depends(get_db)):
    """Latest risk assessment per supplier (one row each, most recent first)."""
    subq = (
        db.query(models.RiskAssessment)
        .order_by(models.RiskAssessment.supplier_id, models.RiskAssessment.created_at.desc())
        .all()
    )
    seen, latest = set(), []
    for r in subq:
        if r.supplier_id not in seen:
            seen.add(r.supplier_id)
            latest.append(r)
    return latest


# ---------------------------------------------------------------------------
# Agent pipeline & Recommendations (Approval workflow)
# ---------------------------------------------------------------------------

@app.post("/api/pipeline/run", response_model=schemas.RecommendationOut)
def run_pipeline(req: schemas.PipelineRunRequest, db: Session = Depends(get_db)):
    try:
        rec = orchestrator.run_pipeline(
            db, req.product_id, req.horizon_days, req.forecast_model
        )
        return rec
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/recommendations", response_model=list[schemas.RecommendationOut])
def list_recommendations(status: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(models.Recommendation)
    if status:
        q = q.filter_by(status=status)
    return q.order_by(models.Recommendation.created_at.desc()).all()


@app.post("/api/recommendations/{rec_id}/decision", response_model=schemas.RecommendationOut)
def decide_recommendation(rec_id: int, decision: schemas.ApprovalRequest,
                           db: Session = Depends(get_db)):
    rec = db.query(models.Recommendation).get(rec_id)
    if rec is None:
        raise HTTPException(404, "Recommendation not found")
    rec.status = "approved" if decision.approve else "rejected"
    rec.decided_at = dt.datetime.utcnow()
    if decision.note:
        rec.reasoning += f"\n\n[Human note: {decision.note}]"
    db.commit()
    db.refresh(rec)
    return rec


@app.get("/api/health")
def health():
    return {"status": "ok", "lstm_available": lstm_model.is_available()}


# Serve the frontend dashboard directly from FastAPI for convenience:
#   just run uvicorn and open http://localhost:8000/
try:
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
except RuntimeError:
    pass  # frontend directory not found when running from a different cwd — that's fine
