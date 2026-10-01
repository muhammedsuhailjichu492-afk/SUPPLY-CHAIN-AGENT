"""
Agentic orchestrator — the "brain" that chains every module together into
one autonomous decision, matching the pipeline in the project spec:

    Forecasting -> Inventory Analysis -> Supplier RAG -> Risk Prediction
        -> Procurement Recommendation -> Approval

Each stage's output feeds the next. The final recommendation is persisted
to the database with status="pending" and requires a human approval call
(agentic-but-supervised, appropriate for real procurement decisions).
"""
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .. import models
from ..forecasting import xgboost_model, lstm_model
from ..inventory import analyzer as inventory_analyzer
from ..supplier_rag.retriever import SupplierRetriever
from ..risk import risk_model
from . import llm_client


def _history_df(db: Session, product_id: int) -> pd.DataFrame:
    rows = (
        db.query(models.DemandHistory)
        .filter_by(product_id=product_id)
        .order_by(models.DemandHistory.date)
        .all()
    )
    return pd.DataFrame([{"date": r.date, "units_sold": r.units_sold} for r in rows])


def run_pipeline(db: Session, product_id: int, horizon_days: int = 30,
                  forecast_model: str = "xgboost") -> models.Recommendation:
    product = db.query(models.Product).get(product_id)
    if product is None:
        raise ValueError(f"No product with id {product_id}")

    # ---- Stage 1: Forecasting -------------------------------------------
    history = _history_df(db, product_id)
    if forecast_model == "lstm" and lstm_model.is_available():
        forecast_result = lstm_model.forecast(product_id, history, horizon_days)
    else:
        forecast_result = xgboost_model.forecast(product_id, history, horizon_days)

    forecast_row = models.DemandForecast(
        product_id=product_id,
        horizon_days=horizon_days,
        predicted_daily_demand=forecast_result["predicted_daily_demand"],
        predicted_total_demand=forecast_result["predicted_total_demand"],
        model_used=forecast_result["model_used"],
    )
    db.add(forecast_row)
    db.commit()
    db.refresh(forecast_row)

    # ---- Stage 2: Inventory analysis --------------------------------------
    inv = db.query(models.InventoryRecord).filter_by(product_id=product_id).first()
    demand_std = float(history["units_sold"].tail(60).std()) if len(history) >= 2 else 0.0
    supplier_links = db.query(models.SupplierProduct).filter_by(product_id=product_id).all()
    avg_lead_time = (
        int(np.mean([l.lead_time_days for l in supplier_links])) if supplier_links else 7
    )

    inv_analysis = inventory_analyzer.analyze(
        on_hand_qty=inv.on_hand_qty if inv else 0,
        predicted_daily_demand=forecast_result["predicted_daily_demand"],
        demand_std_dev=demand_std,
        avg_lead_time_days=avg_lead_time,
        product_id=product_id,
    )

    # ---- Stage 3: Supplier RAG retrieval -----------------------------------
    all_suppliers = [l.supplier for l in supplier_links] or db.query(models.Supplier).limit(5).all()
    retriever = SupplierRetriever().index(db.query(models.Supplier).all())
    query = (
        f"reliable low-risk supplier for {product.category} with fast lead time "
        f"and strong quality, able to supply {product.name}"
    )
    candidate_ids = [s.id for s in all_suppliers]
    ranked = retriever.search(query, top_k=3, candidate_ids=candidate_ids)
    if not ranked and all_suppliers:
        ranked = [(all_suppliers[0], 0.0)]
    top_supplier, relevance = ranked[0] if ranked else (None, 0.0)

    # ---- Stage 4: Risk prediction ------------------------------------------
    risk_row = None
    if top_supplier is not None:
        risk_result = risk_model.predict(db, top_supplier.id)
        risk_row = models.RiskAssessment(
            supplier_id=top_supplier.id,
            risk_score=risk_result["risk_score"],
            risk_label=risk_result["risk_label"],
            top_factors=risk_result["top_factors"],
        )
        db.add(risk_row)
        db.commit()
        db.refresh(risk_row)

        # if the top-ranked supplier is high risk and an alternative exists, prefer it
        if risk_result["risk_label"] == "high" and len(ranked) > 1:
            for alt_supplier, alt_relevance in ranked[1:]:
                alt_risk = risk_model.predict(db, alt_supplier.id)
                if alt_risk["risk_label"] != "high":
                    top_supplier, relevance = alt_supplier, alt_relevance
                    risk_row = models.RiskAssessment(
                        supplier_id=top_supplier.id,
                        risk_score=alt_risk["risk_score"],
                        risk_label=alt_risk["risk_label"],
                        top_factors=alt_risk["top_factors"],
                    )
                    db.add(risk_row)
                    db.commit()
                    db.refresh(risk_row)
                    break

    # ---- Stage 5: Procurement recommendation (agentic reasoning) ----------
    link = next((l for l in supplier_links if top_supplier and l.supplier_id == top_supplier.id), None)
    unit_price = link.unit_price if link else (top_supplier and product.unit_cost)
    lead_time_days = link.lead_time_days if link else avg_lead_time

    reasoning = llm_client.explain_recommendation({
        "product_name": product.name, "sku": product.sku,
        "predicted_daily_demand": forecast_result["predicted_daily_demand"],
        "predicted_total_demand": forecast_result["predicted_total_demand"],
        "horizon_days": horizon_days, "forecast_model": forecast_result["model_used"],
        "on_hand_qty": inv_analysis.on_hand_qty, "days_of_cover": inv_analysis.days_of_cover,
        "stockout_risk": inv_analysis.stockout_risk,
        "recommended_order_qty": inv_analysis.recommended_order_qty,
        "supplier_name": top_supplier.name if top_supplier else "N/A",
        "supplier_region": top_supplier.region if top_supplier else "N/A",
        "risk_label": risk_row.risk_label if risk_row else "unknown",
        "reliability_score": top_supplier.reliability_score if top_supplier else 0,
        "quality_score": top_supplier.quality_score if top_supplier else 0,
        "lead_time_days": lead_time_days or 0,
        "unit_price": unit_price or 0,
    })

    # ---- Stage 6: Persist as a pending recommendation (awaits approval) ---
    recommendation = models.Recommendation(
        product_id=product_id,
        recommended_supplier_id=top_supplier.id if top_supplier else None,
        recommended_order_qty=inv_analysis.recommended_order_qty,
        reasoning=reasoning,
        forecast_id=forecast_row.id,
        risk_id=risk_row.id if risk_row else None,
        status="pending",
    )
    db.add(recommendation)
    db.commit()
    db.refresh(recommendation)
    return recommendation
