"""
Inventory analysis.

Turns (current stock + forecasted demand + supplier lead time) into:
    - days of cover remaining
    - stockout risk flag
    - a statistically-grounded reorder quantity recommendation
      (reorder point + safety stock model, using demand variability
      and lead-time to size the safety buffer — the standard formula
      used in real inventory systems)
"""
import math
from dataclasses import dataclass


Z_SCORE_95 = 1.65  # service-level constant for ~95% in-stock probability


@dataclass
class InventoryAnalysis:
    product_id: int
    on_hand_qty: int
    predicted_daily_demand: float
    days_of_cover: float
    stockout_risk: str          # "low" | "medium" | "high"
    recommended_safety_stock: int
    recommended_reorder_point: int
    recommended_order_qty: int


def analyze(
    on_hand_qty: int,
    predicted_daily_demand: float,
    demand_std_dev: float,
    avg_lead_time_days: int,
    product_id: int,
) -> InventoryAnalysis:
    predicted_daily_demand = max(predicted_daily_demand, 0.01)
    days_of_cover = on_hand_qty / predicted_daily_demand

    # Safety stock sized to absorb demand variability during the lead time window
    safety_stock = int(math.ceil(
        Z_SCORE_95 * demand_std_dev * math.sqrt(max(avg_lead_time_days, 1))
    ))
    reorder_point = int(math.ceil(predicted_daily_demand * avg_lead_time_days + safety_stock))

    if days_of_cover < avg_lead_time_days * 0.75:
        risk = "high"
    elif days_of_cover < avg_lead_time_days * 1.5:
        risk = "medium"
    else:
        risk = "low"

    # order enough to cover a 30-day horizon plus safety stock, minus what's on hand
    target_stock = int(predicted_daily_demand * 30 + safety_stock)
    order_qty = max(0, target_stock - on_hand_qty)

    return InventoryAnalysis(
        product_id=product_id,
        on_hand_qty=on_hand_qty,
        predicted_daily_demand=round(predicted_daily_demand, 2),
        days_of_cover=round(days_of_cover, 1),
        stockout_risk=risk,
        recommended_safety_stock=safety_stock,
        recommended_reorder_point=reorder_point,
        recommended_order_qty=order_qty,
    )
