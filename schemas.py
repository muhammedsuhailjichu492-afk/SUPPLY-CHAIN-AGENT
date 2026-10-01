
import datetime as dt
from typing import Optional, List
from pydantic import BaseModel, ConfigDict


class ProductOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    sku: str
    name: str
    category: str
    unit_cost: float


class InventoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    product_id: int
    on_hand_qty: int
    reorder_point: int
    safety_stock: int
    warehouse: str


class SupplierOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    region: str
    reliability_score: float
    quality_score: float
    avg_lead_time_days: int
    financial_stability: float
    profile_text: str


class SupplierSearchResult(BaseModel):
    supplier: SupplierOut
    relevance_score: float


class ForecastOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    product_id: int
    horizon_days: int
    predicted_daily_demand: float
    predicted_total_demand: float
    model_used: str
    created_at: dt.datetime


class ForecastRequest(BaseModel):
    product_id: int
    horizon_days: int = 30
    model_type: str = "xgboost"  # "xgboost" or "lstm"


class RiskOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    supplier_id: int
    risk_score: float
    risk_label: str
    top_factors: Optional[str] = None
    created_at: dt.datetime


class RecommendationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    product_id: int
    recommended_supplier_id: Optional[int]
    recommended_order_qty: int
    reasoning: str
    status: str
    created_at: dt.datetime


class PipelineRunRequest(BaseModel):
    product_id: int
    horizon_days: int = 30
    forecast_model: str = "xgboost"


class ApprovalRequest(BaseModel):
    approve: bool
    note: Optional[str] = None
