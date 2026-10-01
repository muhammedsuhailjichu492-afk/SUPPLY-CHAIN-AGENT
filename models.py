
import datetime as dt
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, Date
)
from sqlalchemy.orm import relationship
from .database import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String(32), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=False)
    category = Column(String(64), nullable=False)
    unit_cost = Column(Float, nullable=False)

    inventory = relationship("InventoryRecord", back_populates="product", uselist=False)
    demand_history = relationship("DemandHistory", back_populates="product")
    forecasts = relationship("DemandForecast", back_populates="product")


class DemandHistory(Base):
    __tablename__ = "demand_history"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    date = Column(Date, nullable=False)
    units_sold = Column(Integer, nullable=False)

    product = relationship("Product", back_populates="demand_history")


class InventoryRecord(Base):
    __tablename__ = "inventory_records"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), unique=True, nullable=False)
    on_hand_qty = Column(Integer, nullable=False)
    reorder_point = Column(Integer, default=0)
    safety_stock = Column(Integer, default=0)
    warehouse = Column(String(64), default="Main Warehouse")
    updated_at = Column(DateTime, default=dt.datetime.utcnow)

    product = relationship("Product", back_populates="inventory")


class Supplier(Base):
    __tablename__ = "suppliers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    region = Column(String(64), nullable=False)
    reliability_score = Column(Float, default=0.8)   # 0-1, on-time delivery rate
    quality_score = Column(Float, default=0.8)        # 0-1, defect-free rate
    avg_lead_time_days = Column(Integer, default=7)
    financial_stability = Column(Float, default=0.8)  # 0-1 synthetic credit-health proxy
    profile_text = Column(Text, nullable=False)        # free text used by the RAG retriever

    product_links = relationship("SupplierProduct", back_populates="supplier")
    orders = relationship("Order", back_populates="supplier")
    risk_assessments = relationship("RiskAssessment", back_populates="supplier")


class SupplierProduct(Base):
    __tablename__ = "supplier_products"

    id = Column(Integer, primary_key=True, index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    unit_price = Column(Float, nullable=False)
    lead_time_days = Column(Integer, nullable=False)

    supplier = relationship("Supplier", back_populates="product_links")
    product = relationship("Product")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    order_date = Column(Date, nullable=False)
    promised_date = Column(Date, nullable=False)
    delivered_date = Column(Date, nullable=True)
    quantity = Column(Integer, nullable=False)
    defect_rate = Column(Float, default=0.0)
    was_on_time = Column(Boolean, default=True)

    supplier = relationship("Supplier", back_populates="orders")
    product = relationship("Product")


class DemandForecast(Base):
    __tablename__ = "demand_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    horizon_days = Column(Integer, nullable=False)
    predicted_daily_demand = Column(Float, nullable=False)
    predicted_total_demand = Column(Float, nullable=False)
    model_used = Column(String(32), nullable=False)  # 'xgboost' or 'lstm'
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    product = relationship("Product", back_populates="forecasts")


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=False)
    risk_score = Column(Float, nullable=False)     # 0 (low) - 1 (high)
    risk_label = Column(String(16), nullable=False)  # low / medium / high
    top_factors = Column(Text, nullable=True)      # JSON-encoded feature contributions
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    supplier = relationship("Supplier", back_populates="risk_assessments")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    recommended_supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=True)
    recommended_order_qty = Column(Integer, nullable=False)
    reasoning = Column(Text, nullable=False)         # agent-generated explanation
    forecast_id = Column(Integer, ForeignKey("demand_forecasts.id"), nullable=True)
    risk_id = Column(Integer, ForeignKey("risk_assessments.id"), nullable=True)
    status = Column(String(16), default="pending")   # pending / approved / rejected
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    decided_at = Column(DateTime, nullable=True)

    product = relationship("Product")
    recommended_supplier = relationship("Supplier")
