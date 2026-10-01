
import datetime as dt
import random
import numpy as np

from .database import SessionLocal, init_db, engine, Base
from . import models

random.seed(42)
np.random.seed(42)

CATEGORIES = ["Electronics Components", "Packaging Materials", "Raw Metals",
              "Industrial Fasteners", "Safety Equipment"]

PRODUCT_NAMES = [
    ("Microcontroller Unit", "Electronics Components"),
    ("Corrugated Shipping Box", "Packaging Materials"),
    ("Cold-Rolled Steel Sheet", "Raw Metals"),
    ("Hex Bolt M8", "Industrial Fasteners"),
    ("Industrial Safety Gloves", "Safety Equipment"),
    ("LED Display Panel", "Electronics Components"),
    ("Stretch Wrap Film", "Packaging Materials"),
    ("Aluminum Extrusion Bar", "Raw Metals"),
    ("Anti-Vibration Mount", "Industrial Fasteners"),
    ("Respirator Mask N95", "Safety Equipment"),
    ("Power Supply Module", "Electronics Components"),
    ("Pallet Wooden Standard", "Packaging Materials"),
]

REGIONS = ["North America", "East Asia", "South Asia", "Europe", "South America"]

SUPPLIER_ADJECTIVES = ["Precision", "Global", "Reliable", "Summit", "Vertex",
                        "Meridian", "Apex", "Horizon", "Unity", "Pioneer"]
SUPPLIER_NOUNS = ["Manufacturing", "Industries", "Supply Co", "Logistics",
                   "Components", "Materials Group", "Trading", "Works"]


def _random_supplier_profile(name, region, reliability, quality, lead_time, financial):
    """Builds a natural-language profile — this text is what the RAG retriever indexes."""
    certifications = random.sample(
        ["ISO 9001", "ISO 14001", "IATF 16949", "RoHS Compliant", "REACH Compliant"],
        k=random.randint(1, 3),
    )
    strengths = []
    if reliability > 0.85:
        strengths.append("consistently on-time deliveries")
    if quality > 0.85:
        strengths.append("very low defect rates")
    if lead_time < 7:
        strengths.append("fast turnaround / short lead times")
    if financial > 0.85:
        strengths.append("strong financial stability with low bankruptcy risk")
    if not strengths:
        strengths.append("competitive pricing on bulk orders")

    return (
        f"{name} is a supplier based in {region} specializing in industrial "
        f"components. Certifications: {', '.join(certifications)}. "
        f"Known for {', '.join(strengths)}. Average lead time is "
        f"{lead_time} days with a historical on-time delivery rate of "
        f"{reliability*100:.0f}% and a quality pass rate of {quality*100:.0f}%."
    )


def generate(num_days=730):
    print("Resetting database...")
    Base.metadata.drop_all(bind=engine)
    init_db()
    db = SessionLocal()

    # ---- Products -------------------------------------------------------
    products = []
    for i, (pname, category) in enumerate(PRODUCT_NAMES):
        p = models.Product(
            sku=f"SKU-{1000+i}",
            name=pname,
            category=category,
            unit_cost=round(random.uniform(0.5, 250), 2),
        )
        db.add(p)
        products.append(p)
    db.commit()

    # ---- Demand history (trend + weekly seasonality + noise + occasional spikes)
    start_date = dt.date.today() - dt.timedelta(days=num_days)
    for p in products:
        base = random.uniform(20, 200)
        trend = random.uniform(-0.02, 0.05)  # slow drift per day
        for d in range(num_days):
            date = start_date + dt.timedelta(days=d)
            weekday_factor = 1.15 if date.weekday() < 5 else 0.6  # weekday vs weekend
            seasonal = 1 + 0.25 * np.sin(2 * np.pi * d / 365)  # yearly seasonality
            noise = np.random.normal(0, base * 0.12)
            spike = base * random.uniform(1.5, 2.5) if random.random() < 0.01 else 0
            demand = max(0, base * (1 + trend * d / 30) * weekday_factor * seasonal + noise + spike)
            db.add(models.DemandHistory(product_id=p.id, date=date, units_sold=int(demand)))
        db.commit()

    # ---- Inventory snapshot ----------------------------------------------
    for p in products:
        on_hand = random.randint(50, 800)
        db.add(models.InventoryRecord(
            product_id=p.id,
            on_hand_qty=on_hand,
            reorder_point=int(on_hand * random.uniform(0.2, 0.4)),
            safety_stock=int(on_hand * random.uniform(0.1, 0.2)),
            warehouse=random.choice(["Main Warehouse", "Regional DC - East", "Regional DC - West"]),
        ))
    db.commit()

    # ---- Suppliers ---------------------------------------------------------
    suppliers = []
    num_suppliers = 10
    used_names = set()
    for i in range(num_suppliers):
        while True:
            name = f"{random.choice(SUPPLIER_ADJECTIVES)} {random.choice(SUPPLIER_NOUNS)}"
            if name not in used_names:
                used_names.add(name)
                break
        region = random.choice(REGIONS)
        reliability = round(np.clip(np.random.normal(0.85, 0.12), 0.35, 0.99), 2)
        quality = round(np.clip(np.random.normal(0.88, 0.1), 0.4, 0.99), 2)
        lead_time = int(np.clip(np.random.normal(10, 5), 2, 35))
        financial = round(np.clip(np.random.normal(0.8, 0.15), 0.2, 0.99), 2)
        profile = _random_supplier_profile(name, region, reliability, quality, lead_time, financial)
        s = models.Supplier(
            name=name, region=region, reliability_score=reliability,
            quality_score=quality, avg_lead_time_days=lead_time,
            financial_stability=financial, profile_text=profile,
        )
        db.add(s)
        suppliers.append(s)
    db.commit()

    # ---- Supplier <-> product links (2-4 suppliers per product) -----------
    for p in products:
        chosen = random.sample(suppliers, k=random.randint(2, 4))
        for s in chosen:
            price = round(p.unit_cost * random.uniform(0.85, 1.25), 2)
            db.add(models.SupplierProduct(
                supplier_id=s.id, product_id=p.id,
                unit_price=price, lead_time_days=s.avg_lead_time_days + random.randint(-2, 2),
            ))
    db.commit()

    # ---- Historical orders (drives risk model features) -------------------
    for s in suppliers:
        links = db.query(models.SupplierProduct).filter_by(supplier_id=s.id).all()
        for _ in range(random.randint(15, 40)):
            link = random.choice(links)
            order_date = start_date + dt.timedelta(days=random.randint(0, num_days - 30))
            promised = order_date + dt.timedelta(days=link.lead_time_days)
            on_time = random.random() < s.reliability_score
            delay = 0 if on_time else random.randint(1, 12)
            delivered = promised + dt.timedelta(days=delay)
            defect_rate = max(0.0, np.random.normal(1 - s.quality_score, 0.03))
            db.add(models.Order(
                supplier_id=s.id, product_id=link.product_id, order_date=order_date,
                promised_date=promised, delivered_date=delivered,
                quantity=random.randint(50, 1000), defect_rate=round(defect_rate, 4),
                was_on_time=on_time,
            ))
    db.commit()
    db.close()
    print(f"Generated {len(products)} products, {num_suppliers} suppliers, "
          f"{num_days} days of demand history, and historical orders.")


if __name__ == "__main__":
    generate()
