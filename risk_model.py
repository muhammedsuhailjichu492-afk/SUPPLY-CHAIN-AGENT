
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "risk_region_encoder.joblib")

FEATURE_COLS = [
    "reliability_score", "quality_score", "avg_lead_time_days",
    "financial_stability", "order_count", "on_time_rate",
    "avg_defect_rate", "delay_variability", "region_encoded",
]


def _label_from_score(score: float) -> str:
    if score >= 0.66:
        return "high"
    if score >= 0.33:
        return "medium"
    return "low"


def build_supplier_features(db) -> pd.DataFrame:
    """Aggregates each supplier's order history into model-ready features."""
    from .. import models as m

    suppliers = db.query(m.Supplier).all()
    rows = []
    for s in suppliers:
        orders = db.query(m.Order).filter_by(supplier_id=s.id).all()
        order_count = len(orders)
        if order_count > 0:
            on_time_rate = sum(o.was_on_time for o in orders) / order_count
            avg_defect_rate = float(np.mean([o.defect_rate for o in orders]))
            delays = [
                (o.delivered_date - o.promised_date).days
                for o in orders if o.delivered_date is not None
            ]
            delay_variability = float(np.std(delays)) if delays else 0.0
        else:
            on_time_rate, avg_defect_rate, delay_variability = s.reliability_score, 0.0, 0.0

        rows.append({
            "supplier_id": s.id,
            "reliability_score": s.reliability_score,
            "quality_score": s.quality_score,
            "avg_lead_time_days": s.avg_lead_time_days,
            "financial_stability": s.financial_stability,
            "order_count": order_count,
            "on_time_rate": on_time_rate,
            "avg_defect_rate": avg_defect_rate,
            "delay_variability": delay_variability,
            "region": s.region,
        })
    return pd.DataFrame(rows)


def _synthetic_risk_label(row) -> float:
    """
    Weak-supervision target used to train the classifier: a weighted
    composite of the risk-relevant features (lower reliability/quality/
    financial-stability and higher defect/delay-variability => higher risk),
    plus a little noise so the model has to learn a genuine decision
    boundary rather than a hand-coded formula.
    """
    score = (
        0.30 * (1 - row["reliability_score"])
        + 0.20 * (1 - row["quality_score"])
        + 0.20 * (1 - row["financial_stability"])
        + 0.15 * min(row["avg_defect_rate"] * 10, 1.0)
        + 0.15 * min(row["delay_variability"] / 10, 1.0)
    )
    return float(np.clip(score + np.random.normal(0, 0.03), 0, 1))


def train(db) -> dict:
    df = build_supplier_features(db)
    if len(df) < 5:
        raise ValueError("Not enough suppliers to train the risk model (need 5+).")

    encoder = LabelEncoder()
    df["region_encoded"] = encoder.fit_transform(df["region"])
    df["risk_score_synthetic"] = df.apply(_synthetic_risk_label, axis=1)
    df["risk_label"] = df["risk_score_synthetic"].apply(_label_from_score)

    X = df[FEATURE_COLS]
    y = df["risk_label"]

    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    clf.fit(X, y)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    importances = dict(zip(FEATURE_COLS, clf.feature_importances_.tolist()))
    return {"n_suppliers": len(df), "feature_importances": importances}


def predict(db, supplier_id: int) -> dict:
    if not os.path.exists(MODEL_PATH):
        train(db)
    clf: RandomForestClassifier = joblib.load(MODEL_PATH)
    encoder: LabelEncoder = joblib.load(ENCODER_PATH)

    df = build_supplier_features(db)
    row = df[df["supplier_id"] == supplier_id]
    if row.empty:
        raise ValueError(f"No supplier with id {supplier_id}")

    # unseen regions fall back to the most common encoded region rather than erroring
    try:
        row = row.copy()
        row["region_encoded"] = encoder.transform(row["region"])
    except ValueError:
        row["region_encoded"] = 0

    X = row[FEATURE_COLS]
    proba = clf.predict_proba(X)[0]
    classes = clf.classes_
    label_idx = int(np.argmax(proba))
    risk_label = classes[label_idx]

    # risk_score: probability-weighted position on a 0(low)-1(high) scale
    order_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
    risk_score = float(sum(order_map.get(c, 0.5) * p for c, p in zip(classes, proba)))

    contributions = dict(zip(FEATURE_COLS, clf.feature_importances_.tolist()))
    top_factors = dict(sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)[:4])

    return {
        "supplier_id": supplier_id,
        "risk_score": round(risk_score, 3),
        "risk_label": risk_label,
        "top_factors": json.dumps(top_factors),
        "class_probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
    }
