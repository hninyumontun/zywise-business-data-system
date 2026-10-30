from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "recommendation_model.joblib"
FEATURE_COLUMNS = ["quantity", "unit_price", "revenue"]


def recommendation_model_available() -> bool:
    """Return whether the trained recommendation model is available."""
    return MODEL_PATH.exists()


def generate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Generate model-backed recommendations with a transparent fallback."""
    required = {"product", "quantity", "revenue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Recommendation requires columns: {', '.join(sorted(missing))}")

    grouped = (
        df.groupby(["product", "category"], dropna=False)
        .agg(quantity=("quantity", "sum"), revenue=("revenue", "sum"), orders=("product", "size"))
        .reset_index()
    )
    grouped["unit_price"] = grouped["revenue"] / grouped["quantity"].replace(0, pd.NA)
    grouped["unit_price"] = grouped["unit_price"].fillna(0)

    model_used = False
    confidence = pd.Series([pd.NA] * len(grouped), dtype="Float64")
    if MODEL_PATH.exists():
        payload = joblib.load(MODEL_PATH)
        model = payload["model"]
        features = payload.get("features", FEATURE_COLUMNS)
        predictions = model.predict(grouped[features])
        grouped["recommendation"] = predictions
        if hasattr(model, "predict_proba"):
            confidence = pd.Series(model.predict_proba(grouped[features]).max(axis=1), dtype="Float64")
        model_used = True
    else:
        median_quantity = grouped["quantity"].median()
        median_revenue = grouped["revenue"].median()

        def classify(row: pd.Series) -> str:
            if row["quantity"] >= median_quantity and row["revenue"] < median_revenue:
                return "Increase Stock"
            if row["revenue"] >= median_revenue:
                return "Promote"
            return "Maintain"

        grouped["recommendation"] = grouped.apply(classify, axis=1)

    grouped["model_confidence"] = confidence
    grouped["recommendation_method"] = "Decision-tree model" if model_used else "Transparent rule-based fallback"
    grouped["reason"] = grouped.apply(
        lambda row: {
            "Increase Stock": "High quantity with lower revenue contribution suggests stock attention.",
            "Promote": "High revenue contribution makes this product suitable for promotion.",
            "Maintain": "Current sales contribution does not trigger a stronger action.",
        }[row["recommendation"]],
        axis=1,
    )
    return grouped.sort_values(["recommendation", "revenue"], ascending=[True, False]).reset_index(drop=True)
