from __future__ import annotations

import pandas as pd


def calculate_metrics(df: pd.DataFrame) -> dict[str, object]:
    """Calculate dashboard metrics from cleaned business data."""
    if df.empty:
        raise ValueError("No cleaned records are available for analysis.")

    revenue = float(df["revenue"].sum()) if "revenue" in df else 0.0
    orders = int(df["order_id"].nunique()) if "order_id" in df else len(df)
    quantity = float(df["quantity"].sum()) if "quantity" in df else 0.0
    average_order_value = revenue / orders if orders else 0.0

    product_sales = (
        df.groupby("product", dropna=False)["revenue"].sum().sort_values(ascending=False).reset_index()
        if {"product", "revenue"}.issubset(df.columns)
        else pd.DataFrame(columns=["product", "revenue"])
    )
    category_sales = (
        df.groupby("category", dropna=False)["revenue"].sum().sort_values(ascending=False).reset_index()
        if {"category", "revenue"}.issubset(df.columns)
        else pd.DataFrame(columns=["category", "revenue"])
    )

    monthly_sales = pd.DataFrame(columns=["month", "revenue"])
    if {"order_date", "revenue"}.issubset(df.columns):
        monthly_sales = (
            df.assign(month=df["order_date"].dt.to_period("M").astype(str))
            .groupby("month", as_index=False)["revenue"].sum()
            .sort_values("month")
        )

    return {
        "total_revenue": revenue,
        "total_orders": orders,
        "total_quantity": quantity,
        "average_order_value": average_order_value,
        "top_product": product_sales.iloc[0]["product"] if not product_sales.empty else "Unavailable",
        "lowest_product": product_sales.iloc[-1]["product"] if not product_sales.empty else "Unavailable",
        "product_sales": product_sales,
        "category_sales": category_sales,
        "monthly_sales": monthly_sales,
    }

