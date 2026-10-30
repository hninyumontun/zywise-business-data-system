from __future__ import annotations

from io import BytesIO
import re
from typing import BinaryIO

import pandas as pd


COLUMN_ALIASES = {
    "order_id": {"order_id", "orderid", "order_number", "transaction_id", "id"},
    "order_date": {"order_date", "date", "transaction_date", "sale_date"},
    "product": {"product", "product_name", "item", "item_name"},
    "category": {"category", "product_category", "department"},
    "quantity": {"quantity", "qty", "units", "units_sold"},
    "unit_price": {"unit_price", "unitprice", "price", "selling_price"},
    "revenue": {"revenue", "sales", "sales_amount", "total", "total_sales"},
}


def read_uploaded_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Read a CSV or Excel upload into a DataFrame."""
    suffix = file_name.lower().rsplit(".", 1)[-1]
    stream = BytesIO(file_bytes)
    if suffix == "csv":
        return pd.read_csv(stream)
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(stream)
    raise ValueError("Only CSV and Excel files are supported.")


def _normalise_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _canonical_columns(columns: list[object]) -> dict[object, str]:
    mapping: dict[object, str] = {}
    used: set[str] = set()
    for original in columns:
        normalised = _normalise_name(original)
        target = normalised
        for canonical, aliases in COLUMN_ALIASES.items():
            if normalised == canonical or normalised in aliases:
                target = canonical
                break
        if target in used:
            target = normalised
        used.add(target)
        mapping[original] = target
    return mapping


def clean_dataframe(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply traceable cleaning rules and return cleaned data plus a summary."""
    if frame.empty:
        raise ValueError("The uploaded file contains no records.")

    df = frame.copy()
    rows_before = len(df)
    issue_rows: list[dict[str, object]] = []

    original_columns = list(df.columns)
    rename_map = _canonical_columns(original_columns)
    renamed_count = sum(old != new for old, new in rename_map.items())
    df = df.rename(columns=rename_map)
    issue_rows.append({
        "Issue type": "Column names",
        "Detected": renamed_count,
        "Handled": renamed_count,
        "Rule applied": "Map names to standard snake_case fields",
    })

    text_handled = 0
    for column in df.select_dtypes(include=["object", "string"]).columns:
        original = df[column].copy()
        cleaned = original.astype("string").str.strip()
        if column in {"product", "category"}:
            cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.title()
        text_handled += int((original.fillna("").astype(str) != cleaned.fillna("").astype(str)).sum())
        df[column] = cleaned
    issue_rows.append({
        "Issue type": "Text values",
        "Detected": text_handled,
        "Handled": text_handled,
        "Rule applied": "Trim spaces and standardise product/category text",
    })

    date_handled = 0
    date_column = next((c for c in ["order_date", "date"] if c in df.columns), None)
    if date_column:
        original = df[date_column].copy()
        parsed = pd.to_datetime(original, errors="coerce", dayfirst=False)
        date_handled = int((original.notna() & parsed.notna()).sum())
        df[date_column] = parsed
    issue_rows.append({
        "Issue type": "Dates",
        "Detected": date_handled,
        "Handled": date_handled,
        "Rule applied": "Parse valid dates into one date type",
    })

    numeric_handled = 0
    for column in ["quantity", "unit_price", "revenue"]:
        if column in df.columns:
            original = df[column].copy()
            converted = pd.to_numeric(original, errors="coerce")
            numeric_handled += int((original.notna() & converted.notna()).sum())
            df[column] = converted
    issue_rows.append({
        "Issue type": "Numeric fields",
        "Detected": numeric_handled,
        "Handled": numeric_handled,
        "Rule applied": "Convert valid numeric values and preserve invalid values as missing",
    })

    missing_before = int(df.isna().sum().sum())
    for column in ["quantity", "unit_price", "revenue"]:
        if column in df.columns:
            df[column] = df[column].fillna(0)
    missing_after = int(df.isna().sum().sum())
    issue_rows.append({
        "Issue type": "Missing values",
        "Detected": missing_before,
        "Handled": missing_before - missing_after,
        "Rule applied": "Fill numeric missing values with zero for calculation or flag for review",
    })

    duplicate_key = "order_id" if "order_id" in df.columns else None
    duplicate_mask = df.duplicated(subset=[duplicate_key] if duplicate_key else None, keep="first")
    duplicate_count = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy()
    issue_rows.append({
        "Issue type": "Duplicate records",
        "Detected": duplicate_count,
        "Handled": duplicate_count,
        "Rule applied": "Keep the first record for each order identifier",
    })

    if date_column:
        df = df.sort_values(date_column, kind="stable")
    df = df.reset_index(drop=True)
    summary = pd.DataFrame(issue_rows)
    summary["Rows before"] = rows_before
    summary["Rows after"] = len(df)
    return df, summary[["Issue type", "Detected", "Handled", "Rule applied", "Rows before", "Rows after"]]

