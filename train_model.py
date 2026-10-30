"""Train and evaluate the ZyWise recommendation model.

The included CSV is a small synthetic, labelled demonstration dataset. It is
useful for verifying the training workflow, but it is not evidence of real
business performance or production accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "recommendation_training.csv"
MODEL_PATH = ROOT / "models" / "recommendation_model.joblib"
METRICS_DIR = ROOT / "outputs" / "model_metrics"

FEATURE_COLUMNS = ["quantity", "unit_price", "revenue"]
TARGET_COLUMN = "recommendation"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {', '.join(sorted(missing))}")
    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training data must contain at least two recommendation categories.")

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = sorted(y.unique())
    report_text = classification_report(y_test, predictions, labels=labels, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    accuracy = float(accuracy_score(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    dump({"model": model, "features": FEATURE_COLUMNS, "labels": labels}, MODEL_PATH)
    (METRICS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(METRICS_DIR / "confusion_matrix.csv")
    metrics = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "dataset_type": "synthetic labelled demonstration data",
        "total_records": int(len(data)),
        "training_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "model": "DecisionTreeClassifier",
        "random_state": 42,
        "accuracy": accuracy,
        "accuracy_numerator": int((predictions == y_test.to_numpy()).sum()),
        "accuracy_denominator": int(len(y_test)),
        "labels": labels,
        "warning": "This metric is from a small synthetic dataset and must not be presented as production accuracy.",
    }
    (METRICS_DIR / "recommendation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Test accuracy: {metrics['accuracy_numerator']}/{metrics['accuracy_denominator']} ({accuracy:.2%})")
    print(report_text)


if __name__ == "__main__":
    main()
