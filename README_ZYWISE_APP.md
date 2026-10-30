# ZyWise Application

This folder contains the working ZyWise application. It supports CSV and Excel upload, rule-based cleaning, business analytics, a trained recommendation-model demonstration and result downloads.

## Run in VS Code

From the project root:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run zywise_app/app.py
```

The application opens in a browser. Use **Load dashboard sample data** to test the dashboard immediately, or upload a CSV/XLSX file.

## Current modules

- `zywise_app/cleaning.py`: file reading, column mapping, text/date/numeric cleaning, missing values and duplicate handling.
- `zywise_app/analytics.py`: revenue, orders, quantity, average order value, product, category and monthly metrics.
- `zywise_app/recommendations.py`: model-backed recommendations with a transparent rule-based fallback.
- `zywise_app/app.py`: Streamlit user interface and downloads.
- `data/recommendation_training.csv`: synthetic labelled training data for the recommendation workflow.
- `train_model.py`: trains and evaluates the DecisionTreeClassifier.
- `models/recommendation_model.joblib`: saved trained model used by the Recommendations page.
- `outputs/model_metrics/`: accuracy, classification report and confusion matrix evidence.

Authentication and PostgreSQL persistence are implemented. FastAPI endpoints and full automated testing remain future development stages. The training data and reported model metrics are synthetic demonstration evidence and must not be presented as production accuracy.
