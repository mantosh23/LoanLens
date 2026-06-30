# LoanLens

**LoanLens** is an explainable consumer loan credit risk assessment project built with XGBoost, SHAP, and FastAPI. It pairs a model training and interpretability notebook with a lightweight web interface for predicting loan default probabilities.

## Features

- **Model Training & Explainability (`loanlens_credit_risk.ipynb`)**:
  - Exploratory data analysis, validation, and outlier filtering on `credit_risk_dataset.csv`
  - Preprocessing pipelines (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) via `ColumnTransformer`
  - Stratified 5-fold cross-validation comparing Logistic Regression and XGBoost
  - Hyperparameter tuning with `RandomizedSearchCV` and probability calibration with `CalibratedClassifierCV`
  - Global and local feature attribution using SHAP (`TreeExplainer`, summary plot, and waterfall plot)
- **Inference API & Web Interface (`main.py`, `static/`)**:
  - FastAPI backend serving a `/predict` endpoint backed by `credit_risk_model.pkl` and `best_threshold.pkl`
  - Browser-based underwriting form (`static/index.html`, `static/script.js`, `static/style.css`) with automatic loan-to-income ratio calculation and probability gauge readout

## Project Structure

```text
.
├── main.py                      # FastAPI application and /predict endpoint
├── loanlens_credit_risk.ipynb   # EDA, modeling, calibration, and SHAP notebook
├── credit_risk_dataset.csv      # Consumer loan dataset
├── credit_risk_model.pkl        # Serialized calibrated classification pipeline
├── best_threshold.pkl           # Saved classification threshold
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Python runtime version for deployment
├── render.yaml                  # Render web service configuration
└── static/
    ├── index.html               # Web UI markup
    ├── script.js                # Form submission and result gauge logic
    └── style.css                # Styling and theme tokens
```

## Running Locally

1. Create and activate a virtual environment, then install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Start the FastAPI server:

   ```bash
   uvicorn main:app --reload
   ```

3. Open `http://127.0.0.1:8000` in your browser.
