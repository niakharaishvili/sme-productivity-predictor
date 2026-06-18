# SME Productivity Predictor

A machine learning project to predict productivity levels of small and medium-sized enterprises (SMEs) based on digitization adoption, firm size, sector, and other economic indicators.

Inspired by the BMBF-funded **HaMiZu (Handwerk mit Zukunft)** project on digital transformation and SME productivity in the German craft and trade sector.

---

## Project Structure

```
sme_productivity_predictor/
├── data/
│   ├── raw/               # Original, unmodified data
│   └── processed/         # Cleaned and feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Data Cleaning & Feature Engineering
│   ├── 03_modeling.ipynb           # Model Training & Evaluation
│   └── 04_insights.ipynb           # Business Insights & Reporting
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Data loading & synthetic data generation
│   ├── features.py        # Feature engineering functions
│   ├── model.py           # Model training and evaluation
│   └── visualize.py       # Plotting and visualization
├── models/                # Saved trained models (.pkl)
├── reports/               # Generated figures and reports
├── requirements.txt
└── README.md
```

---

## Problem Statement

Predict the **productivity level** of an SME (`Low` / `Medium` / `High`) based on:

| Feature | Description |
|---|---|
| `digitization_score` | Level of digital technology adoption (0–10) |
| `firm_size` | Number of employees |
| `sector` | Industry category (Craft, Trade, Services, etc.) |
| `region` | Geographic location in Germany |
| `innovation_index` | R&D and process innovation score (0–5) |
| `years_in_operation` | Firm age |
| `training_investment` | Annual training spend in EUR |

---

## ML Approach

| Step | Method |
|---|---|
| Baseline | Logistic Regression |
| Main Model | Random Forest Classifier |
| Hyperparameter Tuning | GridSearchCV (5-fold CV) |
| Evaluation | Accuracy, F1-score (macro), Confusion Matrix, Feature Importance |

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

Then open the notebooks in order: `01_eda` → `02_preprocessing` → `03_modeling` → `04_insights`.

---

## Author

Nia Kharaishvili — github.com/niakharaishvili
