# kpcl_selected_item_forecasting

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

specific item code forecasting given by kpcl

## 🚀 Execution Flow

Follow this sequence to run the entire pipeline:

1.  **Data Preparation**:
    ```bash
    python src/data/data_preparation.py
    ```
    *Cleans, filters, diagnoses quality, and splits data into train/test sets.*

2.  **Exploratory Analysis (Optional)**:
    ```bash
    python src/visualization/mstl_analysis.py
    ```
    *Performs MSTL decomposition for all items.*

3.  **Comparative Analysis (Optional)**:
    ```bash
    python src/modeling/compare_models_rmse.py
    ```
    *Compares multiple models (AR, MA, Prophet, etc.) to identify best performers.*

4.  **Modeling & Forecasting**:
    ```bash
    python src/modeling/train_forecast_all_models.py
    ```
    *Trains optimal models and generates 12-week forecasts with dual confidence intervals.*

5.  **Validation**:
    ```bash
    python src/forecast_validation/validate_all_models.py
    ```
    *Compares forecasts against actual test data and generates performance metrics.*

## 📁 Optimized Project Structure

```
├── data
│   ├── raw            <- Original Excel data dump.
│   └── processed      <- Final CSVs and validation results.
├── models             <- Trained .pkl files for all items.
├── reports
│   └── figures        <- Forecast plots, MSTL analysis, and validation charts.
└── src
    ├── data
    │   └── data_preparation.py     <- Loading, Cleaning, and Splitting.
    ├── modeling
    │   ├── compare_models_rmse.py  <- Multi-model comparison.
    │   └── train_forecast_all_models.py <- Best model Training & Forecasting.
    ├── visualization
    │   └── mstl_analysis.py        <- Time-series Decomposition.
    └── forecast_validation
        └── validate_all_models.py  <- Forecast vs Actual Validation.
```

--------

