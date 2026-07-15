# Mooney Prediction Pipeline - Advanced Training & Rolling Calibration

This directory contains the optimized model training and online simulation scripts for predicting rubber compound Mooney Viscosity ($MNY$).

## Scripts Overview

### 1. `train_group_mooney_models_nonlinear.py`
Trains a **Two-Stage Non-Linear Model** that decouples recipe characteristics from process deviations:
* **Stage 1 (Nominal Baseline)**: Uses Ridge Regression to fit recipe features and raw material supplier quality indicators (e.g., `supplier_rubber_viscosity_avg`), adjusting with historical compound-specific biases to handle raw polymer lot switches.
* **Stage 2 (Process Residuals)**: Fits process deviation features (actual process variable minus nominal predicted process value) using a non-linear **LightGBM Regressor** (`LGBMRegressor`) to capture complex, non-linear mixing interactions.
* **Safe-Net Pre-Filtering**: Automatically filters out instrumentation anomalies (power sensor 16-bit saturation at 65,531 kW and temperature flatlines).
* **Physical Kinetics Integration**: Computes equivalent temperature-time integrals for white carbon black silanization reaction rate (`I_silanization`) and scorch reaction rate (`I_scorch`) directly from raw 1Hz curves.
* **Outlier Rejection**: Runs an `IsolationForest` on the aggregated group/pallet level to discard multivariate process anomalies (contamination set to 3%).

### 2. `test_rolling_lot_calibration_cv.py`
Simulates a real-time **Adaptive Lab Feedback Loop (Rolling Calibration)** exclusively on out-of-fold validation sets across 5-Fold Group Cross-Validation:
* Chronologically orders pallets within each production order (`OrderID`).
* Uses the prediction residual (Actual Lab Mooney - Model Predicted Mooney) of the most recently tested pallet of the same order as a rolling offset correction for subsequent pallets:
  $$Y_{\text{calibrated}} = Y_{\text{predicted\_static}} + \text{Residual}_{\text{latest}}$$
* Measures true production-line performance on unseen recipes/lots where raw lot documentation is missing or barcodes are scanned with a delay (lag).

---

## Getting Started

### Prerequisites
Make sure your Python environment has the following libraries installed:
```bash
pip install numpy pandas scikit-learn lightgbm joblib
```

### Running the Training Pipeline
To pre-filter curves, compute kinetics, aggregate, and train all four sub-track models (With/Without Oil, Silica/Carbon-Black):
```bash
python train_group_mooney_models_nonlinear.py
```
* **Outputs**: Serialized model bundles are saved in `../models/<track_name>/group_model_nonlinear/mooney_group_model_bundle.joblib`.
* **Reports**: Validation performance reports (`mny_predictive_modeling_report.md`) are saved in the corresponding folder.

### Running the Rolling Calibration Simulation
To evaluate the out-of-fold performance improvement from the adaptive feedback loop on unseen orders:
```bash
python test_rolling_lot_calibration_cv.py
```

---

## Model Performance (With-Oil High-Silica Track)

| Metric | Static Model (CV) | Rolling Calibrated Model (CV) | Improvement |
| :--- | :---: | :---: | :---: |
| **Group CV $R^2$** | $0.7807$ | **$0.7882$** | **$+0.97\%$** |
| **Group CV RMSE** | $3.5717\text{ MU}$ | **$3.5097\text{ MU}$** | **$-1.73\%$** |
| **Group CV MAE** | $2.6375\text{ MU}$ | **$2.5775\text{ MU}$** | **$-2.28\%$** |
| **Within-Compound Corr** | $0.2697$ | **$0.3555$** | **$+0.0858$** |
