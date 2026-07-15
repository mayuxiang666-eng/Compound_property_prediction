# ----------------------------------------------------
# Mooney Prediction Pipeline V2.0 - Rolling Calibration
# ----------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# Define classes to support unpickling TwoStageNonLinearModel
class Stage1Baseline:
    def __init__(self, recipe_cols):
        self.recipe_cols = recipe_cols
        self.compound_biases = {}
        self.fallback_model = make_pipeline(
            SimpleImputer(strategy='median'), 
            StandardScaler(), 
            Ridge(alpha=10.0)
        )
        self.global_mean = 0.0
        
    def fit(self, X_train_full, y_train):
        self.global_mean = y_train.mean()
        X_rec = X_train_full[self.recipe_cols]
        self.fallback_model.fit(X_rec, y_train)
        base_preds = self.fallback_model.predict(X_rec)
        df_tr = X_train_full.copy()
        df_tr['residual_mny'] = y_train - base_preds
        self.compound_biases = df_tr.groupby('CompoundName')['residual_mny'].mean().to_dict()
        
    def predict(self, X_eval):
        X_rec = X_eval[self.recipe_cols]
        base_preds = self.fallback_model.predict(X_rec)
        final_preds = []
        for i, (idx, row) in enumerate(X_eval.iterrows()):
            comp = row['CompoundName']
            bias = self.compound_biases.get(comp, 0.0)
            final_preds.append(base_preds[i] + bias)
        return np.array(final_preds)

class PhysicsDeviationTransformer:
    def __init__(self, recipe_cols, process_cols):
        self.recipe_cols = recipe_cols
        self.process_cols = process_cols
        self.baseline_models = {}
        self.active_process_cols = []
        
    def fit(self, X_train_full):
        X_rec = X_train_full[self.recipe_cols]
        self.active_process_cols = []
        for col in self.process_cols:
            y_col = X_train_full[col]
            if y_col.isnull().all():
                continue
            col_mean = y_col.mean()
            if pd.isna(col_mean):
                col_mean = 0.0
            y_col_filled = y_col.fillna(col_mean)
            model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=100.0))
            model.fit(X_rec, y_col_filled)
            self.baseline_models[col] = model
            self.active_process_cols.append(col)
            
    def transform(self, X_eval):
        X_out = X_eval.copy()
        X_rec = X_eval[self.recipe_cols]
        for col in self.process_cols:
            if col in self.active_process_cols:
                pred_nominal = self.baseline_models[col].predict(X_rec)
                X_out[col] = X_eval[col] - pred_nominal
            else:
                X_out[col] = 0.0
        return X_out[self.process_cols]

class TwoStageNonLinearModel:
    def __init__(self, recipe_cols, process_cols, residual_model):
        self.recipe_cols = recipe_cols
        self.process_cols = process_cols
        self.baseline_model = Stage1Baseline(recipe_cols)
        self.dev_transformer = PhysicsDeviationTransformer(recipe_cols, process_cols)
        self.residual_model = make_pipeline(
            SimpleImputer(strategy='median'), 
            StandardScaler(), 
            residual_model
        )
        
    def fit(self, X_train_full, y_train):
        self.baseline_model.fit(X_train_full, y_train)
        y_baseline = self.baseline_model.predict(X_train_full)
        y_residual = y_train - y_baseline
        self.dev_transformer.fit(X_train_full)
        X_dev = self.dev_transformer.transform(X_train_full)
        self.residual_model.fit(X_dev, y_residual)
        
    def predict(self, X_eval):
        y_baseline = self.baseline_model.predict(X_eval)
        X_dev = self.dev_transformer.transform(X_eval)
        y_residual_pred = self.residual_model.predict(X_dev)
        return y_baseline + y_residual_pred

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_DIR = os.path.dirname(PARENT_DIR)

def main():
    print("\n=== STEP 1: LOADING CLEANED DATASET ===")
    clean_path = os.path.join(WORKSPACE_DIR, "Mooney_Prediction_Pipeline", "models", "hefei_group_training_dataset_clean.csv")
    if not os.path.exists(clean_path):
        print(f"[Error] Cleaned dataset not found at {clean_path}. Please run train_group_mooney_models_nonlinear.py first.")
        return
        
    df = pd.read_csv(clean_path)
    df['OrderStartTime'] = pd.to_datetime(df['OrderStartTime'])
    df = df.sort_values(by=['OrderID', 'OrderStartTime', 'PalletID']).reset_index(drop=True)
    print(f"Loaded {len(df)} cleaned group/pallet samples.")

    print("\n=== STEP 2: LOADING PRE-TRAINED HIGH-SILICA MODEL BUNDLE ===")
    model_dir = os.path.join(WORKSPACE_DIR, "Mooney_Prediction_Pipeline", "models", "results_with_oil_high_silica", "group_model_nonlinear")
    bundle_path = os.path.join(model_dir, "mooney_group_model_bundle.joblib")
    
    if not os.path.exists(bundle_path):
        model_dir = os.path.join(WORKSPACE_DIR, "Mooney_Prediction_Pipeline", "models", "results_with_oil_high_silica", "group_model")
        bundle_path = os.path.join(model_dir, "mooney_group_model_bundle.joblib")
        
    if not os.path.exists(bundle_path):
        print(f"[Error] No pre-trained model bundle found at {bundle_path}. Please run training first.")
        return
        
    print(f"Loading model bundle from: {bundle_path}")
    # Explicitly map module name to main to avoid unpickling error
    sys.modules['__main__'].TwoStageNonLinearModel = TwoStageNonLinearModel
    sys.modules['__main__'].Stage1Baseline = Stage1Baseline
    sys.modules['__main__'].PhysicsDeviationTransformer = PhysicsDeviationTransformer
    
    bundle = joblib.load(bundle_path)
    model = bundle['model']
    recipe_cols = bundle['recipe_cols']
    process_cols = bundle['process_cols']
    
    # Filter dataset for high silica system
    df['is_silica_system'] = ((df['silica_phr'] >= 25.0) & (df['weight_pct_silian'] > 0.0)).astype(float)
    df_sub = df[(df['is_oil_loading_present'] == 1.0) & (df['is_silica_system'] == 1.0)].copy()
    print(f"Filtered to {len(df_sub)} samples in With-Oil High-Silica track.")

    if len(df_sub) == 0:
        print("[Error] No samples in With-Oil High-Silica track to test.")
        return

    # ----------------------------------------------------
    # Step 3: Simulate Online Predictions and Rolling Bias Offsetting
    # ----------------------------------------------------
    print("\n=== STEP 3: RUNNING ROLLING LOT/ORDER BIAS ESTIMATION SIMULATION ===")
    
    # Generate static predictions
    X = df_sub.copy()
    y_actual = df_sub['MNY'].values
    y_pred_static = model.predict(X)
    
    df_sub['y_actual'] = y_actual
    df_sub['y_pred_static'] = y_pred_static
    df_sub['error_static'] = y_actual - y_pred_static
    
    # We will simulate a rolling window of order-level residual calibration.
    y_pred_rolling = []
    recent_residuals = {}
    
    # Iterate through pallets in chronological order
    for idx, row in df_sub.iterrows():
        order_id = row['OrderID']
        pred_static = row['y_pred_static']
        
        # If we have a previous residual for this OrderID, apply it as an offset (rolling calibration)
        if order_id in recent_residuals:
            offset = recent_residuals[order_id]
            pred_rolling = pred_static + offset
        else:
            # Fall back to static prediction (no offset)
            pred_rolling = pred_static
            
        y_pred_rolling.append(pred_rolling)
        
        # Update the most recent residual for this OrderID (lab measurement feedback loop)
        recent_residuals[order_id] = row['y_actual'] - pred_static

    df_sub['y_pred_rolling'] = y_pred_rolling
    df_sub['error_rolling'] = y_actual - df_sub['y_pred_rolling']

    # ----------------------------------------------------
    # Step 4: Metric Comparison
    # ----------------------------------------------------
    rmse_static = np.sqrt(np.mean(df_sub['error_static']**2))
    mae_static = np.mean(np.abs(df_sub['error_static']))
    
    rmse_rolling = np.sqrt(np.mean(df_sub['error_rolling']**2))
    mae_rolling = np.mean(np.abs(df_sub['error_rolling']))
    
    print("\n" + "="*70)
    print(f"   ROLLING CALIBRATION PERFORMANCE COMPARISON ({bundle['track_name']})")
    print("="*70)
    print(f"  Metric       | Static Model | Rolling Calibrated Model | Improvement")
    print("-"*70)
    print(f"  RMSE (MU)    |  {rmse_static:11.4f} |  {rmse_rolling:22.4f} |  {(rmse_static - rmse_rolling)/rmse_static:+7.2%}")
    print(f"  MAE (MU)     |  {mae_static:11.4f} |  {mae_rolling:22.4f} |  {(mae_static - mae_rolling)/mae_static:+7.2%}")
    print("="*70)
    print("  Note: Rolling calibration uses the actual lab error from the previous pallet")
    print("        of the same order to offset the subsequent pallet's prediction.")
    print("="*70)

if __name__ == '__main__':
    main()
