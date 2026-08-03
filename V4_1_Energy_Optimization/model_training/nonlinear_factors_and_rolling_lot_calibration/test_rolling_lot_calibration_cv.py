# ----------------------------------------------------
# Mooney Prediction Pipeline V2.0 - Rolling Calibration CV
# ----------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

# Define helper classes for Stage 1 Baseline, Physics deviation and TwoStage Model
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
        print(f"[Error] Cleaned dataset not found at {clean_path}.")
        return
        
    df = pd.read_csv(clean_path)
    df['OrderStartTime'] = pd.to_datetime(df['OrderStartTime'])
    
    # Filter dataset for With-Oil High-Silica track
    df['is_silica_system'] = ((df['silica_phr'] >= 25.0) & (df['weight_pct_silian'] > 0.0)).astype(float)
    df_sub = df[(df['is_oil_loading_present'] == 1.0) & (df['is_silica_system'] == 1.0)].copy()
    
    # Sort chronologically within each OrderID to ensure chronological feedback simulation
    df_sub = df_sub.sort_values(by=['OrderID', 'OrderStartTime', 'PalletID']).reset_index(drop=True)
    print(f"Filtered and sorted With-Oil High-Silica track size: {len(df_sub)} samples.")

    # ----------------------------------------------------
    # Define features identical to training pipeline
    # ----------------------------------------------------
    recipe_cols = [
        'Top_Fill_Factor', 'Bot_Fill_Factor', 'Target_Temperature',
        'weight_pct_solid_elastomer', 'weight_pct_natural_rubber', 'weight_pct_silica',
        'weight_pct_oil', 'weight_pct_silian', 'weight_pct_carbon_black', 'silica_phr',
        'is_oil_loading_present', 'ratio_nr_rubber', 'ratio_filler_polymer',
        'ratio_oil_polymer', 'ratio_oil_filler',
        'supplier_rubber_viscosity_avg', 'supplier_silica_moisture_avg', 'supplier_silica_surface_area_avg',
        'supplier_carbon_black_structure_avg', 'supplier_carbon_black_surface_area_avg', 'supplier_carbon_black_moisture_avg'
    ]
    core_process_features = [
        'phys_discharge_temp', 'phys_max_temp', 'phys_eta_app_discharge',
        'Stage6_BottomMixing_Torque_Mean', 'Stage6_BottomMixing_power_Mean', 'Stage6_BottomMixing_Duration',
        'Stage6_BottomMixing_Torque_Integral', 'Stage4_WetMixing_temp_Mean', 'Stage4_WetMixing_Duration',
        'Stage2_DryMixing_Duration', 'Stage2_DryMixing_power_Mean',
        'env_temp_mean', 'env_humidity_mean',
        'I_silanization', 'I_scorch', 'time_to_sil_plateau_duration', 'top_sil_duration',
        'bottom_sil_duration', 'silanization_energy_mj', 'top_avg_sil_temperature', 'bottom_avg_sil_temperature',
        'phys_init_temp', 'phys_temp_integral', 'phys_temp_integral_above_100'
    ]
    
    available_recipe = [r for r in recipe_cols if r in df_sub.columns]
    available_process = [p for p in core_process_features if p in df_sub.columns]

    # ----------------------------------------------------
    # Step 2: 5-Fold Group CV Rolling Simulation
    # ----------------------------------------------------
    print("\n=== STEP 2: SIMULATING ROLLING CALIBRATION ON VALIDATION SETS ONLY ===")
    
    gkf = GroupKFold(n_splits=5)
    
    # Pre-allocate arrays for accumulating validation results
    val_indices = []
    val_actuals = []
    val_preds_static = []
    val_preds_rolling = []
    
    # Track order IDs for each validation sample to calculate within-compound correlations
    val_compounds = []
    val_order_ids = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_sub, df_sub['MNY'], df_sub['OrderID'])):
        print(f"  Processing Fold {fold+1}/5...")
        df_train = df_sub.iloc[train_idx].copy()
        df_val = df_sub.iloc[val_idx].copy()
        
        # 1. Train the Two-Stage model on the train set
        res_model = LGBMRegressor(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=5, 
            num_leaves=15, 
            min_child_samples=10,
            verbosity=-1,
            random_state=42
        )
        ts_model = TwoStageNonLinearModel(available_recipe, available_process, res_model)
        ts_model.fit(df_train, df_train['MNY'])
        
        # 2. Sort the validation fold chronologically to ensure valid feedback simulation
        df_val = df_val.sort_values(by=['OrderID', 'OrderStartTime', 'PalletID']).reset_index(drop=True)
        
        # 3. Predict Static MNY for validation fold
        y_val_actual = df_val['MNY'].values
        y_val_pred_static = ts_model.predict(df_val)
        
        # 4. Simulate Chronological Rolling Calibration on Validation Fold
        y_val_pred_rolling = []
        recent_val_residuals = {} # key: OrderID, value: latest prediction error
        
        for i, row in df_val.iterrows():
            order_id = row['OrderID']
            pred_static = y_val_pred_static[i]
            
            # If a prior tested pallet from the SAME order exists in the validation set, apply its error
            if order_id in recent_val_residuals:
                offset = recent_val_residuals[order_id]
                pred_rolling = pred_static + offset
            else:
                # No previous feedback available yet for this order, fall back to static
                pred_rolling = pred_static
                
            y_val_pred_rolling.append(pred_rolling)
            
            # Update the feedback error for this order
            recent_val_residuals[order_id] = y_val_actual[i] - pred_static
            
        # Accumulate results
        val_indices.extend(df_val.index)
        val_actuals.extend(y_val_actual)
        val_preds_static.extend(y_val_pred_static)
        val_preds_rolling.extend(y_val_pred_rolling)
        val_compounds.extend(df_val['CompoundName'].values)
        val_order_ids.extend(df_val['OrderID'].values)

    val_actuals = np.array(val_actuals)
    val_preds_static = np.array(val_preds_static)
    val_preds_rolling = np.array(val_preds_rolling)
    
    # ----------------------------------------------------
    # Step 3: Compute Out-of-Fold Validation Metrics
    # ----------------------------------------------------
    rmse_static = np.sqrt(mean_squared_error(val_actuals, val_preds_static))
    mae_static = mean_absolute_error(val_actuals, val_preds_static)
    r2_static = r2_score(val_actuals, val_preds_static)
    
    rmse_rolling = np.sqrt(mean_squared_error(val_actuals, val_preds_rolling))
    mae_rolling = mean_absolute_error(val_actuals, val_preds_rolling)
    r2_rolling = r2_score(val_actuals, val_preds_rolling)

    # Within-Compound Correlation for Validation Sets
    df_eval_val = pd.DataFrame({
        'CompoundName': val_compounds,
        'OrderID': val_order_ids,
        'Actual_MNY': val_actuals,
        'Pred_Static': val_preds_static,
        'Pred_Rolling': val_preds_rolling
    })
    
    comp_stats_static = df_eval_val.groupby('CompoundName').agg(
        mean_actual=('Actual_MNY', 'mean'),
        mean_pred_static=('Pred_Static', 'mean'),
        mean_pred_rolling=('Pred_Rolling', 'mean'),
        count=('Actual_MNY', 'count')
    ).reset_index()
    
    df_eval_val = df_eval_val.merge(comp_stats_static, on='CompoundName', how='left')
    df_eval_val['actual_dev'] = df_eval_val['Actual_MNY'] - df_eval_val['mean_actual']
    df_eval_val['pred_static_dev'] = df_eval_val['Pred_Static'] - df_eval_val['mean_pred_static']
    df_eval_val['pred_rolling_dev'] = df_eval_val['Pred_Rolling'] - df_eval_val['mean_pred_rolling']
    
    df_multi = df_eval_val[df_eval_val['count'] >= 3]
    
    dev_corr_static = df_multi['actual_dev'].corr(df_multi['pred_static_dev']) if len(df_multi) > 1 else np.nan
    dev_corr_rolling = df_multi['actual_dev'].corr(df_multi['pred_rolling_dev']) if len(df_multi) > 1 else np.nan

    print("\n" + "="*80)
    print("   OUT-OF-FOLD GROUP CV ROLLING CALIBRATION REPORT (With-Oil High-Silica)")
    print("="*80)
    print(f"  Metric                      | Static Model | Rolling Calibrated Model | Improvement")
    print("-"*80)
    print(f"  Group CV R^2                |  {r2_static:11.4f} |  {r2_rolling:22.4f} |  {((r2_rolling - r2_static) / abs(r2_static)):+7.2%}")
    print(f"  Group CV RMSE (MU)          |  {rmse_static:11.4f} |  {rmse_rolling:22.4f} |  {((rmse_static - rmse_rolling)/rmse_static):+7.2%}")
    print(f"  Group CV MAE (MU)           |  {mae_static:11.4f} |  {mae_rolling:22.4f} |  {((mae_static - mae_rolling)/mae_static):+7.2%}")
    print(f"  Within-Compound Correlation |  {dev_corr_static:11.4f} |  {dev_corr_rolling:22.4f} |  {(dev_corr_rolling - dev_corr_static):+7.4f}")
    print("="*80)
    print("  Note: This report simulates rolling calibration exclusively on out-of-fold validation")
    print("        orders, representing true online prediction performance on unseen recipes/lots.")
    print("="*80)

if __name__ == '__main__':
    main()
