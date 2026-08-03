# ============================================================================
# Export & Package V3.7 Production Hybrid Model Pipeline
# ============================================================================
# Fits full production V3.7 Mooney prediction pipeline and serializes all artifacts:
# 1. models/v37_production_model_package/hybrid_model.joblib (Stage 1 + 1b + Stage 2 Experts)
# 2. models/v37_production_model_package/stage3_calibrator.joblib (Stage 3 EWMA)
# 3. models/v37_production_model_package/feature_metadata.json (Feature names & schema)
# 4. models/v37_production_model_package/online_inference_api.py (Production Inference Entrypoint)
# ============================================================================

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

from feature_engineering.clustering import cluster_silica_carbon_black
from feature_engineering.stage1_recipe_features import extract_stage1_recipe_features
from feature_engineering.stage2_process_features import extract_stage2_process_features
from feature_engineering.silica_pid_feature_builder import build_silica_pid_features
from feature_engineering.cb_dispersion_feature_builder import build_cb_dispersion_features
from model_training.effective_weighting import compute_effective_sample_weights
from model_training.hybrid_unified_model import HybridUnifiedMooneyModel
from model_training.label_group_handler import add_label_group_information
from model_training.stage3_online_calibration import Stage3DelayedFeedbackCalibrator


def package_production_v37_model():
    print("=" * 95)
    print("  PACKAGING & EXPORTING V3.7 PRODUCTION MODEL ARTIFACTS")
    print("=" * 95)

    model_dir = os.path.join(pipeline_root, 'models', 'v37_production_model_package')
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load Data
    data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../../data/stage_statistics_enriched_all_features_weather_v4.csv',
    ))
    if not os.path.exists(data_path):
        data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '../../data/enriched_mny_all.csv',
        ))

    df_raw = pd.read_csv(data_path, low_memory=False)
    if 'MNY' not in df_raw.columns and 'Mooney_Viscosity' in df_raw.columns:
        df_raw['MNY'] = df_raw['Mooney_Viscosity']
    df_clean = df_raw.dropna(subset=['MNY']).copy()

    if 'CompoundName' not in df_clean.columns and 'Compound' in df_clean.columns:
        df_clean['CompoundName'] = df_clean['Compound']
    if 'OrderID' not in df_clean.columns and 'Order_No' in df_clean.columns:
        df_clean['OrderID'] = df_clean['Order_No']

    # Build Features
    pid_feats = build_silica_pid_features(df_clean)
    cb_feats = build_cb_dispersion_features(df_clean)

    for c in pid_feats.columns:
        df_clean[c] = pid_feats[c]
    for c in cb_feats.columns:
        df_clean[c] = cb_feats[c]

    df_clean = cluster_silica_carbon_black(df_clean)
    df_clean = add_label_group_information(df_clean)
    df_clean = compute_effective_sample_weights(df_clean)

    s1_cols = extract_stage1_recipe_features(df_clean)
    s2_cols_base = extract_stage2_process_features(df_clean)
    s2_cols = list(set(s2_cols_base + list(pid_feats.columns) + list(cb_feats.columns)))

    for col in set(s1_cols + s2_cols):
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)

    # Train Full Production Model on 100% Data
    model = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True, use_cb_subsystem=True)
    model.fit(df_clean, s1_cols, s2_cols, target_col='MNY', cluster_col='material_system')

    # Save Model Artifacts
    joblib.dump(model, os.path.join(model_dir, 'hybrid_model.joblib'))

    calibrator = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha_base=0.3, adaptive=True)
    joblib.dump(calibrator, os.path.join(model_dir, 'stage3_calibrator.joblib'))

    metadata = {
        'model_version': 'V3.7 Production Hybrid Model',
        'train_samples_N': len(df_clean),
        's1_feature_count': len(s1_cols),
        's2_feature_count': len(s2_cols),
        's1_features': s1_cols,
        's2_features': s2_cols,
        'material_systems': ['Silica', 'CarbonBlack'],
    }
    with open(os.path.join(model_dir, 'feature_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Generate Online Inference Entrypoint API Code
    api_code = '''# ============================================================================
# Production Online Inference API Entrypoint (V3.7 Mooney Prediction)
# ============================================================================

import os
import json
import joblib
import pandas as pd
import numpy as np

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

class MooneyPredictionService:
    def __init__(self):
        self.model = joblib.load(os.path.join(PACKAGE_DIR, "hybrid_model.joblib"))
        self.calibrator = joblib.load(os.path.join(PACKAGE_DIR, "stage3_calibrator.joblib"))
        with open(os.path.join(PACKAGE_DIR, "feature_metadata.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
            
    def predict_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Receives raw PLC batch data, extracts features, and outputs Mooney predictions.
        """
        # Ensure numerical fallback for model features
        s1_cols = self.meta["s1_features"]
        s2_cols = self.meta["s2_features"]
        
        for c in set(s1_cols + s2_cols):
            if c in batch_df.columns:
                batch_df[c] = pd.to_numeric(batch_df[c], errors="coerce").fillna(0.0)
            else:
                batch_df[c] = 0.0
                
        # Stage 1 + 1b + Stage 2 Subsystem Prediction
        uncal_preds, s1_preds, s1b_biases, s2_res = self.model.predict(batch_df, cluster_col="material_system")
        
        # Stage 3 EWMA Calibration
        cal_preds, _ = self.calibrator.calibrate_time_series(batch_df, uncal_preds, target_col="MNY", group_col="CompoundName")
        
        res = batch_df.copy()
        res["stage1_recipe_baseline"] = s1_preds
        res["stage2_process_delta"] = s2_res
        res["predicted_mooney_viscosity"] = cal_preds
        return res

if __name__ == "__main__":
    service = MooneyPredictionService()
    print("V3.7 Production Mooney Prediction Service Ready.")
'''

    with open(os.path.join(model_dir, 'online_inference_api.py'), 'w', encoding='utf-8') as f:
        f.write(api_code)

    print("\n" + "=" * 95)
    print("      V3.7 PRODUCTION MODEL PACKAGE EXPORTED SUCCESSFULLY")
    print(f"      Location: {model_dir}")
    print("=" * 95)


if __name__ == '__main__':
    package_production_v37_model()
