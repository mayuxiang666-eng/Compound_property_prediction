# ============================================================================
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
