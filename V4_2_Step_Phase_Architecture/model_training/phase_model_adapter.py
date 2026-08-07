"""
V4.2 Module D: Phase Model Adapter
Wraps existing V4.1 MixingEnergyPredictionModel and HybridUnifiedMooneyModel for V4.2 architecture.
"""
import os
import sys
import numpy as np
import pandas as pd

pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

from V4_1_Energy_Optimization.model_training.energy_model import MixingEnergyPredictionModel
from Mooney_Prediction_Pipeline.model_training.hybrid_unified_model import HybridUnifiedMooneyModel

class PhaseModelAdapter:
    """
    Adapter wrapping V4.1 phase-level energy and Mooney models.
    Maps V4.2 phase feature names to expected V4.1 input columns.
    """
    def __init__(
        self,
        energy_model: MixingEnergyPredictionModel,
        mooney_model: HybridUnifiedMooneyModel
    ):
        self.energy_model = energy_model
        self.mooney_model = mooney_model

    def predict_energy(self, df_phase: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts total batch kWh and kWh/ton using V4.1 6-stage energy model.
        """
        return self.energy_model.predict(df_phase)

    def predict_mooney(self, df_phase: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts Mooney viscosity using V4.1 hybrid unified model.
        Returns: (final_pred, pred_s1, pred_s1b_bias, pred_s2_res)
        """
        return self.mooney_model.predict(df_phase, cluster_col='material_system')

if __name__ == '__main__':
    energy_model = MixingEnergyPredictionModel(mode='mode_b')
    mooney_model = HybridUnifiedMooneyModel()
    adapter = PhaseModelAdapter(energy_model, mooney_model)
    print("[OK] Task 6: Phase Model Adapter initialized successfully.")
