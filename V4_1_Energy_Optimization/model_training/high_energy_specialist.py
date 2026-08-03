import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from model_training.energy_model import MixingEnergyPredictionModel


class HighEnergySpecialist:
    """Offline Mode B residual specialist for batches predicted to be high-energy risks."""

    def __init__(
        self,
        target_branch: str = 'Silica_OilWet',
        risk_threshold: float = 0.50,
        min_calibrator_samples: int = 30,
    ):
        self.target_branch = target_branch
        self.risk_threshold = risk_threshold
        self.min_calibrator_samples = min_calibrator_samples
        self.batch_thresholds: dict[str, float] = {}
        self.ton_thresholds: dict[str, float] = {}
        self.global_batch_threshold = np.nan
        self.global_ton_threshold = np.nan
        self.risk_model: LGBMClassifier | None = None
        self.risk_prior = 0.0
        self.residual_model: LGBMRegressor | None = None
        self.residual_lower = 0.0
        self.residual_upper = 0.0
        self.calibrator_sample_count = 0

    def _threshold_series(self, df: pd.DataFrame, thresholds: dict[str, float], fallback: float) -> pd.Series:
        return df['system_route_branch'].map(thresholds).fillna(fallback)

    def label_high_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        labels = pd.DataFrame(index=df.index)
        target_mask = df['system_route_branch'].eq(self.target_branch)
        batch_threshold = self._threshold_series(df, self.batch_thresholds, self.global_batch_threshold)
        ton_threshold = self._threshold_series(df, self.ton_thresholds, self.global_ton_threshold)
        labels['high_energy_batch_flag'] = (
            target_mask & (df['total_kwh_per_batch'] >= batch_threshold)
        ).astype(int)
        labels['high_energy_ton_flag'] = (
            target_mask & (df['kwh_per_ton'] >= ton_threshold)
        ).astype(int)
        labels['high_energy_union_flag'] = (
            (labels['high_energy_batch_flag'] == 1) | (labels['high_energy_ton_flag'] == 1)
        ).astype(int)
        return labels

    def fit(
        self,
        mode_b_model: MixingEnergyPredictionModel,
        df_train: pd.DataFrame,
        df_calibration: pd.DataFrame,
    ) -> None:
        branch_train = df_train.loc[df_train['system_route_branch'] == self.target_branch].copy()
        branch_calibration = df_calibration.loc[df_calibration['system_route_branch'] == self.target_branch].copy()
        if branch_train.empty:
            return
        self.batch_thresholds = {
            self.target_branch: float(branch_train['total_kwh_per_batch'].quantile(0.90))
        }
        self.ton_thresholds = {
            self.target_branch: float(branch_train['kwh_per_ton'].quantile(0.90))
        }
        self.global_batch_threshold = self.batch_thresholds[self.target_branch]
        self.global_ton_threshold = self.ton_thresholds[self.target_branch]

        train_labels = self.label_high_energy(branch_train)
        y_risk = train_labels['high_energy_ton_flag'].to_numpy()
        self.risk_prior = float(y_risk.mean())
        if np.unique(y_risk).size > 1:
            self.risk_model = LGBMClassifier(
                n_estimators=60,
                learning_rate=0.06,
                max_depth=3,
                min_child_samples=25,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            )
            self.risk_model.fit(mode_b_model.transform_pre_batch_features(branch_train), y_risk)

        calibration_labels = self.label_high_energy(branch_calibration)
        calibration_mask = calibration_labels['high_energy_union_flag'].to_numpy(dtype=bool)
        self.calibrator_sample_count = int(calibration_mask.sum())
        if self.calibrator_sample_count < self.min_calibrator_samples:
            return

        base_prediction, _ = mode_b_model.predict(branch_calibration)
        residual_target = branch_calibration['total_kwh_per_batch'].to_numpy() - base_prediction
        selected_residuals = residual_target[calibration_mask]
        self.residual_lower, self.residual_upper = np.quantile(selected_residuals, [0.05, 0.95]).tolist()
        self.residual_model = LGBMRegressor(
            n_estimators=35,
            learning_rate=0.06,
            max_depth=3,
            min_child_samples=15,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        calibration_features = mode_b_model.transform_pre_batch_features(branch_calibration)
        self.residual_model.fit(calibration_features[calibration_mask], selected_residuals)

    def predict_components(self, mode_b_model: MixingEnergyPredictionModel, df: pd.DataFrame) -> dict[str, np.ndarray]:
        base_prediction, _ = mode_b_model.predict(df)
        features = mode_b_model.transform_pre_batch_features(df)
        target_mask = (df['system_route_branch'].to_numpy() == self.target_branch)
        if self.risk_model is None:
            risk_probability = np.full(len(df), self.risk_prior, dtype=float)
        else:
            risk_probability = self.risk_model.predict_proba(features)[:, 1]

        activate_specialist = (risk_probability >= self.risk_threshold) & target_mask
        residual_correction = np.zeros(len(df), dtype=float)
        if self.residual_model is not None:
            raw_correction = self.residual_model.predict(features)
            residual_correction = np.clip(raw_correction, self.residual_lower, self.residual_upper)
            residual_correction = np.where(activate_specialist, residual_correction, 0.0)

        return {
            'mode_b_base_prediction_kwh': base_prediction,
            'high_energy_probability': risk_probability,
            'high_energy_specialist_active': activate_specialist,
            'high_energy_residual_correction_kwh': residual_correction,
            'high_energy_adjusted_prediction_kwh': base_prediction + residual_correction,
            'target_branch': np.full(len(df), self.target_branch, dtype=object),
        }