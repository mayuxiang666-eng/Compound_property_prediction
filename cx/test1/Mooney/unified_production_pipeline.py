"""
Unified Production Pipeline (主干生产级全步序因果与基准管道)
=========================================================
100% Data-Driven & Optimization-Iterated Baselines.
100% Transparent Physical Causal Chains. Zero Black-Box Shortcuts.

Capabilities:
  1. Forward Prediction:
     Input process setpoint modifications across all 16 steps.
     Outputs: Predicted Absolute Mooney, Predicted Absolute Energy, Tolerance Bands [Pred ± 3.0 MU / Pred ± 2.0 kWh].
  2. Inverse Recipe Tuning:
     Input target Mooney, solves minimum-energy step setpoint adjustments.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data_driven_baseline_engine import (
    load_clean_dataset, 
    solve_baseline_by_loss_minimization, 
    solve_dynamic_sensor_rheology_baseline
)
from full_step_causal_engine import FullStepCausalEngine, FULL_STEP_SENSITIVITIES

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"

class UnifiedProductionPipeline:
    def __init__(self, clean_df: pd.DataFrame = None):
        self.causal_engine = FullStepCausalEngine()
        if clean_df is None:
            clean_df = load_clean_dataset()
        self.clean_df = clean_df
        
        # Solve global baseline via L-BFGS-B loss minimization directly from data
        print("[1/2] Numerically solving global data-driven baseline via L-BFGS-B loss minimization...")
        self.baseline_info = solve_baseline_by_loss_minimization(self.clean_df, verbose=False)
        print(f"      Solved Baseline: M0* = {self.baseline_info['M0']:.2f} MU, E0* = {self.baseline_info['E0']:.2f} kWh (Iterations: {self.baseline_info['iterations']})")
        print("[2/2] Unified production pipeline ready.")

    def predict_batch(self, 
                      base_mooney: float = None, 
                      base_energy: float = None, 
                      step_mods: dict = None,
                      sensor_row: pd.Series = None,
                      reference_batch: pd.Series = None) -> dict:
        """
        Executes transparent forward prediction for a batch.
        """
        # Determine baseline strictly from data/calculation
        if reference_batch is not None:
            m_base = float(reference_batch['Mooney'])
            e_base = float(reference_batch['Energy_Target'])
            baseline_source = f"由前序参考车次实测数据锚定 (车次 #{int(reference_batch['BatchNumber'])})"
        elif sensor_row is not None:
            m_base = solve_dynamic_sensor_rheology_baseline(sensor_row)
            e_base = self.baseline_info['E0']
            baseline_source = "由本车前30秒传感器机械阻力矩与温升动力学方程实时解算"
        elif base_mooney is not None:
            m_base = float(base_mooney)
            e_base = float(base_energy if base_energy is not None else self.baseline_info['E0'])
            baseline_source = "由外部传入基准"
        else:
            m_base = self.baseline_info['M0']
            e_base = self.baseline_info['E0']
            baseline_source = f"由全量数据容差死区损失极小化（L-BFGS-B 第 {self.baseline_info['iterations']} 轮迭代）求解所得"
            
        final_mooney, final_energy, trace = self.causal_engine.simulate_step_modifications(
            base_mooney=m_base,
            base_energy=e_base,
            step_mods=step_mods or {}
        )
        
        return {
            'baseline_mooney': m_base,
            'baseline_energy': e_base,
            'baseline_source': baseline_source,
            'predicted_mooney': final_mooney,
            'predicted_energy': final_energy,
            'mooney_band': (final_mooney - 3.0, final_mooney + 3.0),
            'energy_band': (final_energy - 2.0, final_energy + 2.0),
            'step_trace': trace
        }

    def optimize_setpoints_for_target(self, current_mooney: float, target_mooney: float) -> dict:
        """
        Solves optimal step setpoint adjustments to hit target Mooney with minimal energy.
        """
        delta_needed = target_mooney - current_mooney
        
        # Available controllable knobs:
        # Knob 1: MST_S07 (Step 7 rpm)
        # Knob 2: MSB_S01 (Step 1 time)
        # Knob 3: MSB_S03 (Dump temp)
        def cost(x):
            d_rpm, d_msb_time, d_temp = x
            dm = (-0.620 * d_rpm - 0.062 * d_msb_time - 0.102 * d_temp)
            de = (0.205 * d_rpm + 0.205 * (d_msb_time / 10.0) + 0.085 * d_temp)
            return 100.0 * (dm - delta_needed)**2 + 0.5 * (de**2) + 0.1 * (d_rpm**2 + 0.01*d_msb_time**2)
            
        bounds = [(-6.0, 10.0), (-25.0, 30.0), (-5.0, 8.0)]
        res = minimize(cost, [0.0, 0.0, 0.0], bounds=bounds, method='L-BFGS-B')
        
        opt_rpm, opt_time, opt_temp = res.x
        final_dm = -0.620 * opt_rpm - 0.062 * opt_time - 0.102 * opt_temp
        final_de = 0.205 * opt_rpm + 0.205 * (opt_time / 10.0) + 0.085 * opt_temp
        
        return {
            'target_mooney': target_mooney,
            'current_mooney': current_mooney,
            'delta_needed': delta_needed,
            'opt_rpm_mst7': opt_rpm,
            'opt_time_msb1': opt_time,
            'opt_temp_msb3': opt_temp,
            'expected_delta_mooney': final_dm,
            'expected_final_mooney': current_mooney + final_dm,
            'expected_delta_energy': final_de
        }

if __name__ == "__main__":
    pipeline = UnifiedProductionPipeline()
    res = pipeline.predict_batch(step_mods={'MST_S07': {'d_rpm': 5.0}})
    print("\n" + "=" * 80)
    print("【管道正向测试输出】")
    print(f"基准来源: {res['baseline_source']}")
    print(f"动态基准门尼: {res['baseline_mooney']:.2f} MU")
    print(f"预测绝对门尼: {res['predicted_mooney']:.2f} MU (容差区间: [{res['mooney_band'][0]:.2f}, {res['mooney_band'][1]:.2f}])")
    print(f"预测绝对能耗: {res['predicted_energy']:.2f} kWh (容差区间: [{res['energy_band'][0]:.2f}, {res['energy_band'][1]:.2f}])")
    print("=" * 80)
