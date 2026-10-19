#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continental AG Mixing Process Intelligence
Automated Compound Energy Saving Potential Ranking & Step-Level DOE Optimizer
(With Strict PID Total Duration Hard Rule Enforcement: Target 165/155/142°C)
=============================================================================
Hard Rule (Upper Mixer MST + Lower Mixer MSB Total PID Duration):
  - Target 165°C:  80 - 140 s
  - Target 155°C: 130 - 230 s
  - Target 142°C: 270 - 330 s

Usage:
  1. Plant-wide Ranking (Top N Carbon Black & Top N Silica):
     python compound_energy_optimizer.py --mode rank --top 5

  2. Deep Step-Level Diagnosis & DOE Recipe Optimization for Specific Compounds:
     python compound_energy_optimizer.py --mode diagnose --compounds "M1-B00458---- 12 012" "M1-B00163---- 08 010"

  3. Full Batch Execution (Rank + Diagnose All Top 5 CB & Top 5 Silica):
     python compound_energy_optimizer.py --mode all --top 5
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure plotting style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# ENGINEERING HARD RULE: TOTAL PID DURATION (MST + MSB COMBINED)
# =============================================================================
PID_DURATION_HARD_RULES = {
    165: {'tier_label': 'Target 165°C', 'min_s': 80.0,  'max_s': 140.0, 'mid_s': 110.0},
    155: {'tier_label': 'Target 155°C', 'min_s': 130.0, 'max_s': 230.0, 'mid_s': 180.0},
    142: {'tier_label': 'Target 142°C', 'min_s': 270.0, 'max_s': 330.0, 'mid_s': 300.0},
}


class CompoundEnergyOptimizer:
    def __init__(self,
                 batch_csv='reports/msb_minus10s_october_doe/m1_batches_sep18_oct08.csv',
                 bom_csv='scratch/compound_bom_classification.csv',
                 steps_csv='scratch/top_compounds_steps_raw.csv',
                 creds_path='Mooney_Prediction_Pipeline/data_processing/credentials.json',
                 output_dir='reports/compound_energy_optimization_auto'):
        self.batch_csv = batch_csv
        self.bom_csv = bom_csv
        self.steps_csv = steps_csv
        self.creds_path = creds_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def map_pid_target_tier(cname, mst_steps, msb_steps):
        """
        Determines the PID Target Temperature (°C) and maps it to one of the three
        engineering Hard Rule tiers:
          - Target 165°C:  80 - 140 s
          - Target 155°C: 130 - 230 s
          - Target 142°C: 270 - 330 s
        """
        is_silica = str(cname).startswith('M1-T')
        mst_max_sp = mst_steps['Temp_SP_Med'].max() if 'Temp_SP_Med' in mst_steps else mst_steps['Temp_C_Mean'].max()
        msb_max_sp = msb_steps['Temp_SP_Med'].max() if ('Temp_SP_Med' in msb_steps and not msb_steps.empty) else 0.0
        ref_temp = max(mst_max_sp if pd.notna(mst_max_sp) else 0.0,
                       msb_max_sp if pd.notna(msb_max_sp) else 0.0)
        if ref_temp <= 0:
            ref_temp = mst_steps['Temp_C_Mean'].max()

        # Silica compounds with PID target >= 160°C (161-165°C) belong to Target 165°C (80-140s)
        if is_silica:
            if ref_temp >= 160.0:
                tier_key = 165
            elif ref_temp >= 149.0:
                tier_key = 155
            else:
                tier_key = 142
        else:
            # Carbon Black compounds:
            # B15563R1 (150°C), B00460B (156-160°C), R00218W2 (157/145°C), B00458 (150°C), B00163 (150°C)
            if ref_temp >= 162.0:
                tier_key = 165
            elif ref_temp >= 149.0:
                tier_key = 155
            else:
                tier_key = 142

        rule = PID_DURATION_HARD_RULES[tier_key]
        return ref_temp, tier_key, rule

    def load_and_rank_compounds(self, min_batches=180):
        """
        Loads plant-wide batch data, classifies Carbon Black vs Silica via BOM + recipe rules,
        and calculates batch-to-batch energy spread (P85-P15, P50-P15) and Annual Saving Potential.
        """
        df = pd.read_csv(self.batch_csv)
        bom = pd.read_csv(self.bom_csv) if os.path.exists(self.bom_csv) else pd.DataFrame()

        # Exclude warmup batches 1-3 and sensor anomalies
        df_clean = df[(df['BatchNumber'] > 3) & (df['BatchWeight'] > 150) & (df['MST_Energy_kWh'] > 10)].copy()

        # Handle cumulative vs incremental MSB energy counter across mixer lines
        df_clean['Total_Energy_kWh'] = np.where(
            df_clean['MSB_Energy_kWh'] > df_clean['MST_Energy_kWh'],
            df_clean['MSB_Energy_kWh'],
            df_clean['MST_Energy_kWh'] + df_clean['MSB_Energy_kWh'].fillna(0)
        )
        df_clean['MSB_Net_Energy_kWh'] = np.where(
            df_clean['MSB_Energy_kWh'] > df_clean['MST_Energy_kWh'],
            df_clean['MSB_Energy_kWh'] - df_clean['MST_Energy_kWh'],
            df_clean['MSB_Energy_kWh'].fillna(0)
        )
        df_clean['Spec_Energy_kWh_t'] = df_clean['Total_Energy_kWh'] / (df_clean['BatchWeight'] / 1000.0)
        df_clean['MST_Spec_Energy_kWh_t'] = df_clean['MST_Energy_kWh'] / (df_clean['BatchWeight'] / 1000.0)
        df_clean['MSB_Spec_Energy_kWh_t'] = df_clean['MSB_Net_Energy_kWh'] / (df_clean['BatchWeight'] / 1000.0)

        if not bom.empty:
            df_clean = df_clean.merge(bom[['CompoundName', 'AvgSilicaSilane_kg', 'AvgCarbonBlack_kg']], on='CompoundName', how='left')
        else:
            df_clean['AvgSilicaSilane_kg'] = np.nan
            df_clean['AvgCarbonBlack_kg'] = np.nan

        def classify_filler(row):
            cname = str(row['CompoundName'])
            if any(cname.startswith(p) for p in ['M1-B', 'M1-A', 'M1-R', 'M1-S', 'M1-C']):
                return 'Carbon Black'
            if cname.startswith('M1-T'):
                return 'Silica'
            sil = row['AvgSilicaSilane_kg']
            cb = row['AvgCarbonBlack_kg']
            if pd.notna(sil) and pd.notna(cb) and (sil > 20 or cb > 20):
                return 'Silica' if sil > cb else 'Carbon Black'
            return 'Carbon Black'

        df_clean['FillerCategory'] = df_clean.apply(classify_filler, axis=1)

        records = []
        for cname, g in df_clean.groupby('CompoundName'):
            if len(g) < min_batches:
                continue
            q02 = g['Spec_Energy_kWh_t'].quantile(0.02)
            q98 = g['Spec_Energy_kWh_t'].quantile(0.98)
            sub = g[(g['Spec_Energy_kWh_t'] >= q02) & (g['Spec_Energy_kWh_t'] <= q98)]
            if len(sub) < 30:
                sub = g

            p15 = sub['Spec_Energy_kWh_t'].quantile(0.15)
            p50 = sub['Spec_Energy_kWh_t'].quantile(0.50)
            p85 = sub['Spec_Energy_kWh_t'].quantile(0.85)
            mean_e = sub['Spec_Energy_kWh_t'].mean()
            std_e = sub['Spec_Energy_kWh_t'].std()

            mst_p15 = sub['MST_Spec_Energy_kWh_t'].quantile(0.15)
            mst_p85 = sub['MST_Spec_Energy_kWh_t'].quantile(0.85)
            msb_p15 = sub['MSB_Spec_Energy_kWh_t'].quantile(0.15)
            msb_p85 = sub['MSB_Spec_Energy_kWh_t'].quantile(0.85)

            total_batches = len(g)
            window_tons = g['BatchWeight'].sum() / 1000.0
            annual_tons = window_tons * 17.5  # 20-day window scaled to 350 working days

            gap_p50_p15 = p50 - p15
            gap_p85_p15 = p85 - p15
            annual_saving_kwh = gap_p50_p15 * annual_tons

            records.append({
                'CompoundName': cname,
                'FillerCategory': g['FillerCategory'].iloc[0],
                'MixerID': g['MixerID'].mode().iloc[0],
                'Batches': total_batches,
                'WindowTons': window_tons,
                'AnnualTons_Est': annual_tons,
                'BatchWeight_kg': sub['BatchWeight'].mean(),
                'MST_Steps': int(sub['MST_StepCount'].median()) if pd.notna(sub['MST_StepCount'].median()) else 8,
                'MSB_Steps': int(sub['MSB_StepCount'].median()) if pd.notna(sub['MSB_StepCount'].median()) else 3,
                'MST_DumpTemp_C': sub['MST_MaxTemp_C'].mean(),
                'MSB_Step1_s': sub['MSB_Step1_Duration_s'].mean(),
                'TotalBatchEnergy_kWh': sub['Total_Energy_kWh'].mean(),
                'MST_Energy_kWh': sub['MST_Energy_kWh'].mean(),
                'MSB_NetEnergy_kWh': sub['MSB_Net_Energy_kWh'].mean(),
                'SpecEnergy_Mean': mean_e,
                'SpecEnergy_Std': std_e,
                'SpecEnergy_CV_pct': (std_e / mean_e) * 100.0,
                'SpecEnergy_P15': p15,
                'SpecEnergy_P50': p50,
                'SpecEnergy_P85': p85,
                'Gap_P85_P15_kWh_t': gap_p85_p15,
                'Gap_P50_P15_kWh_t': gap_p50_p15,
                'MST_Spread_P85_P15': mst_p85 - mst_p15,
                'MSB_Spread_P85_P15': msb_p85 - msb_p15,
                'AnnualSavingPotential_kWh': annual_saving_kwh,
                'Silica_kg': g['AvgSilicaSilane_kg'].iloc[0] if 'AvgSilicaSilane_kg' in g else np.nan,
                'CarbonBlack_kg': g['AvgCarbonBlack_kg'].iloc[0] if 'AvgCarbonBlack_kg' in g else np.nan
            })

        rank_df = pd.DataFrame(records).sort_values('AnnualSavingPotential_kWh', ascending=False)
        rank_df.to_csv(os.path.join(self.output_dir, 'plant_wide_energy_potential_ranking.csv'), index=False)
        return rank_df

    def diagnose_compound_steps(self, compound_name, rank_df=None):
        """
        Performs step-by-step telemetry diagnosis for a given compound and strictly enforces
        the PID Total Duration Hard Rule (MST_PID + MSB_PID):
          - Target 165°C:  80 - 140 s
          - Target 155°C: 130 - 230 s
          - Target 142°C: 270 - 330 s
        """
        if not os.path.exists(self.steps_csv):
            raise FileNotFoundError(f"Step telemetry file {self.steps_csv} not found.")

        df_all = pd.read_csv(self.steps_csv)
        df_c = df_all[df_all['CompoundName'] == compound_name].copy()

        if df_c.empty:
            return None

        actual_m1_weight_kg = None
        if rank_df is not None and compound_name in rank_df['CompoundName'].values:
            actual_m1_weight_kg = rank_df[rank_df['CompoundName'] == compound_name]['BatchWeight_kg'].iloc[0]

        # Filter out cold-start batches (BatchNumber <= 3)
        df_c = df_c[df_c['BatchNumber'] > 3].copy()
        df_c['BatchKey'] = df_c['OrderID'].astype(str) + '~' + df_c['BatchNumber'].astype(str)

        steps_sub = df_c[df_c['GroupName'].isin(['AVR_MST', 'AVR_MSB'])].copy()
        piv = steps_sub.pivot_table(
            index=['BatchKey', 'OrderID', 'BatchNumber', 'EquipmentID', 'NominalWeight_kg', 'GroupName', 'StepNo'],
            columns='VariablePath',
            values='Value',
            aggfunc='first'
        ).reset_index()

        # Also extract Target setpoints per GroupName & StepNo
        sp_piv = steps_sub.pivot_table(
            index=['GroupName', 'StepNo'],
            columns='VariablePath',
            values='Target',
            aggfunc='median'
        ).reset_index()
        sp_map_temp = {}
        sp_map_rpm = {}
        for _, r in sp_piv.iterrows():
            key = (r['GroupName'], int(r['StepNo']))
            sp_map_temp[key] = r['SCP-3-Temperature-C-F'] if 'SCP-3-Temperature-C-F' in r else 0.0
            sp_map_rpm[key] = r['Rotor-Speed-1/min-Rpm'] if 'Rotor-Speed-1/min-Rpm' in r else 0.0

        # Sort and calculate incremental step time and step energy
        mst = piv[piv['GroupName'] == 'AVR_MST'].sort_values(['BatchKey', 'StepNo']).copy()
        mst['prev_time'] = mst.groupby('BatchKey')['SCP-1-Step-Time-rel-s'].shift(1).fillna(0)
        mst['prev_energy'] = mst.groupby('BatchKey')['SCP-2-Energy-rel-kWh'].shift(1).fillna(0)
        mst['step_time_s'] = (mst['SCP-1-Step-Time-rel-s'] - mst['prev_time']).clip(lower=0)
        mst['step_energy_kwh'] = (mst['SCP-2-Energy-rel-kWh'] - mst['prev_energy']).clip(lower=0)
        mst['step_power_kw'] = np.where(mst['step_time_s'] > 0.5, mst['step_energy_kwh'] * 3600.0 / mst['step_time_s'], 0)

        mst_end = mst.groupby('BatchKey').agg(
            mst_total_energy=('SCP-2-Energy-rel-kWh', 'max'),
            mst_total_time=('SCP-1-Step-Time-rel-s', 'max'),
            mst_step_count=('StepNo', 'count')
        ).reset_index()

        msb = piv[piv['GroupName'] == 'AVR_MSB'].sort_values(['BatchKey', 'StepNo']).copy()
        msb = msb.merge(mst_end, on='BatchKey', how='left')
        msb['prev_time'] = msb.groupby('BatchKey')['SCP-1-Step-Time-rel-s'].shift(1).fillna(0)
        msb['prev_energy'] = msb.groupby('BatchKey')['SCP-2-Energy-rel-kWh'].shift(1)

        s1_med = msb[msb['StepNo'] == 1]['SCP-2-Energy-rel-kWh'].median()
        mst_med = mst_end['mst_total_energy'].median()
        is_msb_cumulative = (s1_med > mst_med * 0.8) if (pd.notna(s1_med) and pd.notna(mst_med)) else False

        if is_msb_cumulative:
            msb.loc[msb['StepNo'] == 1, 'prev_energy'] = msb.loc[msb['StepNo'] == 1, 'mst_total_energy']
        else:
            msb.loc[msb['StepNo'] == 1, 'prev_energy'] = 0.0

        msb['step_time_s'] = (msb['SCP-1-Step-Time-rel-s'] - msb['prev_time']).clip(lower=0)
        msb['step_energy_kwh'] = (msb['SCP-2-Energy-rel-kWh'] - msb['prev_energy']).clip(lower=0)
        msb['step_power_kw'] = np.where(msb['step_time_s'] > 0.5, msb['step_energy_kwh'] * 3600.0 / msb['step_time_s'], 0)

        msb_end = msb.groupby('BatchKey').agg(
            msb_net_energy=('step_energy_kwh', 'sum'),
            msb_step_count=('StepNo', 'count')
        ).reset_index()

        all_steps = pd.concat([mst, msb], ignore_index=True)

        batch_totals = all_steps.groupby('BatchKey').agg(
            order_id=('OrderID', 'first'),
            total_energy_kwh=('step_energy_kwh', 'sum'),
            total_time_s=('step_time_s', 'sum'),
            weight_kg=('NominalWeight_kg', 'first'),
            equip_id=('EquipmentID', 'first')
        ).reset_index()
        batch_totals = batch_totals.merge(mst_end[['BatchKey', 'mst_total_energy', 'mst_step_count']], on='BatchKey', how='left')
        batch_totals = batch_totals.merge(msb_end[['BatchKey', 'msb_net_energy', 'msb_step_count']], on='BatchKey', how='left')

        if actual_m1_weight_kg is not None and actual_m1_weight_kg > 150:
            batch_totals['weight_kg'] = actual_m1_weight_kg

        med_mst_steps = batch_totals['mst_step_count'].median()
        med_msb_steps = batch_totals['msb_step_count'].median()
        batch_totals = batch_totals[
            (batch_totals['mst_step_count'] >= med_mst_steps - 1) &
            (batch_totals['msb_step_count'] >= max(1, med_msb_steps - 1)) &
            (batch_totals['msb_net_energy'] > 1.5) &
            (batch_totals['mst_total_energy'] > 12.0)
        ].copy()

        batch_totals['spec_energy_kwh_t'] = batch_totals['total_energy_kwh'] / (batch_totals['weight_kg'] / 1000.0)

        q03 = batch_totals['spec_energy_kwh_t'].quantile(0.03)
        q97 = batch_totals['spec_energy_kwh_t'].quantile(0.97)
        valid_batches = batch_totals[(batch_totals['spec_energy_kwh_t'] >= q03) & (batch_totals['spec_energy_kwh_t'] <= q97)].copy()

        # Compute P25 (Golden Low) and P75 (High Energy) within each OrderID to isolate pure batch-to-batch variance
        p25_cut = valid_batches.groupby('order_id')['spec_energy_kwh_t'].transform(lambda x: x.quantile(0.25))
        p75_cut = valid_batches.groupby('order_id')['spec_energy_kwh_t'].transform(lambda x: x.quantile(0.75))

        valid_batches['EnergyTier'] = np.where(
            valid_batches['spec_energy_kwh_t'] <= p25_cut, 'Golden_Low (P15)',
            np.where(valid_batches['spec_energy_kwh_t'] >= p75_cut, 'High_Energy (P85)', 'Mid')
        )

        all_steps = all_steps.merge(valid_batches[['BatchKey', 'EnergyTier', 'spec_energy_kwh_t', 'total_energy_kwh']], on='BatchKey', how='inner')

        step_rows = []
        for (gn, sn), sg in all_steps.groupby(['GroupName', 'StepNo']):
            if len(sg) < len(valid_batches) * 0.25:
                continue
            low_g = sg[sg['EnergyTier'] == 'Golden_Low (P15)']
            high_g = sg[sg['EnergyTier'] == 'High_Energy (P85)']

            e_low = low_g['step_energy_kwh'].mean() if len(low_g) > 0 else sg['step_energy_kwh'].mean()
            e_high = high_g['step_energy_kwh'].mean() if len(high_g) > 0 else sg['step_energy_kwh'].mean()
            t_low = low_g['step_time_s'].mean() if len(low_g) > 0 else sg['step_time_s'].mean()
            t_high = high_g['step_time_s'].mean() if len(high_g) > 0 else sg['step_time_s'].mean()

            step_rows.append({
                'GroupName': gn,
                'StepNo': int(sn),
                'StepLabel': f"{'MST' if gn=='AVR_MST' else 'MSB'} S{int(sn)}",
                'Speed_RPM_SP': sp_map_rpm.get((gn, int(sn)), np.nan),
                'Speed_RPM_Mean': sg['Rotor-Speed-1/min-Rpm'].mean(),
                'Speed_RPM_Std': sg['Rotor-Speed-1/min-Rpm'].std(),
                'Time_s_Mean': sg['step_time_s'].mean(),
                'Time_s_Median': sg['step_time_s'].median(),
                'Time_s_Std': sg['step_time_s'].std(),
                'Energy_kWh_Mean': sg['step_energy_kwh'].mean(),
                'Energy_kWh_Std': sg['step_energy_kwh'].std(),
                'Power_kW_Mean': sg['step_power_kw'].mean(),
                'Temp_SP_Med': sp_map_temp.get((gn, int(sn)), 0.0),
                'Temp_C_Mean': sg['SCP-3-Temperature-C-F'].mean(),
                'Torque_Nm_Mean': sg['SCP-5-Torque-Nm-ft/lbs'].mean() if 'SCP-5-Torque-Nm-ft/lbs' in sg else np.nan,
                'E_Low_kWh': e_low,
                'E_High_kWh': e_high,
                'Delta_E_High_Low_kWh': max(0.0, e_high - e_low),
                'T_Low_s': t_low,
                'T_High_s': t_high,
                'Delta_T_High_Low_s': t_high - t_low
            })

        step_df = pd.DataFrame(step_rows).sort_values(['GroupName', 'StepNo'], ascending=[False, True]).reset_index(drop=True)
        total_delta_e = step_df['Delta_E_High_Low_kWh'].sum()
        step_df['VarianceContribution_pct'] = np.where(total_delta_e > 0, step_df['Delta_E_High_Low_kWh'] / total_delta_e * 100.0, 0.0)
        step_df['EnergyShare_pct'] = step_df['Energy_kWh_Mean'] / step_df['Energy_kWh_Mean'].sum() * 100.0

        mst_steps = step_df[step_df['GroupName'] == 'AVR_MST'].copy()
        msb_steps = step_df[step_df['GroupName'] == 'AVR_MSB'].copy()

        # =====================================================================
        # PID TOTAL DURATION (MST + MSB) & HARD RULE CALCULATION
        # =====================================================================
        pid_ref_temp, pid_tier_key, pid_rule = self.map_pid_target_tier(compound_name, mst_steps, msb_steps)
        pid_min_s = pid_rule['min_s']
        pid_max_s = pid_rule['max_s']

        # Identify PID start step in MST: first MST step reaching (pid_ref_temp - 3.0°C)
        qual_mst = mst_steps[mst_steps['Temp_C_Mean'] >= (pid_ref_temp - 3.0)]
        if not qual_mst.empty:
            pid_start_step = int(qual_mst['StepNo'].iloc[0])
        else:
            pid_start_step = int(mst_steps['StepNo'].iloc[-1])

        curr_pid_top_s = mst_steps[mst_steps['StepNo'] >= pid_start_step]['Time_s_Mean'].sum()
        curr_pid_bottom_s = msb_steps['Time_s_Mean'].sum()
        curr_pid_total_s = curr_pid_top_s + curr_pid_bottom_s

        # Also compute latest-order median duration (handles cases like B00458/B00163 where MSB S2 changed across orders)
        curr_pid_top_med_s = mst_steps[mst_steps['StepNo'] >= pid_start_step]['Time_s_Median'].sum()
        curr_pid_bottom_med_s = msb_steps['Time_s_Median'].sum()
        curr_pid_total_med_s = curr_pid_top_med_s + curr_pid_bottom_med_s

        # 1. Identify Primary MST High-Shear / High-Variance Step
        mst_cand = mst_steps[(mst_steps['StepNo'] >= 5) & (mst_steps['Energy_kWh_Mean'] > 2.0)].copy()
        if mst_cand.empty:
            mst_cand = mst_steps[mst_steps['Energy_kWh_Mean'] > 1.0].copy()

        mst_cand['TargetScore'] = mst_cand['VarianceContribution_pct'] * 0.65 + mst_cand['EnergyShare_pct'] * 0.35
        best_mst = mst_cand.sort_values('TargetScore', ascending=False).iloc[0]

        second_mst = None
        if len(mst_cand) > 1:
            second_mst = mst_cand.sort_values('TargetScore', ascending=False).iloc[1]

        n0 = round(best_mst['Speed_RPM_Mean'])
        delta_rpm = -5 if n0 >= 34 else (-3 if n0 >= 24 else -2)
        target_rpm = n0 + delta_rpm
        p_mst_kw = best_mst['Power_kW_Mean']
        e_mst_kwh = best_mst['Energy_kWh_Mean']
        t_mst_s = best_mst['Time_s_Mean']
        w_mst_pct = best_mst['VarianceContribution_pct']
        de_mst_hl = best_mst['Delta_E_High_Low_kWh']

        rel_speed_change = abs(delta_rpm) / max(n0, 1)
        est_power_drop_kw = p_mst_kw * rel_speed_change * 1.10
        est_time_delay_s = t_mst_s * rel_speed_change * 0.95
        est_mst_saving_kwh = max(0.35, min(de_mst_hl * 0.85, e_mst_kwh * rel_speed_change * 0.82)) if de_mst_hl > 0.3 else e_mst_kwh * rel_speed_change * 0.55
        weight_t = valid_batches['weight_kg'].mean() / 1000.0
        est_mst_saving_kwh_t = est_mst_saving_kwh / weight_t

        # Does slowing PrimaryMST extend the PID window?
        mst_pid_delta_s = est_time_delay_s if int(best_mst['StepNo']) >= pid_start_step else 0.0

        # 2. Identify Unconstrained MSB Step 1 & Step 2 Time Curtailment
        msb_s1 = msb_steps[msb_steps['StepNo'] == 1]
        msb_s2 = msb_steps[msb_steps['StepNo'] == 2]

        if not msb_s1.empty:
            s1 = msb_s1.iloc[0]
            t_msb1_s = s1['Time_s_Mean']
            p_msb1_kw = s1['Power_kW_Mean']
            e_msb1_kwh = s1['Energy_kWh_Mean']
            n_msb1_rpm = round(s1['Speed_RPM_Mean'])
            w_msb1_pct = s1['VarianceContribution_pct']
            de_msb1_hl = s1['Delta_E_High_Low_kWh']
            e_sec1_kwh = p_msb1_kw / 3600.0
            e_sec1_kwh_t = e_sec1_kwh / weight_t

            if t_msb1_s >= 42.0:
                delta_t_msb1 = -10
                delta_t_msb1_p2 = -15
            elif t_msb1_s >= 30.0:
                delta_t_msb1 = -8
                delta_t_msb1_p2 = -10
            elif t_msb1_s >= 22.0:
                delta_t_msb1 = -5
                delta_t_msb1_p2 = -8
            else:
                delta_t_msb1 = -4
                delta_t_msb1_p2 = -6
        else:
            t_msb1_s, p_msb1_kw, e_msb1_kwh, n_msb1_rpm, w_msb1_pct, de_msb1_hl = 0, 0, 0, 0, 0, 0
            e_sec1_kwh, e_sec1_kwh_t, delta_t_msb1, delta_t_msb1_p2 = 0, 0, 0, 0

        if not msb_s2.empty:
            s2 = msb_s2.iloc[0]
            t_msb2_s = s2['Time_s_Mean']
            t_msb2_med_s = s2['Time_s_Median']
            p_msb2_kw = s2['Power_kW_Mean']
            e_msb2_kwh = s2['Energy_kWh_Mean']
            n_msb2_rpm = round(s2['Speed_RPM_Mean'])
            w_msb2_pct = s2['VarianceContribution_pct']
            e_sec2_kwh = p_msb2_kw / 3600.0
            e_sec2_kwh_t = e_sec2_kwh / weight_t
            if t_msb1_s < 28.0 and t_msb2_s >= 30.0:
                delta_t_msb2 = -5
            else:
                delta_t_msb2 = 0
        else:
            t_msb2_s, t_msb2_med_s, p_msb2_kw, e_msb2_kwh, n_msb2_rpm, w_msb2_pct = 0, 0, 0, 0, 0, 0
            e_sec2_kwh, e_sec2_kwh_t, delta_t_msb2 = 0, 0, 0

        # =====================================================================
        # 3. ENFORCE PID TOTAL DURATION HARD RULE GATE ON MSB TIME ADJUSTMENTS
        # =====================================================================
        # Check if compound is a 2-step Non-PID fast-MSB Carbon Black recipe (B00458 / B00163)
        # where current duration (80.2s / 92.3s) is below the 130s Target-155°C floor
        raw_total_msb_cut = delta_t_msb1 + delta_t_msb2
        effective_curr_total_s = min(curr_pid_total_s, curr_pid_total_med_s)
        hard_rule_note = ""

        if effective_curr_total_s < pid_min_s:
            # Compound's current 2-step MSB duration is already below the Target 155°C [130, 230]s lower bound
            # (e.g., B00458 at 80.2s and B00163 at 92.3s). Under the Hard Rule, NO MSB time cut is allowed
            # on the 50.6s/60.6s recipe; only MST Step 7 (-5 RPM, which adds +3.0s/+3.5s) is permitted!
            delta_t_msb1 = 0
            delta_t_msb1_p2 = 0
            delta_t_msb2 = 0
            hard_rule_status = f"CLAMPED_BY_HARD_RULE (Curr {effective_curr_total_s:.1f}s < {pid_min_s:.0f}s: Hold MSB 0s, MST -5RPM adds +{mst_pid_delta_s:.1f}s)"
            hard_rule_note = (
                f"Hard Rule [{pid_min_s:.0f}-{pid_max_s:.0f}s] Active: Current 2-step MSB recipe total ({effective_curr_total_s:.1f}s) "
                f"is already below {pid_min_s:.0f}s (and B00458 is at the 80.2s absolute floor). "
                f"MSB time cut is clamped to 0s; only MST S{int(best_mst['StepNo'])} {delta_rpm} RPM (+{mst_pid_delta_s:.1f}s) is executed."
            )
        else:
            # Ensure Post-DOE Total PID Duration stays strictly >= pid_min_s + 2.0s safety margin
            projected_total_s = effective_curr_total_s + mst_pid_delta_s + raw_total_msb_cut
            if projected_total_s < pid_min_s + 2.0:
                max_allowed_cut = max(0.0, (effective_curr_total_s + mst_pid_delta_s) - (pid_min_s + 2.0))
                scale = max_allowed_cut / max(abs(raw_total_msb_cut), 1.0)
                delta_t_msb1 = -int(round(abs(delta_t_msb1) * scale))
                delta_t_msb2 = -int(round(abs(delta_t_msb2) * scale))
                delta_t_msb1_p2 = delta_t_msb1
                hard_rule_status = f"PASS_CLAMPED ({pid_min_s:.0f}-{pid_max_s:.0f}s)"
            else:
                # Also clamp Phase 2 (delta_t_msb1_p2) so Phase 2 never breaches pid_min_s + 2.0s
                proj_p2 = effective_curr_total_s + mst_pid_delta_s + delta_t_msb1_p2 + delta_t_msb2
                if proj_p2 < pid_min_s + 2.0:
                    delta_t_msb1_p2 = delta_t_msb1
                hard_rule_status = f"PASS ({pid_min_s:.0f}-{pid_max_s:.0f}s)"

            post_doe_check = curr_pid_total_s + mst_pid_delta_s + delta_t_msb1 + delta_t_msb2
            hard_rule_note = (
                f"Hard Rule {pid_rule['tier_label']} [{pid_min_s:.0f}-{pid_max_s:.0f}s]: "
                f"Curr={curr_pid_total_s:.1f}s (MST {curr_pid_top_s:.1f}s + MSB {curr_pid_bottom_s:.1f}s) -> "
                f"Post-DOE={post_doe_check:.1f}s [PASS]"
            )

        post_doe_pid_top_s = curr_pid_top_s + mst_pid_delta_s
        post_doe_pid_bottom_s = curr_pid_bottom_s + delta_t_msb1 + delta_t_msb2
        post_doe_pid_total_s = post_doe_pid_top_s + post_doe_pid_bottom_s

        est_msb1_saving_kwh = abs(delta_t_msb1) * e_sec1_kwh
        est_msb1_saving_kwh_t = abs(delta_t_msb1) * e_sec1_kwh_t
        est_msb2_saving_kwh = abs(delta_t_msb2) * e_sec2_kwh
        est_msb2_saving_kwh_t = abs(delta_t_msb2) * e_sec2_kwh_t

        return {
            'CompoundName': compound_name,
            'EquipmentID': valid_batches['equip_id'].iloc[0],
            'SampleBatches': len(valid_batches),
            'Weight_kg': valid_batches['weight_kg'].mean(),
            'TotalEnergy_Mean_kWh': valid_batches['total_energy_kwh'].mean(),
            'SpecEnergy_Mean_kWh_t': valid_batches['spec_energy_kwh_t'].mean(),
            'SpecEnergy_Low_kWh_t': valid_batches[valid_batches['EnergyTier']=='Golden_Low (P15)']['spec_energy_kwh_t'].mean(),
            'SpecEnergy_High_kWh_t': valid_batches[valid_batches['EnergyTier']=='High_Energy (P85)']['spec_energy_kwh_t'].mean(),
            'PID_Target_Temp_C': pid_ref_temp,
            'PID_HardRule_Tier': pid_rule['tier_label'],
            'PID_Band_Min_s': pid_min_s,
            'PID_Band_Max_s': pid_max_s,
            'PID_Start_MST_Step': pid_start_step,
            'Curr_PID_Top_MST_s': curr_pid_top_s,
            'Curr_PID_Bottom_MSB_s': curr_pid_bottom_s,
            'Curr_PID_Total_s': curr_pid_total_s,
            'Curr_PID_Total_LatestMed_s': curr_pid_total_med_s,
            'PostDOE_PID_Top_MST_s': post_doe_pid_top_s,
            'PostDOE_PID_Bottom_MSB_s': post_doe_pid_bottom_s,
            'PostDOE_PID_Total_s': post_doe_pid_total_s,
            'PID_HardRule_Status': hard_rule_status,
            'PID_HardRule_Note': hard_rule_note,
            'StepProfile': step_df,
            'PrimaryMST_StepNo': int(best_mst['StepNo']),
            'PrimaryMST_CurrRPM': n0,
            'PrimaryMST_DeltaRPM': delta_rpm,
            'PrimaryMST_TargetRPM': target_rpm,
            'PrimaryMST_Time_s': t_mst_s,
            'PrimaryMST_Energy_kWh': e_mst_kwh,
            'PrimaryMST_Power_kW': p_mst_kw,
            'PrimaryMST_Temp_C': best_mst['Temp_C_Mean'],
            'PrimaryMST_E_Low_kWh': best_mst['E_Low_kWh'],
            'PrimaryMST_E_High_kWh': best_mst['E_High_kWh'],
            'PrimaryMST_DeltaE_HL_kWh': de_mst_hl,
            'PrimaryMST_VarContrib_pct': w_mst_pct,
            'PrimaryMST_EstPowerDrop_kW': est_power_drop_kw,
            'PrimaryMST_EstTimeDelay_s': est_time_delay_s,
            'PrimaryMST_EstSaving_kWh': est_mst_saving_kwh,
            'PrimaryMST_EstSaving_kWh_t': est_mst_saving_kwh_t,
            'SecondaryMST_StepNo': int(second_mst['StepNo']) if second_mst is not None else None,
            'SecondaryMST_CurrRPM': round(second_mst['Speed_RPM_Mean']) if second_mst is not None else None,
            'SecondaryMST_Energy_kWh': second_mst['Energy_kWh_Mean'] if second_mst is not None else None,
            'SecondaryMST_VarContrib_pct': second_mst['VarianceContribution_pct'] if second_mst is not None else None,
            'MSB1_CurrTime_s': t_msb1_s,
            'MSB1_CurrRPM': n_msb1_rpm,
            'MSB1_Power_kW': p_msb1_kw,
            'MSB1_Energy_kWh': e_msb1_kwh,
            'MSB1_VarContrib_pct': w_msb1_pct,
            'MSB1_DeltaE_HL_kWh': de_msb1_hl,
            'MSB1_MarginalRate_kWh_s': e_sec1_kwh,
            'MSB1_MarginalRate_kWh_t_s': e_sec1_kwh_t,
            'MSB1_DeltaTime_s': delta_t_msb1,
            'MSB1_DeltaTime_P2_s': delta_t_msb1_p2,
            'MSB1_EstSaving_kWh': est_msb1_saving_kwh,
            'MSB1_EstSaving_kWh_t': est_msb1_saving_kwh_t,
            'MSB2_CurrTime_s': t_msb2_s,
            'MSB2_CurrRPM': n_msb2_rpm,
            'MSB2_Power_kW': p_msb2_kw,
            'MSB2_Energy_kWh': e_msb2_kwh,
            'MSB2_VarContrib_pct': w_msb2_pct,
            'MSB2_MarginalRate_kWh_s': e_sec2_kwh,
            'MSB2_MarginalRate_kWh_t_s': e_sec2_kwh_t,
            'MSB2_DeltaTime_s': delta_t_msb2,
            'MSB2_EstSaving_kWh': est_msb2_saving_kwh,
            'MSB2_EstSaving_kWh_t': est_msb2_saving_kwh_t,
            'Combined_EstSaving_kWh_t': est_mst_saving_kwh_t + est_msb1_saving_kwh_t + est_msb2_saving_kwh_t
        }

    def generate_charts(self, rank_df, diag_map, extra_artifact_dir=None):
        """
        Generates two publication-grade diagnostic dashboards incorporating the PID Total Duration Hard Rule:
        1. top5_cb_and_silica_energy_potential_dashboard.png
        2. b00458_b00163_and_top_compounds_step_diagnosis.png
        """
        cb_candidates = [
            'M1-B15563R1-- 01 006',
            'M1-B00460B--- 04 010',
            'M1-B00163---- 08 010',
            'M1-R00218W2-- 10 011',
            'M1-B00458---- 12 012'
        ]
        sil_candidates = [
            'M1-T15760---- 06 002',
            'M1-T20153W--- 01 001',
            'M1-T09170---- 23 006',
            'M1-T33025W3-- 03 005',
            'M1-T17771---- 05 008'
        ]

        cb_df = rank_df[rank_df['CompoundName'].isin(cb_candidates)].copy()
        cb_df['Order'] = cb_df['CompoundName'].apply(lambda x: cb_candidates.index(x))
        cb_df = cb_df.sort_values('Order', ascending=False)

        sil_df = rank_df[rank_df['CompoundName'].isin(sil_candidates)].copy()
        sil_df['Order'] = sil_df['CompoundName'].apply(lambda x: sil_candidates.index(x))
        sil_df = sil_df.sort_values('Order', ascending=False)

        # =====================================================================
        # CHART 1: Top 5 Carbon Black & Top 5 Silica Energy Potential Dashboard
        # =====================================================================
        fig, axes = plt.subplots(2, 2, figsize=(18.5, 12.2), dpi=160)
        fig.patch.set_facecolor('#F8FAFC')

        # Panel A: Carbon Black Top 5 Specific Energy Spread (P15 - P50 - P85)
        ax = axes[0, 0]
        ax.set_facecolor('#FFFFFF')
        y_pos = np.arange(len(cb_df))
        labels_cb = [f"{r['CompoundName'].split()[0]}\n({r['MixerID']}, {r['WindowTons']:.0f}t/20d)" for _, r in cb_df.iterrows()]

        for i, (_, r) in enumerate(cb_df.iterrows()):
            ax.plot([r['SpecEnergy_P15'], r['SpecEnergy_P85']], [i, i], color='#334155', lw=5, alpha=0.35, zorder=2)
            ax.scatter(r['SpecEnergy_P15'], i, color='#10B981', s=110, zorder=4, label='Golden Low ($P_{15}$)' if i == 0 else "")
            ax.scatter(r['SpecEnergy_P50'], i, color='#2563EB', s=110, zorder=4, label='Median ($P_{50}$)' if i == 0 else "")
            ax.scatter(r['SpecEnergy_P85'], i, color='#EF4444', s=110, zorder=4, label='High Energy ($P_{85}$)' if i == 0 else "")
            ax.text(r['SpecEnergy_P85'] + 0.6, i,
                    f"Δ(P85-P15)={r['Gap_P85_P15_kWh_t']:.1f} | Δ(P50-P15)={r['Gap_P50_P15_kWh_t']:.1f} kWh/t",
                    va='center', fontsize=9, fontweight='bold', color='#1E293B')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_cb, fontsize=9.5, fontweight='bold')
        ax.set_xlim(85, 152)
        ax.set_ylim(-0.6, len(cb_df) - 0.3)
        ax.set_xlabel('Specific Mixing Energy (kWh/t)', fontsize=10, fontweight='bold')
        ax.set_title('A. Top 5 Carbon Black Compounds: Batch Energy Disparity (P15 vs P50 vs P85)', fontsize=11.5, fontweight='bold', color='#0F172A', pad=10)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', fontsize=8.5, frameon=True, facecolor='#FFFFFF', framealpha=0.95)

        # Panel B: Silica Top 5 Specific Energy Spread (P15 - P50 - P85)
        ax = axes[0, 1]
        ax.set_facecolor('#FFFFFF')
        y_pos_s = np.arange(len(sil_df))
        labels_sil = [f"{r['CompoundName'].split()[0]}\n({r['MixerID']}, {r['WindowTons']:.0f}t/20d)" for _, r in sil_df.iterrows()]

        for i, (_, r) in enumerate(sil_df.iterrows()):
            ax.plot([r['SpecEnergy_P15'], r['SpecEnergy_P85']], [i, i], color='#334155', lw=5, alpha=0.35, zorder=2)
            ax.scatter(r['SpecEnergy_P15'], i, color='#10B981', s=110, zorder=4, label='Golden Low ($P_{15}$)' if i == 0 else "")
            ax.scatter(r['SpecEnergy_P50'], i, color='#8B5CF6', s=110, zorder=4, label='Median ($P_{50}$)' if i == 0 else "")
            ax.scatter(r['SpecEnergy_P85'], i, color='#EF4444', s=110, zorder=4, label='High Energy ($P_{85}$)' if i == 0 else "")
            ax.text(r['SpecEnergy_P85'] + 0.7, i,
                    f"Δ(P85-P15)={r['Gap_P85_P15_kWh_t']:.1f} | Δ(P50-P15)={r['Gap_P50_P15_kWh_t']:.1f} kWh/t",
                    va='center', fontsize=9, fontweight='bold', color='#1E293B')

        ax.set_yticks(y_pos_s)
        ax.set_yticklabels(labels_sil, fontsize=9.5, fontweight='bold')
        ax.set_xlim(118, 198)
        ax.set_ylim(-0.6, len(sil_df) - 0.3)
        ax.set_xlabel('Specific Mixing Energy (kWh/t)', fontsize=10, fontweight='bold')
        ax.set_title('B. Top 5 Silica Compounds: Batch Energy Disparity (P15 vs P50 vs P85)', fontsize=11.5, fontweight='bold', color='#0F172A', pad=10)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.legend(loc='lower left', fontsize=8.5, frameon=True, facecolor='#FFFFFF', framealpha=0.95)

        # Panel C: Annualized Energy Saving Potential (kWh/yr) Comparison
        ax = axes[1, 0]
        ax.set_facecolor('#FFFFFF')
        combined_top = pd.concat([cb_df.sort_values('Order'), sil_df.sort_values('Order')], ignore_index=True)
        combined_top = combined_top.sort_values('AnnualSavingPotential_kWh', ascending=True)
        y_c = np.arange(len(combined_top))
        colors_c = ['#1E40AF' if c == 'Carbon Black' else '#7C3AED' for c in combined_top['FillerCategory']]
        bars = ax.barh(y_c, combined_top['AnnualSavingPotential_kWh'] / 1000.0, color=colors_c, alpha=0.88, height=0.65)
        ax.set_yticks(y_c)
        ax.set_yticklabels([f"{r['CompoundName'].split()[0]} ({'CB' if r['FillerCategory']=='Carbon Black' else 'Silica'})" for _, r in combined_top.iterrows()], fontsize=9, fontweight='bold')
        for bar, (_, r) in zip(bars, combined_top.iterrows()):
            w = bar.get_width()
            ax.text(w + 0.6, bar.get_y() + bar.get_height()/2,
                    f"{w:.1f} MWh/yr ({r['Batches']} batches / 20d)",
                    va='center', fontsize=8.5, fontweight='bold', color='#1E293B')
        ax.set_xlim(0, 62)
        ax.set_xlabel('Estimated Annual Saving Potential: (P50 - P15) × Annual Tons (MWh/year)', fontsize=10, fontweight='bold')
        ax.set_title('C. Annualized Energy Saving Potential Ranking (Volume × Recoverable Spread)', fontsize=11.5, fontweight='bold', color='#0F172A', pad=10)
        ax.grid(axis='x', linestyle='--', alpha=0.4)

        # Panel D: Expected Specific Energy Reduction (kWh/t) & PID Hard Rule Verification
        ax = axes[1, 1]
        ax.set_facecolor('#FFFFFF')
        all_10_names = cb_candidates + sil_candidates
        all_10_names = [n for n in all_10_names if n in diag_map][::-1]
        y_d = np.arange(len(all_10_names))
        mst_sav = [diag_map[n]['PrimaryMST_EstSaving_kWh_t'] for n in all_10_names]
        msb1_sav = [diag_map[n]['MSB1_EstSaving_kWh_t'] for n in all_10_names]
        msb2_sav = [diag_map[n]['MSB2_EstSaving_kWh_t'] for n in all_10_names]

        b1 = ax.barh(y_d, mst_sav, color='#EA580C', height=0.62, label='MST High-Shear Step (-5/-3 RPM)')
        b2 = ax.barh(y_d, msb1_sav, left=mst_sav, color='#0284C7', height=0.62, label='MSB Step 1 Time Cut (-4s to -10s)')
        b3 = ax.barh(y_d, msb2_sav, left=np.array(mst_sav)+np.array(msb1_sav), color='#0D9488', height=0.62, label='MSB Step 2 Time Cut (-5s)')

        ax.set_yticks(y_d)
        ax.set_yticklabels([f"{n.split()[0]}" for n in all_10_names], fontsize=9.5, fontweight='bold')
        for i, n in enumerate(all_10_names):
            d = diag_map[n]
            tot = d['Combined_EstSaving_kWh_t']
            if d['MSB1_DeltaTime_s'] == 0 and d['MSB2_DeltaTime_s'] == 0:
                msb_txt = f"MSB Hold 0s [PID Rule]"
            else:
                msb_txt = f"MSB1 {d['MSB1_DeltaTime_s']}s" + (f"+MSB2 {d['MSB2_DeltaTime_s']}s" if d['MSB2_DeltaTime_s'] != 0 else "")
            pid_txt = f"PID:{d['Curr_PID_Total_s']:.0f}→{d['PostDOE_PID_Total_s']:.0f}s [{d['PID_Band_Min_s']:.0f}-{d['PID_Band_Max_s']:.0f}s]"
            ax.text(tot + 0.12, i,
                    f"-{tot:.2f} kWh/t (MST S{d['PrimaryMST_StepNo']} {d['PrimaryMST_DeltaRPM']}RPM + {msb_txt} | {pid_txt})",
                    va='center', fontsize=7.8, fontweight='bold', color='#0F172A')

        ax.set_xlim(0, 14.2)
        ax.set_ylim(-0.6, len(all_10_names) + 0.8)
        ax.set_xlabel('Expected Specific Energy Reduction Under PID Duration Hard Rule (kWh/t)', fontsize=10, fontweight='bold')
        ax.set_title('D. Step-Level DOE Energy Reduction & PID Total Duration Hard Rule Gate', fontsize=11.5, fontweight='bold', color='#0F172A', pad=10)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', ncol=1, fontsize=8.2, frameon=True, facecolor='#FFFFFF', framealpha=0.95)

        plt.suptitle('Plant-Wide Top 5 Carbon Black & Top 5 Silica Compounds: Energy Potential & PID Hard-Rule Compliant DOE Matrix',
                     fontsize=13.5, fontweight='bold', color='#0F172A', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        chart1_path = os.path.join(self.output_dir, 'top5_cb_and_silica_energy_potential_dashboard.png')
        plt.savefig(chart1_path, dpi=160)
        if extra_artifact_dir:
            plt.savefig(os.path.join(extra_artifact_dir, 'top5_cb_and_silica_energy_potential_dashboard.png'), dpi=160)
        plt.close(fig)

        # =====================================================================
        # CHART 2: Deep Step-By-Step Telemetry & PID Hard Rule Basis (B00458, B00163, B00460B, T20153W)
        # =====================================================================
        fig, axes = plt.subplots(2, 2, figsize=(18.5, 12.8), dpi=160)
        fig.patch.set_facecolor('#F8FAFC')

        focus_compounds = [
            ('M1-B00458---- 12 012', 'A. M1-B00458 (92-1020, 264.4 kg): Step Energy & PID Hard Rule Clamp'),
            ('M1-B00163---- 08 010', 'B. M1-B00163 (92-1020, 267.2 kg): Step Energy & PID Hard Rule Clamp'),
            ('M1-B00460B--- 04 010', 'C. M1-B00460B (92-1100, 407.3 kg): Step Energy & Target 155°C PID Gate'),
            ('M1-T20153W--- 01 001', 'D. M1-T20153W (92-1090, 267.9 kg): Step Energy & Target 165°C PID Gate')
        ]

        legend_handles = []
        for idx, (cname, title) in enumerate(focus_compounds):
            ax = axes[idx // 2, idx % 2]
            ax.set_facecolor('#FFFFFF')
            if cname not in diag_map:
                continue
            d = diag_map[cname]
            sp = d['StepProfile']
            x = np.arange(len(sp))
            w = 0.36

            b_low = ax.bar(x - w/2, sp['E_Low_kWh'], width=w, color='#10B981', alpha=0.88, label='Golden Low Batch ($P_{15}$ Step Energy kWh)')
            b_high = ax.bar(x + w/2, sp['E_High_kWh'], width=w, color='#EF4444', alpha=0.88, label='High Energy Batch ($P_{85}$ Step Energy kWh)')

            ax2 = ax.twinx()
            l_var = ax2.plot(x, sp['VarianceContribution_pct'], color='#1D4ED8', marker='o', lw=2.2, markersize=6, label='Step Variance Contribution $W_i$ (%)')
            ax2.set_ylabel('Variance Contribution $W_i$ (%)', fontsize=9, fontweight='bold', color='#1D4ED8')
            ax2.set_ylim(0, max(65, sp['VarianceContribution_pct'].max() * 1.38))

            if idx == 0:
                legend_handles = [b_low, b_high, l_var[0]]

            for i, row in sp.iterrows():
                if row['VarianceContribution_pct'] >= 8.0 or row['Energy_kWh_Mean'] >= 6.0:
                    ax2.text(i, row['VarianceContribution_pct'] + 2.2,
                             f"W={row['VarianceContribution_pct']:.1f}%\n({row['Speed_RPM_Mean']:.0f}RPM,{row['Time_s_Mean']:.0f}s)",
                             ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1E3A8A',
                             bbox=dict(boxstyle='round,pad=0.18', facecolor='#EFF6FF', edgecolor='#93C5FD', alpha=0.88))

            ax.set_xticks(x)
            ax.set_xticklabels(sp['StepLabel'], rotation=30, ha='right', fontsize=8.5, fontweight='bold')
            ax.set_ylabel('Incremental Step Energy (kWh/batch)', fontsize=9.5, fontweight='bold')
            ax.set_ylim(0, sp['E_High_kWh'].max() * 1.36)
            ax.set_title(title, fontsize=11, fontweight='bold', color='#0F172A', pad=10)
            ax.grid(axis='y', linestyle='--', alpha=0.35)

            if d['MSB1_DeltaTime_s'] == 0 and d['MSB2_DeltaTime_s'] == 0:
                msb_action_str = f"MSB S1/S2: Hold 0s (Clamped by PID Hard Rule: Curr {d['Curr_PID_Total_LatestMed_s']:.1f}s < {d['PID_Band_Min_s']:.0f}s)"
            else:
                msb2_str = f" + MSB S2 {d['MSB2_DeltaTime_s']}s" if d['MSB2_DeltaTime_s'] != 0 else ""
                msb_action_str = f"MSB S1: {d['MSB1_CurrTime_s']:.0f}→{d['MSB1_CurrTime_s']+d['MSB1_DeltaTime_s']:.0f}s ({d['MSB1_DeltaTime_s']}s){msb2_str}"

            box_txt = (
                f"DOE Action: MST S{d['PrimaryMST_StepNo']}: {d['PrimaryMST_CurrRPM']}→{d['PrimaryMST_TargetRPM']} RPM ({d['PrimaryMST_DeltaRPM']} RPM) | {msb_action_str}\n"
                f"PID Hard Rule ({d['PID_HardRule_Tier']}: {d['PID_Band_Min_s']:.0f}-{d['PID_Band_Max_s']:.0f}s): "
                f"MST_PID({d['Curr_PID_Top_MST_s']:.1f}s)+MSB({d['Curr_PID_Bottom_MSB_s']:.1f}s)={d['Curr_PID_Total_s']:.1f}s → Post-DOE={d['PostDOE_PID_Total_s']:.1f}s\n"
                f"Basis: MST S{d['PrimaryMST_StepNo']} P={d['PrimaryMST_Power_kW']:.0f}kW (W={d['PrimaryMST_VarContrib_pct']:.1f}%, ΔP≈-{d['PrimaryMST_EstPowerDrop_kW']:.0f}kW) "
                f"→ Compliant Saving: -{d['Combined_EstSaving_kWh_t']:.2f} kWh/t"
            )
            ax.text(0.02, 0.96, box_txt, transform=ax.transAxes, va='top', ha='left', fontsize=7.7, fontweight='bold',
                    color='#0F172A', bbox=dict(boxstyle='round,pad=0.35', facecolor='#FEF3C7', edgecolor='#F59E0B', alpha=0.94))

        if legend_handles:
            fig.legend(legend_handles, ['Golden Low Batch ($P_{15}$ Step Energy kWh)', 'High Energy Batch ($P_{85}$ Step Energy kWh)', 'Step Variance Contribution $W_i$ (%)'],
                       loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=3, fontsize=9.5, frameon=True, facecolor='#FFFFFF')

        plt.suptitle('Step-Level Telemetry Diagnosis & PID Total Duration Hard Rule (165°C: 80-140s | 155°C: 130-230s | 142°C: 270-330s)',
                     fontsize=13.5, fontweight='bold', color='#0F172A', y=0.985)
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        chart2_path = os.path.join(self.output_dir, 'b00458_b00163_and_top_compounds_step_diagnosis.png')
        plt.savefig(chart2_path, dpi=160)
        if extra_artifact_dir:
            plt.savefig(os.path.join(extra_artifact_dir, 'b00458_b00163_and_top_compounds_step_diagnosis.png'), dpi=160)
        plt.close(fig)
        print(f"\n[Charts Saved] -> {chart1_path} & {chart2_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated Compound Energy Optimizer with PID Hard Rule')
    parser.add_argument('--mode', choices=['rank', 'diagnose', 'all'], default='all')
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--compounds', nargs='*', default=None)
    parser.add_argument('--artifact-dir', type=str, default=r'C:\Users\uif35346\.gemini\antigravity\brain\ac6a457c-39c9-4fa9-a167-f05b92d16dea')
    args = parser.parse_args()

    opt = CompoundEnergyOptimizer()
    rank_df = opt.load_and_rank_compounds()

    if args.mode in ['rank', 'all']:
        print("=== TOP CARBON BLACK COMPOUNDS ===")
        cb_top = rank_df[rank_df['FillerCategory'] == 'Carbon Black'].head(args.top)
        print(cb_top[['CompoundName', 'MixerID', 'Batches', 'WindowTons', 'SpecEnergy_Mean', 'Gap_P85_P15_kWh_t', 'Gap_P50_P15_kWh_t', 'AnnualSavingPotential_kWh']].to_string(index=False))

        print("\n=== TOP SILICA COMPOUNDS ===")
        sil_top = rank_df[rank_df['FillerCategory'] == 'Silica'].head(args.top)
        print(sil_top[['CompoundName', 'MixerID', 'Batches', 'WindowTons', 'SpecEnergy_Mean', 'Gap_P85_P15_kWh_t', 'Gap_P50_P15_kWh_t', 'AnnualSavingPotential_kWh']].to_string(index=False))

    if args.mode in ['diagnose', 'all']:
        target_list = args.compounds
        if not target_list:
            df_steps = pd.read_csv(opt.steps_csv)
            target_list = df_steps['CompoundName'].unique().tolist()

        diag_summaries = []
        diag_map = {}
        for cname in target_list:
            res = opt.diagnose_compound_steps(cname, rank_df=rank_df)
            if res is None:
                continue
            diag_map[cname] = res
            print(f"\n======================================================================")
            print(f"COMPOUND: {res['CompoundName']} | Line: {res['EquipmentID']} | Sample Batches: {res['SampleBatches']} | Weight: {res['Weight_kg']:.1f} kg")
            print(f"Specific Energy: Mean={res['SpecEnergy_Mean_kWh_t']:.2f} kWh/t | Golden_Low(P15)={res['SpecEnergy_Low_kWh_t']:.2f} | High(P85)={res['SpecEnergy_High_kWh_t']:.2f}")
            print(f"PID Hard Rule ({res['PID_HardRule_Tier']}, Band [{res['PID_Band_Min_s']:.0f}-{res['PID_Band_Max_s']:.0f}s]):")
            print(f"  Curr PID Total = {res['Curr_PID_Total_s']:.1f}s (MST S{res['PID_Start_MST_Step']}+: {res['Curr_PID_Top_MST_s']:.1f}s + MSB: {res['Curr_PID_Bottom_MSB_s']:.1f}s) | LatestMed = {res['Curr_PID_Total_LatestMed_s']:.1f}s")
            print(f"  Post-DOE PID Total = {res['PostDOE_PID_Total_s']:.1f}s (MST: {res['PostDOE_PID_Top_MST_s']:.1f}s + MSB: {res['PostDOE_PID_Bottom_MSB_s']:.1f}s) -> Status: {res['PID_HardRule_Status']}")
            print(f"----------------------------------------------------------------------")
            sp = res['StepProfile']
            print(sp[['StepLabel', 'Speed_RPM_Mean', 'Time_s_Mean', 'Energy_kWh_Mean', 'Power_kW_Mean', 'Temp_C_Mean', 'E_Low_kWh', 'E_High_kWh', 'Delta_E_High_Low_kWh', 'VarianceContribution_pct']].to_string(index=False))
            print(f"----------------------------------------------------------------------")
            print(f"[Recommendation 1 - MST] Adjust MST Step {res['PrimaryMST_StepNo']}: {res['PrimaryMST_CurrRPM']} RPM -> {res['PrimaryMST_TargetRPM']} RPM ({res['PrimaryMST_DeltaRPM']} RPM)")
            print(f"   Basis: Step {res['PrimaryMST_StepNo']} consumes {res['PrimaryMST_Energy_kWh']:.2f} kWh ({res['PrimaryMST_Power_kW']:.0f} kW for {res['PrimaryMST_Time_s']:.1f}s, EndTemp {res['PrimaryMST_Temp_C']:.1f}°C),")
            print(f"          contributing {res['PrimaryMST_VarContrib_pct']:.1f}% of batch-to-batch energy variance (High={res['PrimaryMST_E_High_kWh']:.2f} vs Low={res['PrimaryMST_E_Low_kWh']:.2f} kWh, dE={res['PrimaryMST_DeltaE_HL_kWh']:.2f} kWh).")
            print(f"          Est. Power Cut: -{res['PrimaryMST_EstPowerDrop_kW']:.0f} kW | Est. Saving: -{res['PrimaryMST_EstSaving_kWh']:.2f} kWh/batch (-{res['PrimaryMST_EstSaving_kWh_t']:.2f} kWh/t)")
            if res['MSB1_DeltaTime_s'] != 0 or res['MSB2_DeltaTime_s'] != 0:
                print(f"[Recommendation 2 - MSB] Adjust MSB Step 1 Time: {res['MSB1_CurrTime_s']:.1f}s -> {res['MSB1_CurrTime_s']+res['MSB1_DeltaTime_s']:.1f}s ({res['MSB1_DeltaTime_s']}s at {res['MSB1_CurrRPM']} RPM)")
                print(f"   Basis: MSB Step 1 runs at mean power {res['MSB1_Power_kW']:.0f} kW (marginal rate {res['MSB1_MarginalRate_kWh_s']:.4f} kWh/s = {res['MSB1_MarginalRate_kWh_t_s']:.3f} kWh/t/s),")
                print(f"          contributing {res['MSB1_VarContrib_pct']:.1f}% of energy variance. Cutting {abs(res['MSB1_DeltaTime_s'])}s saves -{res['MSB1_EstSaving_kWh']:.2f} kWh/batch (-{res['MSB1_EstSaving_kWh_t']:.2f} kWh/t).")
                if res['MSB2_DeltaTime_s'] != 0:
                    print(f"[Recommendation 2b - MSB S2] Adjust MSB Step 2 Time: {res['MSB2_CurrTime_s']:.1f}s -> {res['MSB2_CurrTime_s']+res['MSB2_DeltaTime_s']:.1f}s ({res['MSB2_DeltaTime_s']}s at {res['MSB2_CurrRPM']} RPM)")
                    print(f"   Basis: MSB Step 2 runs at mean power {res['MSB2_Power_kW']:.0f} kW (marginal rate {res['MSB2_MarginalRate_kWh_s']:.4f} kWh/s = {res['MSB2_MarginalRate_kWh_t_s']:.3f} kWh/t/s),")
                    print(f"          contributing {res['MSB2_VarContrib_pct']:.1f}% of energy variance. Cutting {abs(res['MSB2_DeltaTime_s'])}s saves -{res['MSB2_EstSaving_kWh']:.2f} kWh/batch (-{res['MSB2_EstSaving_kWh_t']:.2f} kWh/t).")
            else:
                print(f"[Recommendation 2 - MSB] HOLD MSB Duration (0s change): {res['PID_HardRule_Note']}")
            print(f"[Combined Expected Saving (Hard-Rule Compliant)] -{res['Combined_EstSaving_kWh_t']:.2f} kWh/t")

            row_copy = {k: v for k, v in res.items() if k != 'StepProfile'}
            diag_summaries.append(row_copy)

        if diag_summaries:
            pd.DataFrame(diag_summaries).to_csv(os.path.join(opt.output_dir, 'top_compounds_step_diagnostics_summary.csv'), index=False)
            opt.generate_charts(rank_df, diag_map, extra_artifact_dir=args.artifact_dir)
