# ----------------------------------------------------
# Mooney Prediction Pipeline V2.0 Path Bootstrap
# ----------------------------------------------------
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(PARENT_DIR)
sys.path.extend([
    PARENT_DIR,
    os.path.join(PARENT_DIR, 'data_processing'),
    os.path.join(PARENT_DIR, 'model_training'),
    os.path.join(PARENT_DIR, 'model_analysis'),
])
# ----------------------------------------------------

import os
import sys
import json

import numpy as np
import pandas as pd
import pyodbc

sys.path.insert(0, 'data processing')
from curve_segmenter_all_compounds import (
    MAX_VALID_VALUE,
    fill_anomalies_by_compound,
    find_bottom_mixer_end,
    find_bottom_mixer_start,
    find_ram_first_drop,
    find_temp_discharge_start,
    find_temp_pid_start,
    hex_to_series,
)
from new_compound_inference import predict_new_compound


OUTPUT_FEATURE_CSV = 'scratch/latest_sfe_order_features.csv'
OUTPUT_PREDICTION_CSV = 'scratch/latest_sfe_order_prediction.csv'


def connect_mms(database='SFEPLANT'):
    with open(os.path.join(PARENT_DIR, 'data_processing', 'credentials.json'), 'r', encoding='utf-8') as file:
        creds = json.load(file)
    c = creds['HF_MMS']
    conn_str = (
        f"DRIVER={c['driver']};"
        f"SERVER={c['server']};"
        f"DATABASE={database};"
        f"UID={c['username']};"
        f"PWD={c['password']};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)


def query_latest_order_rows(candidate_count=20):
    sql = f"""
    WITH latest_batches AS (
        SELECT TOP ({candidate_count})
            a.CompoundDescription,
            a.OrderID,
            a.CompoundName,
            a.OrderStartTime,
            bh.BatchNumber,
            bh.BatchWeight,
            bc.Curve1 AS temp,
            bc.Curve2 AS power,
            bc.Curve5 AS Torque,
            bc.Curve6 AS RotorSpeed,
            bc.Curve7 AS WayofRam
        FROM dbo.Orders a
        JOIN dbo.BatchHeader bh
            ON a.OrderID = bh.OrderID
        OUTER APPLY (
            SELECT TOP 1 Curve1, Curve2, Curve5, Curve6, Curve7
            FROM dbo.BatchCurve bc
            WHERE bc.OrderID = a.OrderID
              AND bc.BatchNumber = bh.BatchNumber
            ORDER BY bc.Timestamp DESC
        ) bc
        WHERE bc.Curve1 IS NOT NULL
          AND bc.Curve2 IS NOT NULL
          AND a.OrderID NOT LIKE '%H%'
          AND (a.CompoundName LIKE 'M1-%' OR a.CompoundName LIKE 'm1-%' OR a.CompoundName LIKE 'R1-%' OR a.CompoundName LIKE 'r1-%')
        ORDER BY a.OrderStartTime DESC, a.OrderID DESC, bh.BatchNumber DESC
    ),
    mat_pivot AS (
        SELECT
            om.OrderID,
            bh.BatchNumber,
            SUM(CASE WHEN om.MaterialCode LIKE 'CE%' AND om.MaterialCode NOT LIKE 'CE19%'
                THEN CAST(om.BatchWeight AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 END) AS weight_pct_solid_elastomer,
            SUM(CASE WHEN om.MaterialCode LIKE 'CN%'
                THEN CAST(om.BatchWeight AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 END) AS weight_pct_natural_rubber,
            SUM(CASE WHEN om.MaterialCode LIKE 'CS100%'
                THEN CAST(om.BatchWeight AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 END) AS weight_pct_silica,
            SUM(CASE WHEN om.MaterialCode LIKE 'CS%' AND om.MaterialCode NOT LIKE 'CS100%'
                THEN CAST(om.BatchWeight AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 END) AS weight_pct_oil,
            SUM(CASE WHEN om.MaterialCode LIKE 'CA551%'
                THEN CAST(om.BatchWeight AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 END) AS weight_pct_silian,
            SUM(CASE WHEN om.MaterialCode LIKE 'CC%'
                THEN CAST(om.BatchWeight AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 END) AS weight_pct_carbon_black
        FROM dbo.OrderMaterials om
        JOIN dbo.BatchHeader bh
            ON om.OrderID = bh.OrderID
        WHERE om.OrderID IN (SELECT OrderID FROM latest_batches)
        GROUP BY om.OrderID, bh.BatchNumber
    )
    SELECT
        lb.CompoundDescription,
        lb.OrderID,
        lb.CompoundName,
        lb.OrderStartTime,
        oil.StepNo AS StepNumber,
        oil.Value AS CurrentValue,
        oil.PrevStepValue,
        lb.BatchNumber,
        d.ParameterValue AS Top_Fill_Factor,
        d2.ParameterValue AS Bot_Fill_Factor,
        d3.ParameterValue AS Target_Temperature,
        lb.temp,
        lb.power,
        lb.Torque,
        lb.RotorSpeed,
        lb.WayofRam,
        m.weight_pct_solid_elastomer,
        m.weight_pct_natural_rubber,
        m.weight_pct_silica,
        m.weight_pct_oil,
        m.weight_pct_silian,
        m.weight_pct_carbon_black
    FROM latest_batches lb
    OUTER APPLY (
        SELECT TOP 1
            bd.StepNo,
            bd.Value,
            bd_prev.Value AS PrevStepValue
        FROM dbo.BatchData bd
        LEFT JOIN dbo.BatchData bd_prev
            ON bd_prev.OrderID = bd.OrderID
           AND bd_prev.BatchNumber = bd.BatchNumber
           AND bd_prev.StepNo = bd.StepNo - 1
           AND bd_prev.VariablePath = 'SCP-1-Step-Time-rel-s'
           AND bd_prev.GroupName = 'AVR_MST'
        WHERE bd.OrderID = lb.OrderID
          AND bd.BatchNumber = lb.BatchNumber
          AND bd.VariablePath = 'SCP-1-Step-Time-rel-s'
          AND bd.GroupName = 'AVR_MST'
          AND EXISTS (
              SELECT 1
              FROM dbo.RecipeMaterials rm
              WHERE rm.RecipeID = lb.CompoundDescription
                AND rm.StepNumber = bd.StepNo
                AND rm.MaterialCode LIKE '%CS4%'
          )
    ) oil
    LEFT JOIN dbo.RecipeCBS3Parameters d
        ON lb.CompoundDescription = d.RecipeID
       AND d.ParameterID = 'MI.1.RHT.Fill-Factor'
    LEFT JOIN dbo.RecipeCBS3Parameters d2
        ON lb.CompoundDescription = d2.RecipeID
       AND d2.ParameterID = 'MI.1.RHB.Fill-Factor'
    LEFT JOIN dbo.RecipeCBS3Parameters d3
        ON lb.CompoundDescription = d3.RecipeID
       AND d3.ParameterID = 'MI.1.RHT.Target-Temperature'
    LEFT JOIN mat_pivot m
        ON m.OrderID = lb.OrderID
       AND m.BatchNumber = lb.BatchNumber
    ORDER BY lb.OrderStartTime DESC, lb.OrderID DESC, lb.BatchNumber DESC
    """
    with connect_mms('SFEPLANT') as conn:
        return pd.read_sql(sql, conn)


def query_silica_phr(recipe_id):
    sql = """
    SELECT SUM(CAST(Pphr AS FLOAT)) AS silica_phr
    FROM dbo.RecipeMaterials
    WHERE RecipeID = ?
      AND (MaterialCode LIKE 'CS10%' OR MaterialCode LIKE 'CS12%')
    """
    with connect_mms('SFEPLANT') as conn:
        value = pd.read_sql(sql, conn, params=[recipe_id]).iloc[0]['silica_phr']
    return 0.0 if pd.isna(value) else float(value)


def extract_features_for_row(row):
    temp = hex_to_series(row['temp'])
    power = hex_to_series(row['power'])
    torque = hex_to_series(row['Torque'])
    speed = hex_to_series(row['RotorSpeed'])
    ram = hex_to_series(row['WayofRam'])

    tp_len = min(len(temp), len(power))
    if tp_len < 10:
        raise ValueError('Latest order has too short or missing temp/power curves.')

    temp = temp[:tp_len]
    power = power[:tp_len]
    torque_full = np.full(tp_len, np.nan)
    speed_full = np.full(tp_len, np.nan)
    ram_full = np.full(tp_len, np.nan)
    torque_full[:min(len(torque), tp_len)] = torque[:min(len(torque), tp_len)]
    speed_full[:min(len(speed), tp_len)] = speed[:min(len(speed), tp_len)]
    ram_full[:min(len(ram), tp_len)] = ram[:min(len(ram), tp_len)]
    torque = torque_full
    speed = speed_full
    ram = ram_full

    decoded = [{
        'row_data': row.to_dict(),
        'OrderID': row['OrderID'],
        'BatchNumber': row['BatchNumber'],
        'Compound': row['CompoundName'],
        'Target_Temperature': float(row['Target_Temperature']) if pd.notna(row.get('Target_Temperature')) else 140.0,
        'temp': temp,
        'power': power,
        'Torque': torque,
        'RotorSpeed': speed,
        'WayofRam': ram,
        'PrevStepValue': row.get('PrevStepValue'),
        'CurrentValue': row.get('CurrentValue'),
    }]

    for d in decoded:
        for col in ['temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam']:
            arr = d[col]
            arr[arr > MAX_VALID_VALUE] = np.nan
    for col in ['temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam']:
        fill_anomalies_by_compound(decoded, col)

    d = decoded[0]
    temp = d['temp']
    power = d['power']
    torque = d['Torque']
    speed = d['RotorSpeed']
    ram = d['WayofRam']
    target_temp = d['Target_Temperature']
    total_len = len(temp)
    w_oil = float(d['row_data'].get('weight_pct_oil', 0.0)) if pd.notna(d['row_data'].get('weight_pct_oil')) else 0.0
    has_oil = pd.notna(d['PrevStepValue']) and pd.notna(d['CurrentValue']) and w_oil > 0.0

    t1 = find_ram_first_drop(ram)
    if has_oil:
        is_oil_loading_present = 1.0
        t2 = int(round(float(d['PrevStepValue'])))
        t3 = int(round(float(d['CurrentValue'])))
        if t2 >= total_len - 15:
            t2 = total_len - 15
        if t3 >= total_len - 10:
            t3 = total_len - 10
        if t2 <= t1:
            t2 = t1 + 5
        if t3 <= t2:
            t3 = t2 + 5
        t4 = find_temp_pid_start(temp, target_temp, t3)
        if t4 <= t3:
            t4 = t3 + 5
        t5 = find_temp_discharge_start(temp, t4)
        if t5 <= t4:
            t5 = t4 + 5
        t_bot_start = find_bottom_mixer_start(temp, t5)
        if t5 > total_len:
            t5 = total_len
        if t4 > t5 - 2:
            t4 = t5 - 2
        if t3 > t4 - 2:
            t3 = t4 - 2
        if t2 > t3 - 2:
            t2 = t3 - 2
        if t1 > t2 - 2:
            t1 = t2 - 2
        t1 = max(t1, 0)
        t2 = max(t2, t1)
        t3 = max(t3, t2)
        t4 = max(t4, t3)
        t5 = max(t5, t4)
        t_bot_start = min(max(t_bot_start, t5), total_len)
        t_bot_end = find_bottom_mixer_end(temp, power, t_bot_start)
        if t_bot_end <= t_bot_start or t_bot_end > total_len:
            t_bot_end = total_len
        stages = {
            'Stage1_Loading': (0, t1),
            'Stage2_DryMixing': (t1, t2),
            'Stage3_OilLoading': (t2, t3),
            'Stage4_WetMixing': (t3, t4),
            'Stage5_PID': (t4, t5),
            'Stage6_BottomMixing': (t_bot_start, t_bot_end),
        }
    else:
        is_oil_loading_present = 0.0
        t4 = find_temp_pid_start(temp, target_temp, t1)
        if t4 <= t1:
            t4 = t1 + 5
        t5 = find_temp_discharge_start(temp, t4)
        if t5 <= t4:
            t5 = t4 + 5
        t_bot_start = find_bottom_mixer_start(temp, t5)
        if t5 > total_len:
            t5 = total_len
        if t4 > t5 - 2:
            t4 = t5 - 2
        if t1 > t4 - 2:
            t1 = t4 - 2
        t1 = max(t1, 0)
        t4 = max(t4, t1)
        t5 = max(t5, t4)
        t_bot_start = min(max(t_bot_start, t5), total_len)
        t_bot_end = find_bottom_mixer_end(temp, power, t_bot_start)
        if t_bot_end <= t_bot_start or t_bot_end > total_len:
            t_bot_end = total_len
        t2 = t1
        t3 = t1
        stages = {
            'Stage1_Loading': (0, t1),
            'Stage2_DryMixing': (t1, t4),
            'Stage3_OilLoading': (t1, t1),
            'Stage4_WetMixing': (t1, t1),
            'Stage5_PID': (t4, t5),
            'Stage6_BottomMixing': (t_bot_start, t_bot_end),
        }

    features = d['row_data'].copy()
    for col in ['temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam']:
        features.pop(col, None)
    features['batch_information_fk_final'] = f"{features['OrderID']}_{int(features['BatchNumber']):04d}"
    features['silica_phr'] = query_silica_phr(features['CompoundDescription'])
    features.update({
        'is_oil_loading_present': is_oil_loading_present,
        'idx_t1_Loading': t1,
        'idx_t2_DryMixing': t2,
        'idx_t3_OilLoading': t3,
        'idx_t4_PID_Start': t4,
        'idx_t5_Discharge': t5,
        'idx_t6_BottomMixing_Start': t_bot_start,
        'idx_t7_BottomMixing_End': t_bot_end,
        'idx_total_length': total_len,
    })

    for stage_name, (s, e) in stages.items():
        s_idx = int(s)
        e_idx = int(min(e, total_len))
        duration = e_idx - s_idx
        features[f'{stage_name}_Duration'] = duration
        if duration <= 0:
            for col in ['temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam', 'eta_torque']:
                features[f'{stage_name}_{col}_Mean'] = np.nan
                features[f'{stage_name}_{col}_Std'] = np.nan
            features[f'{stage_name}_RotorSpeed_Integral'] = np.nan
            features[f'{stage_name}_Torque_Integral'] = np.nan
            features[f'{stage_name}_power_Integral'] = np.nan
            continue
        seg_temp = temp[s_idx:e_idx]
        seg_power = power[s_idx:e_idx]
        seg_torque = torque[s_idx:e_idx]
        seg_speed = speed[s_idx:e_idx]
        seg_ram = ram[s_idx:e_idx]
        speed_nonzero_seg = np.where(seg_speed > 0, seg_speed, np.nan)
        seg_eta_torque = seg_torque / speed_nonzero_seg
        for col_name, arr in [('temp', seg_temp), ('power', seg_power), ('Torque', seg_torque), ('RotorSpeed', seg_speed), ('WayofRam', seg_ram), ('eta_torque', seg_eta_torque)]:
            valid = arr[~np.isnan(arr)]
            features[f'{stage_name}_{col_name}_Mean'] = np.nanmean(arr) if len(valid) > 0 else np.nan
            features[f'{stage_name}_{col_name}_Std'] = np.nanstd(arr) if len(valid) > 0 else np.nan
        for col_name, arr in [('RotorSpeed', seg_speed), ('Torque', seg_torque), ('power', seg_power)]:
            valid = arr[~np.isnan(arr)]
            features[f'{stage_name}_{col_name}_Integral'] = np.trapezoid(valid, dx=1.0) if len(valid) > 1 else (valid[0] if len(valid) == 1 else 0.0)

    discharge_temp = temp[int(t5)] if t5 < len(temp) else temp[-1]
    max_temp = np.nanmax(temp)
    t_max_temp = np.nanargmax(temp)
    init_temp = temp[int(t1)] if t1 < len(temp) else temp[0]
    features.update({
        'phys_discharge_temp': discharge_temp,
        'phys_max_temp': max_temp,
        'phys_t_max_temp': t_max_temp,
        'phys_init_temp': init_temp,
        'phys_temp_rise_rate': (discharge_temp - init_temp) / total_len if total_len > 0 else np.nan,
        'phys_temp_integral': np.trapezoid(temp[~np.isnan(temp)], dx=1.0),
        'phys_temp_change_rate_std': np.nanstd(np.diff(temp)),
        'phys_avg_power': np.nanmean(power),
        'phys_power_integral': np.trapezoid(power[~np.isnan(power)], dx=1.0),
        'phys_peak_power': np.nanmax(power),
        'phys_t_peak_power': np.nanargmax(power),
        'phys_max_power_drop_rate': -np.nanmin(np.diff(power)),
        'time_total_mixing': total_len,
        'time_pct_Loading': t1 / total_len,
        'time_pct_DryMixing': (t2 - t1) / total_len if has_oil else (t4 - t1) / total_len,
        'time_pct_OilLoading': (t3 - t2) / total_len if has_oil else 0.0,
        'time_pct_WetMixing': (t4 - t3) / total_len if has_oil else 0.0,
        'time_pct_PID': (t5 - t4) / total_len,
        'time_pct_Discharge': (total_len - t5) / total_len,
        'time_reach_discharge': t5,
        'phys_shear_history_total': np.trapezoid(speed[~np.isnan(speed)], dx=1.0) if len(speed[~np.isnan(speed)]) > 1 else 0.0,
        'setting_ram_lifts_count': np.sum(np.diff(ram) > 10),
        'setting_stable_ram_discharge': np.nanmean(ram[int(t5):]) if len(ram[int(t5):][~np.isnan(ram[int(t5):])]) > 0 else np.nan,
    })

    pid_power = power[int(t4):int(t5)]
    features['phys_power_stability_pid'] = np.nanstd(pid_power) / np.nanmean(pid_power) if len(pid_power[~np.isnan(pid_power)]) > 2 and np.nanmean(pid_power) != 0 else np.nan
    speed_nonzero = np.where(speed > 0, speed, np.nan)
    eta_app = power / (speed_nonzero ** 2)
    features['phys_eta_app_overall_mean'] = np.nanmean(eta_app)
    features['phys_eta_app_wetmix_mean'] = np.nanmean(eta_app[int(t3):int(t4)]) if has_oil and t4 > t3 else np.nan
    features['phys_eta_app_pid_mean'] = np.nanmean(eta_app[int(t4):int(t5)]) if t5 > t4 else np.nan
    features['phys_eta_app_discharge'] = eta_app[int(t5)] if t5 < len(eta_app) else np.nan

    w_solid = float(features.get('weight_pct_solid_elastomer', 0.0)) if pd.notna(features.get('weight_pct_solid_elastomer')) else 0.0
    w_nr = float(features.get('weight_pct_natural_rubber', 0.0)) if pd.notna(features.get('weight_pct_natural_rubber')) else 0.0
    w_cb = float(features.get('weight_pct_carbon_black', 0.0)) if pd.notna(features.get('weight_pct_carbon_black')) else 0.0
    w_silica = float(features.get('weight_pct_silica', 0.0)) if pd.notna(features.get('weight_pct_silica')) else 0.0
    total_rubber = w_solid + w_nr
    total_filler = w_cb + w_silica
    features['ratio_nr_rubber'] = w_nr / total_rubber if total_rubber > 0 else 0.0
    features['ratio_filler_polymer'] = total_filler / total_rubber if total_rubber > 0 else 0.0
    features['ratio_oil_polymer'] = w_oil / total_rubber if total_rubber > 0 else 0.0
    features['ratio_oil_filler'] = w_oil / total_filler if total_filler > 0 else 0.0

    s1_start, s1_end = stages['Stage1_Loading']
    s1_power = power[int(s1_start):int(s1_end)]
    s1_torque = torque[int(s1_start):int(s1_end)]
    features['Stage1_power_Max'] = np.nanmax(s1_power) if len(s1_power[~np.isnan(s1_power)]) > 0 else np.nan
    features['Stage1_Torque_Max'] = np.nanmax(s1_torque) if len(s1_torque[~np.isnan(s1_torque)]) > 0 else np.nan
    s2_start, s2_end = stages['Stage2_DryMixing']
    s2_power = power[int(s2_start):int(s2_end)]
    s2_valid = s2_power[~np.isnan(s2_power)]
    features['Stage2_power_decay_slope'] = np.polyfit(np.arange(len(s2_valid)), s2_valid, 1)[0] if len(s2_valid) > 3 else np.nan
    s2_torque = torque[int(s2_start):int(s2_end)]
    s2_speed = speed[int(s2_start):int(s2_end)]
    s2_eta = s2_torque / np.where(s2_speed > 0, s2_speed, np.nan)
    s2_eta_valid = s2_eta[~np.isnan(s2_eta)]
    features['Stage2_eta_torque_End'] = s2_eta_valid[-1] if len(s2_eta_valid) > 0 else np.nan

    total_solids = w_solid + w_nr + w_cb + w_silica
    for stage_name in stages.keys():
        power_integral = features.get(f'{stage_name}_power_Integral', 0.0)
        features[f'{stage_name}_Specific_Energy'] = power_integral / total_solids if total_solids > 0 and pd.notna(power_integral) else 0.0
    temp_above_100 = np.maximum(0.0, temp - 100.0)
    features['phys_temp_integral_above_100'] = np.trapezoid(temp_above_100[~np.isnan(temp_above_100)], dx=1.0)

    return features


def main():
    os.makedirs('scratch', exist_ok=True)
    latest_rows = query_latest_order_rows()
    if latest_rows.empty:
        raise RuntimeError('No recent SFEPLANT order with BatchCurve was found.')

    last_error = None
    feature_row = None
    selected_source_row = None
    for _, row in latest_rows.iterrows():
        try:
            feature_row = extract_features_for_row(row)
            selected_source_row = row
            break
        except Exception as exc:
            last_error = exc
            continue
    if feature_row is None:
        raise RuntimeError(f'Could not extract features from recent candidate orders: {last_error}')

    feature_df = pd.DataFrame([feature_row])
    feature_df.to_csv(OUTPUT_FEATURE_CSV, index=False, encoding='utf-8-sig')
    is_oil = float(feature_df.iloc[0]['is_oil_loading_present']) == 1.0
    bundle = os.path.join(PARENT_DIR, 'models', 'results_with_oil/mooney_model_bundle.joblib') if is_oil else os.path.join(PARENT_DIR, 'models', 'results_without_oil/mooney_model_bundle.joblib')
    if not os.path.exists(bundle):
        raise FileNotFoundError(f'Model bundle not found: {bundle}. Run train_mooney_models.py first.')

    predict_new_compound(bundle, OUTPUT_FEATURE_CSV, OUTPUT_PREDICTION_CSV)
    print('\nLatest predicted batch:')
    print(f"  OrderID: {selected_source_row['OrderID']}")
    print(f"  BatchNumber: {selected_source_row['BatchNumber']}")
    print(f"  CompoundName: {selected_source_row['CompoundName']}")
    print(f"  OrderStartTime: {selected_source_row['OrderStartTime']}")
    print(f"  Model bundle: {bundle}")
    pred = pd.read_csv(OUTPUT_PREDICTION_CSV).iloc[0]
    print(f"  Predicted MNY (Base): {pred['predicted_MNY_base']:.3f}")
    if 'predicted_MNY_calibrated' in pred:
        cal_mny = pred['predicted_MNY_calibrated']
        bias = pred.get('few_shot_bias_correction', 0.0)
        source = pred.get('bias_correction_source', 'None')
        print(f"  Predicted MNY (Calibrated): {cal_mny:.3f} (Applied Bias: {bias:+.3f} from {source})")
    print(f"  Reliability: {pred['applicability_reliability']}")
    print(f"  Applicability distance: {pred['applicability_distance']:.3f}")
    print(f"  Out-of-reference feature count: {int(pred['out_of_reference_range_feature_count'])}")
    print(f"  Feature CSV: {OUTPUT_FEATURE_CSV}")
    print(f"  Prediction CSV: {OUTPUT_PREDICTION_CSV}")


if __name__ == '__main__':
    main()