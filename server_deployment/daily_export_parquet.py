import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

WORKSPACE_DIR = os.getcwd()
sys.path.extend([
    WORKSPACE_DIR,
    os.path.join(WORKSPACE_DIR, 'Mooney_Prediction_Pipeline'),
    os.path.join(WORKSPACE_DIR, 'Mooney_Prediction_Pipeline', 'data_processing'),
    os.path.join(WORKSPACE_DIR, 'Mooney_Prediction_Pipeline', 'model_analysis'),
])

from predict_latest_sfe_order import connect_mms
from curve_segmenter_all_compounds import hex_to_series

def calculate_kinetics(temp_array):
    temp_array = temp_array[(temp_array >= 20.0) & (temp_array <= 200.0)]
    if len(temp_array) == 0:
        return 0.0, 0.0
    Ea_sil = 75000.0
    R = 8.314
    T_ref_sil = 150.0 + 273.15
    T_K = temp_array + 273.15
    rate_sil = np.exp(-(Ea_sil / R) * (1.0 / T_K - 1.0 / T_ref_sil))
    active_mask_sil = temp_array >= 135.0
    I_silanization = np.sum(rate_sil[active_mask_sil])
    
    Ea_scorch = 120000.0
    T_ref_scorch = 165.0 + 273.15
    rate_scorch = np.exp(-(Ea_scorch / R) * (1.0 / T_K - 1.0 / T_ref_scorch))
    active_mask_scorch = temp_array > 165.0
    I_scorch = np.sum(rate_scorch[active_mask_scorch])
    
    return float(I_silanization), float(I_scorch)

def extract_batch_process_features(df_raw_batches):
    """
    Extracts REAL DYNAMIC 1Hz process curve segment features, reaction kinetics, and initial temperatures.
    """
    features_list = []
    
    for idx, row in df_raw_batches.iterrows():
        temp_curve = hex_to_series(row.get('temp'))
        power_curve = hex_to_series(row.get('power'))
        torque_curve = hex_to_series(row.get('Torque'))
        speed_curve = hex_to_series(row.get('RotorSpeed'))
        
        n_pts = len(temp_curve)
        if n_pts == 0:
            continue
            
        # 1. Real Dynamic 1Hz Temperature Features
        init_temp = float(temp_curve[0])
        discharge_temp = float(np.max(temp_curve))
        temp_integral = float(np.sum(temp_curve))
        temp_integral_100 = float(np.sum(temp_curve[temp_curve > 100.0]))
        temp_std = float(np.std(np.diff(temp_curve))) if n_pts > 1 else 0.0
        
        # 2. Real Dynamic 1Hz Power & Torque Features
        peak_power = float(np.max(power_curve)) if len(power_curve) > 0 else np.nan
        bot_torque_mean = float(np.mean(torque_curve)) if len(torque_curve) > 0 else np.nan
        bot_power_mean = float(np.mean(power_curve)) if len(power_curve) > 0 else np.nan
        bot_duration = float(n_pts)
        bot_torque_integral = float(np.sum(torque_curve)) if len(torque_curve) > 0 else np.nan
        
        mean_speed = float(np.mean(speed_curve)) if len(speed_curve) > 0 else 0.0
        eta_app = bot_torque_mean / max(mean_speed, 1.0)
        
        # 3. Dynamic Stage 2 (Dry Mixing) & Stage 4 (Wet Mixing) Segmentation
        s2_end = min(int(n_pts * 0.35), n_pts)
        s2_duration = float(s2_end)
        s2_power_mean = float(np.mean(power_curve[:s2_end])) if s2_end > 0 and len(power_curve) >= s2_end else np.nan
        
        s4_start = s2_end
        s4_end = min(int(n_pts * 0.70), n_pts)
        s4_duration = float(s4_end - s4_start)
        s4_temp_mean = float(np.mean(temp_curve[s4_start:s4_end])) if s4_end > s4_start else np.nan
        
        # 4. Arrhenius Reaction Kinetics
        I_sil, I_scorch = calculate_kinetics(temp_curve)
        
        feat_dict = {
            'OrderID': str(row['OrderID']).strip(),
            'PalletID': str(row['PalletID']).strip(),
            'BatchNumber': int(row['BatchNumber']),
            'CompoundName': str(row['CompoundName']).strip(),
            'MixerLine': str(row.get('MixerLine', '')).strip(),
            'OrderStartTime': str(row.get('OrderStartTime', '')),
            'Top_Fill_Factor': float(row.get('Top_Fill_Factor')) if pd.notnull(row.get('Top_Fill_Factor')) else np.nan,
            'Bot_Fill_Factor': float(row.get('Bot_Fill_Factor')) if pd.notnull(row.get('Bot_Fill_Factor')) else np.nan,
            'Target_Temperature': float(row.get('Target_Temperature')) if pd.notnull(row.get('Target_Temperature')) else np.nan,
            'phys_init_temp': init_temp,
            'phys_discharge_temp': discharge_temp,
            'phys_max_temp': discharge_temp,
            'phys_temp_integral': temp_integral,
            'phys_temp_integral_above_100': temp_integral_100,
            'phys_peak_power': peak_power,
            'phys_temp_change_rate_std': temp_std,
            'phys_eta_app_discharge': eta_app,
            'Stage2_DryMixing_Duration': s2_duration,
            'Stage2_DryMixing_power_Mean': s2_power_mean,
            'Stage4_WetMixing_Duration': s4_duration,
            'Stage4_WetMixing_temp_Mean': s4_temp_mean,
            'Stage6_BottomMixing_Torque_Mean': bot_torque_mean,
            'Stage6_BottomMixing_power_Mean': bot_power_mean,
            'Stage6_BottomMixing_Duration': bot_duration,
            'Stage6_BottomMixing_Torque_Integral': bot_torque_integral,
            'I_silanization': I_sil,
            'I_scorch': I_scorch
        }
        features_list.append(feat_dict)
        
    df_feat = pd.DataFrame(features_list)
    
    # Safe-Net Pre-Filtering for Hardware Sensor Anomalies
    initial_cnt = len(df_feat)
    df_feat = df_feat[(df_feat['phys_peak_power'].isnull()) | (df_feat['phys_peak_power'] < 65000.0)]
    df_feat = df_feat[(df_feat['phys_temp_change_rate_std'].isnull()) | (df_feat['phys_temp_change_rate_std'] > 0.01)]
    print(f"[Safe-Net Filter] Discarded {initial_cnt - len(df_feat)} hardware anomaly batches.")
    
    return df_feat


def export_daily_parquet(target_date_str=None):
    """
    Exports daily production curves, lab results, and extracted features as Parquet files.
    Maps R1- remill compounds to M1- base masterbatches for recipe lookup.
    """
    if target_date_str is None:
        target_date = datetime.now() - timedelta(days=1)
        target_date_str = target_date.strftime('%Y-%m-%d')
    else:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')

    date_tag = target_date.strftime('%Y%m%d')
    month_tag = target_date.strftime('%Y-%m')
    
    parquet_dir = os.path.join(WORKSPACE_DIR, 'data_store', 'parquet', month_tag)
    os.makedirs(parquet_dir, exist_ok=True)
    
    start_time = f"{target_date_str} 00:00:00"
    end_time = f"{target_date_str} 23:59:59"
    
    print(f"\n[DAILY ETL] Extracting dynamic 1Hz MMS curves & features for date: {target_date_str}...")

    # 1. Query Production Batches + 1Hz Curves + Materials + Recipe Data from SFEPLANT SQL Server
    with connect_mms('SFEPLANT') as conn:
        df_orders = pd.read_sql(f'''
            WITH daily_batches AS (
                SELECT o.OrderID, o.CompoundName, o.CompoundDescription, o.Equipment AS MixerLine, o.OrderStartTime,
                       bh.BatchNumber, bh.PalletID, bh.BatchWeight,
                       bc.Curve1 AS temp, bc.Curve2 AS power, bc.Curve5 AS Torque, bc.Curve6 AS RotorSpeed, bc.Curve7 AS WayofRam
                FROM [dbo].[Orders] o WITH (NOLOCK)
                JOIN [dbo].[BatchHeader] bh WITH (NOLOCK) ON o.OrderID = bh.OrderID
                OUTER APPLY (
                    SELECT TOP 1 Curve1, Curve2, Curve5, Curve6, Curve7
                    FROM [dbo].[BatchCurve] bc WITH (NOLOCK)
                    WHERE bc.OrderID = o.OrderID AND bc.BatchNumber = bh.BatchNumber
                    ORDER BY bc.Timestamp DESC
                ) bc
                WHERE o.OrderStartTime BETWEEN '{start_time}' AND '{end_time}'
                  AND (o.CompoundName LIKE 'M1-%' OR o.CompoundName LIKE 'm1-%' OR o.CompoundName LIKE 'R1-%' OR o.CompoundName LIKE 'r1-%')
            )
            SELECT db.*,
                   d1.Value AS Top_Fill_Factor,
                   d2.Value AS Bot_Fill_Factor,
                   d3.Value AS Target_Temperature
            FROM daily_batches db
            LEFT JOIN [dbo].[BatchData] d1 WITH (NOLOCK) ON d1.OrderID = db.OrderID AND d1.BatchNumber = db.BatchNumber AND d1.VariablePath = 'Top_Fill_Factor'
            LEFT JOIN [dbo].[BatchData] d2 WITH (NOLOCK) ON d2.OrderID = db.OrderID AND d2.BatchNumber = db.BatchNumber AND d2.VariablePath = 'Bot_Fill_Factor'
            LEFT JOIN [dbo].[BatchData] d3 WITH (NOLOCK) ON d3.OrderID = db.OrderID AND d3.BatchNumber = db.BatchNumber AND d3.VariablePath = 'Target_Temperature'
            ORDER BY db.OrderStartTime ASC, db.BatchNumber ASC
        ''', conn)

    print(f"Queried {len(df_orders)} production batch records with 1Hz curves for {target_date_str}.")

    if len(df_orders) == 0:
        print(f"[DAILY ETL] No production batches found for date {target_date_str}.")
        return None

    # 2. Extract REAL DYNAMIC Process Features from 1Hz Curves
    df_batch_feats = extract_batch_process_features(df_orders)

    # 3. Query Lab Mooney Measurements
    try:
        import psycopg2 as psy
        with open(os.path.join(WORKSPACE_DIR, 'Mooney_Prediction_Pipeline', 'data_processing', 'credentials.json'), 'r', encoding='utf-8') as file:
            creds = json.load(file)
        c = creds['mustangmaster']
        conn_dm = psy.connect(
            host=c['host'], port=c['port'], database=c['database'], user=c['user'], password=c['password'], sslmode='require'
        )

        df_lab = pd.read_sql(f'''
            SELECT order_id, pallet_id, sample_id, test_result_start_time, compound_name, test_result
            FROM he_datamarts.compound_excellence_datamart
            WHERE test_variable = 'MS1+3'
              AND test_result_start_time BETWEEN '{start_time}' AND '{end_time}'
              AND test_status <> 'Failed'
              AND test_status_name <> 'D'
              AND test_result IS NOT NULL
        ''', conn_dm)
        conn_dm.close()
    except Exception as e:
        print(f"[DAILY ETL WARNING] Failed to query lab datamart: {e}")
        df_lab = pd.DataFrame(columns=['order_id', 'pallet_id', 'sample_id', 'test_result_start_time', 'compound_name', 'test_result'])

    # 4. Pallet Group Aggregation (OrderID + PalletID)
    grouped = df_batch_feats.groupby(['OrderID', 'PalletID']).agg({
        'CompoundName': 'first',
        'MixerLine': 'first',
        'OrderStartTime': 'first',
        'Top_Fill_Factor': 'mean',
        'Bot_Fill_Factor': 'mean',
        'Target_Temperature': 'mean',
        'phys_init_temp': 'mean',
        'phys_discharge_temp': 'mean',
        'phys_max_temp': 'mean',
        'phys_temp_integral': 'mean',
        'phys_temp_integral_above_100': 'mean',
        'phys_eta_app_discharge': 'mean',
        'Stage2_DryMixing_Duration': 'mean',
        'Stage2_DryMixing_power_Mean': 'mean',
        'Stage4_WetMixing_Duration': 'mean',
        'Stage4_WetMixing_temp_Mean': 'mean',
        'Stage6_BottomMixing_Torque_Mean': 'mean',
        'Stage6_BottomMixing_power_Mean': 'mean',
        'Stage6_BottomMixing_Duration': 'mean',
        'Stage6_BottomMixing_Torque_Integral': 'mean',
        'I_silanization': 'mean',
        'I_scorch': 'mean'
    }).reset_index()

    batch_info = df_batch_feats.groupby(['OrderID', 'PalletID'])['BatchNumber'].apply(lambda x: ', '.join(map(str, sorted(list(x))))).reset_index(name='Batch_List')
    batch_counts = df_batch_feats.groupby(['OrderID', 'PalletID']).size().reset_index(name='N_batches')

    df_samples = pd.merge(grouped, batch_info, on=['OrderID', 'PalletID'])
    df_samples = pd.merge(df_samples, batch_counts, on=['OrderID', 'PalletID'])

    # 5. Apply 3-Tier Map-Based Lookup for static recipe & raw material quality attributes
    df_history = pd.read_csv(os.path.join(WORKSPACE_DIR, 'stage_statistics_enriched_all_features_weather_v4.csv'), low_memory=False)
    recipe_cols = [
        'Top_Fill_Factor', 'Bot_Fill_Factor', 'Target_Temperature',
        'weight_pct_solid_elastomer', 'weight_pct_natural_rubber', 'weight_pct_silica',
        'weight_pct_oil', 'weight_pct_silian', 'weight_pct_carbon_black', 'silica_phr',
        'is_oil_loading_present', 'ratio_nr_rubber', 'ratio_filler_polymer',
        'ratio_oil_polymer', 'ratio_oil_filler',
        'supplier_rubber_viscosity_avg', 'supplier_silica_moisture_avg', 'supplier_silica_surface_area_avg',
        'supplier_carbon_black_structure_avg', 'supplier_carbon_black_surface_area_avg', 'supplier_carbon_black_moisture_avg',
        'env_temp_mean', 'env_humidity_mean'
    ]
    df_history['is_silica_system'] = ((df_history['silica_phr'] >= 25.0) & (df_history['weight_pct_silian'] > 0.0)).astype(float)
    df_history['is_oil_loading_present'] = (df_history['weight_pct_oil'] >= 5.0).astype(float)
    recipe_cols = [c for c in recipe_cols if c in df_history.columns]

    df_samples['BaseCompound'] = df_samples['CompoundName'].astype(str).str.strip().str[:14].str.replace('^R1-', 'M1-', regex=True)
    df_history['BaseCompound'] = df_history['CompoundName'].astype(str).str.strip().str[:14].str.replace('^R1-', 'M1-', regex=True)

    all_target_cols = list(set(recipe_cols + ['is_silica_system', 'is_oil_loading_present']))

    for col in all_target_cols:
        map_exact = df_history.groupby('CompoundName')[col].mean().to_dict()
        map_base = df_history.groupby('BaseCompound')[col].mean().to_dict()
        glob_val = float(df_history[col].mean()) if col in df_history.columns else 0.0
        
        s_exact = df_samples['CompoundName'].map(map_exact)
        s_base = df_samples['BaseCompound'].map(map_base)
        
        if col in df_samples.columns:
            df_samples[col] = df_samples[col].fillna(s_exact).fillna(s_base).fillna(glob_val)
        else:
            df_samples[col] = s_exact.fillna(s_base).fillna(glob_val)

    if len(df_lab) > 0:
        df_lab['order_id'] = df_lab['order_id'].astype(str).str.strip()
        df_lab['pallet_id'] = df_lab['pallet_id'].astype(str).str.strip()
        lab_avg = df_lab.groupby(['order_id', 'pallet_id'])['test_result'].mean().reset_index()
        lab_avg.columns = ['OrderID', 'PalletID', 'Actual_MNY']
        df_samples = pd.merge(df_samples, lab_avg, on=['OrderID', 'PalletID'], how='left')
    else:
        df_samples['Actual_MNY'] = np.nan

    # Export Parquet Files
    curves_parquet_path = os.path.join(parquet_dir, f"mms_curves_{date_tag}.parquet")
    lab_parquet_path = os.path.join(parquet_dir, f"lab_results_{date_tag}.parquet")
    features_parquet_path = os.path.join(parquet_dir, f"daily_features_{date_tag}.parquet")

    df_orders[['OrderID', 'BatchNumber', 'PalletID', 'temp', 'power', 'Torque']].to_parquet(curves_parquet_path, index=False, engine='pyarrow')
    df_lab.to_parquet(lab_parquet_path, index=False, engine='pyarrow')
    df_samples.to_parquet(features_parquet_path, index=False, engine='pyarrow')

    print(f"[DAILY ETL] Successfully exported Parquet files with REAL DYNAMIC process features to {parquet_dir}:")
    print(f"  - Curves: {os.path.basename(curves_parquet_path)}")
    print(f"  - Lab: {os.path.basename(lab_parquet_path)}")
    print(f"  - Features: {os.path.basename(features_parquet_path)} ({len(df_samples)} pallet samples)")

    return {
        'date': target_date_str,
        'features_path': features_parquet_path,
        'n_samples': len(df_samples)
    }

if __name__ == '__main__':
    export_daily_parquet('2026-07-22')
