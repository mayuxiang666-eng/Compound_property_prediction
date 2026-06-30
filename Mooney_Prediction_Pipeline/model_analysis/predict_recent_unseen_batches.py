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
import psycopg2 as psy
import pyodbc
import argparse
import torch
import joblib

BASE_DIR = WORKSPACE_ROOT
DATA_PROCESSING_DIR = os.path.join(BASE_DIR, 'data processing')

# Ensure data processing directory is in path for imports
if DATA_PROCESSING_DIR not in sys.path:
    sys.path.insert(0, DATA_PROCESSING_DIR)
from new_compound_inference import predict_new_compound
from predict_latest_sfe_order import connect_mms, extract_features_for_row
from validate_recent_ce_gap import query_sample_batch_mapping, query_order_batch_row, predict_feature_row, load_training_batch_keys
from train_mooney_nn_models import Autoencoder, DirectMLP, BottleneckMLP

def get_compound_family(name):
    if not isinstance(name, str):
        return 'Unknown'
    prefix = name.split()[0] if name.split() else name
    return prefix.rstrip('-')

# Constants
DEFAULT_TRAINING_CSV = os.path.join(WORKSPACE_ROOT, "stage_statistics_enriched_all_features_weather_v4.csv")
OUTPUT_CSV = os.path.join(WORKSPACE_ROOT, "results_recent_validation.csv")
TEMP_FEATURE_CSV = os.path.join(WORKSPACE_ROOT, 'scratch/recent_val_temp_features.csv')
TEMP_PRED_CSV = os.path.join(WORKSPACE_ROOT, 'scratch/recent_val_temp_prediction.csv')

# PyTorch Model helper functions
def fill_nan_series(arr):
    if len(arr) == 0:
        return arr
    s = pd.Series(arr)
    s = s.interpolate(method='linear', limit_direction='both').ffill().bfill().fillna(0.0)
    return s.values

def resample_segment(segment, num_points):
    n = len(segment)
    if n == 0:
        return np.zeros(num_points)
    xp = np.arange(n)
    x_new = np.linspace(0, n - 1, num_points)
    return np.interp(x_new, xp, segment)

def extract_nn_features_for_row(db_row, feature_row):
    from curve_segmenter_all_compounds import hex_to_series, fill_anomalies_by_compound, MAX_VALID_VALUE
    
    temp = hex_to_series(db_row['temp'])
    power = hex_to_series(db_row['power'])
    torque = hex_to_series(db_row['Torque'])
    speed = hex_to_series(db_row['RotorSpeed'])
    ram = hex_to_series(db_row['WayofRam'])

    tp_len = min(len(temp), len(power))
    if tp_len < 10:
        raise ValueError('Curves are too short.')
        
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

    # FIX: Add all expected keys including 'Compound' for fill_anomalies_by_compound
    decoded = [{
        'row_data': db_row.to_dict(),
        'OrderID': db_row['OrderID'],
        'BatchNumber': db_row['BatchNumber'],
        'Compound': db_row['CompoundName'],
        'Target_Temperature': float(db_row['Target_Temperature']) if pd.notna(db_row.get('Target_Temperature')) else 140.0,
        'temp': temp,
        'power': power,
        'Torque': torque,
        'RotorSpeed': speed,
        'WayofRam': ram,
        'PrevStepValue': db_row.get('PrevStepValue'),
        'CurrentValue': db_row.get('CurrentValue'),
    }]

    for d in decoded:
        for col in ['temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam']:
            arr = d[col]
            arr[arr > MAX_VALID_VALUE] = np.nan
            
    for col in ['temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam']:
        fill_anomalies_by_compound(decoded, col)

    d = decoded[0]
    temp = fill_nan_series(d['temp'])
    power = fill_nan_series(d['power'])
    torque = fill_nan_series(d['Torque'])
    speed = fill_nan_series(d['RotorSpeed'])
    ram = fill_nan_series(d['WayofRam'])

    t1 = int(feature_row['idx_t1_Loading'])
    t2 = int(feature_row['idx_t2_DryMixing'])
    t3 = int(feature_row['idx_t3_OilLoading'])
    t4 = int(feature_row['idx_t4_PID_Start'])
    t5 = int(feature_row['idx_t5_Discharge'])
    t_bot_start = int(feature_row['idx_t6_BottomMixing_Start'])
    t_bot_end = int(feature_row['idx_t7_BottomMixing_End'])
    
    is_oil = float(feature_row['is_oil_loading_present']) == 1.0
    
    if is_oil:
        stages = [
            (0, t1, "Stage1_Loading"),
            (t1, t2, "Stage2_DryMixing"),
            (t2, t3, "Stage3_OilLoading"),
            (t3, t4, "Stage4_WetMixing"),
            (t4, t5, "Stage5_PID"),
            (t_bot_start, t_bot_end, "Stage6_BottomMixing")
        ]
    else:
        stages = [
            (0, t1, "Stage1_Loading"),
            (t1, t4, "Stage2_DryMixing"),
            (t4, t5, "Stage5_PID"),
            (t_bot_start, t_bot_end, "Stage6_BottomMixing")
        ]

    K_POINTS = 20
    resampled_features = {}
    for s, e, stage_name in stages:
        s_idx = int(s)
        e_idx = int(e)
        
        for var_name, var_arr in [('temp', temp), ('power', power), ('Torque', torque), ('RotorSpeed', speed), ('WayofRam', ram)]:
            segment = var_arr[s_idx:e_idx]
            
            if stage_name == "Stage6_BottomMixing" and var_name == "WayofRam":
                resampled = np.zeros(K_POINTS)
            else:
                resampled = resample_segment(segment, K_POINTS)
                
            resampled_features[f"{stage_name}_{var_name}"] = resampled
            
    var_list = ['temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam']
    feature_vector = []
    for s_info in stages:
        stage_name = s_info[2]
        for var in var_list:
            feature_vector.append(resampled_features[f"{stage_name}_{var}"])
            
    feature_vector = np.concatenate(feature_vector)
    
    recipe_dict = {
        'weight_pct_solid_elastomer': float(db_row['weight_pct_solid_elastomer']) if pd.notna(db_row.get('weight_pct_solid_elastomer')) else 0.0,
        'weight_pct_natural_rubber': float(db_row['weight_pct_natural_rubber']) if pd.notna(db_row.get('weight_pct_natural_rubber')) else 0.0,
        'weight_pct_silica': float(db_row['weight_pct_silica']) if pd.notna(db_row.get('weight_pct_silica')) else 0.0,
        'weight_pct_oil': float(db_row['weight_pct_oil']) if pd.notna(db_row.get('weight_pct_oil')) else 0.0,
        'weight_pct_silian': float(db_row['weight_pct_silian']) if pd.notna(db_row.get('weight_pct_silian')) else 0.0,
        'weight_pct_carbon_black': float(db_row['weight_pct_carbon_black']) if pd.notna(db_row.get('weight_pct_carbon_black')) else 0.0,
        'supplier_rubber_viscosity_avg': float(db_row.get('supplier_rubber_viscosity_avg', 0.0)) if pd.notna(db_row.get('supplier_rubber_viscosity_avg')) else 0.0,
        'supplier_silica_moisture_avg': float(db_row.get('supplier_silica_moisture_avg', 0.0)) if pd.notna(db_row.get('supplier_silica_moisture_avg')) else 0.0,
        'supplier_silica_surface_area_avg': float(db_row.get('supplier_silica_surface_area_avg', 0.0)) if pd.notna(db_row.get('supplier_silica_surface_area_avg')) else 0.0,
        'supplier_carbon_black_structure_avg': float(db_row.get('supplier_carbon_black_structure_avg', 0.0)) if pd.notna(db_row.get('supplier_carbon_black_structure_avg')) else 0.0,
        'supplier_carbon_black_surface_area_avg': float(db_row.get('supplier_carbon_black_surface_area_avg', 0.0)) if pd.notna(db_row.get('supplier_carbon_black_surface_area_avg')) else 0.0,
        'supplier_carbon_black_moisture_avg': float(db_row.get('supplier_carbon_black_moisture_avg', 0.0)) if pd.notna(db_row.get('supplier_carbon_black_moisture_avg')) else 0.0,
        'Top_Fill_Factor': float(db_row['Top_Fill_Factor']) if pd.notna(db_row.get('Top_Fill_Factor')) else 0.0,
        'Bot_Fill_Factor': float(db_row['Bot_Fill_Factor']) if pd.notna(db_row.get('Bot_Fill_Factor')) else 0.0,
    }
    recipe_features = np.array(list(recipe_dict.values()))
    
    return feature_vector, recipe_features

nn_models_cache = {}

def get_nn_predictions(is_oil, feature_vector, recipe_features):
    track_name = "With-Oil" if is_oil else "Without-Oil"
    
    if track_name not in nn_models_cache:
        track_dir = os.path.join(PARENT_DIR, "models", f"results_nn_ae/{track_name.lower().replace('-', '_')}")
        input_dim = 600 if is_oil else 400
        
        # Load scalers
        curves_scaler = joblib.load(os.path.join(track_dir, 'curves_scaler.joblib'))
        recipe_scaler = joblib.load(os.path.join(track_dir, 'recipe_scaler.joblib'))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        ae_model = Autoencoder(input_dim=input_dim, latent_dim=32).to(device)
        ae_model.load_state_dict(torch.load(os.path.join(track_dir, 'autoencoder.pth'), map_location=device))
        ae_model.eval()
        
        recipe_dim = recipe_scaler.mean_.shape[0]
        mlp_model = DirectMLP(input_dim=input_dim, recipe_dim=recipe_dim).to(device)
        mlp_model.load_state_dict(torch.load(os.path.join(track_dir, 'mlp.pth'), map_location=device))
        mlp_model.eval()
        
        ae_mlp_model = BottleneckMLP(latent_dim=32, recipe_dim=recipe_dim).to(device)
        ae_mlp_model.load_state_dict(torch.load(os.path.join(track_dir, 'ae_mlp.pth'), map_location=device))
        ae_mlp_model.eval()
        
        nn_models_cache[track_name] = {
            'curves_scaler': curves_scaler,
            'recipe_scaler': recipe_scaler,
            'ae': ae_model,
            'mlp': mlp_model,
            'ae_mlp': ae_mlp_model,
            'device': device
        }
        
    models = nn_models_cache[track_name]
    device = models['device']
    
    # Scale features
    curves_scaled = models['curves_scaler'].transform(feature_vector.reshape(1, -1))
    recipe_scaled = models['recipe_scaler'].transform(recipe_features.reshape(1, -1))
    
    # Convert to tensors
    curves_tensor = torch.tensor(curves_scaled, dtype=torch.float32).to(device)
    recipe_tensor = torch.tensor(recipe_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred_mlp = models['mlp'](curves_tensor, recipe_tensor).cpu().numpy().item()
        _, latent = models['ae'](curves_tensor)
        pred_ae_mlp = models['ae_mlp'](latent, recipe_tensor).cpu().numpy().item()
        
    return float(pred_mlp), float(pred_ae_mlp)

def load_db_credentials():
    with open(os.path.join(PARENT_DIR, 'data_processing', 'credentials.json'), 'r', encoding='utf-8') as file:
        return json.load(file)

def connect_datamart(creds):
    c = creds['mustangmaster']
    return psy.connect(
        host=c['host'],
        port=c['port'],
        database=c['database'],
        user=c['user'],
        password=c['password'],
        sslmode='require',
    )

def query_recent_tests(creds, days_limit=14, limit=300):
    """Query recent MS1+3 tests from Redshift"""
    sql = f"""
    SELECT
        order_id,
        pallet_id,
        sample_id,
        equipment_id,
        test_variable,
        test_status,
        test_status_name,
        test_result_start_time,
        test_result_end_time,
        test_name,
        test_result,
        test_target,
        uom,
        compound_name,
        mixer,
        master_recipe,
        plant_id,
        is_retested,
        retest_number
    FROM he_datamarts.compound_excellence_datamart
    WHERE test_variable = 'MS1+3'
      AND test_result_start_time >= CURRENT_DATE - INTERVAL '{days_limit} day'
      AND test_status <> 'Failed'
      AND test_status_name <> 'D'
      AND test_result IS NOT NULL
      AND order_id IS NOT NULL
      AND sample_id IS NOT NULL
      AND plant_id = '9200'
      AND equipment_id = 'MV5'
      AND compound_name NOT LIKE '%M1-X%'
      AND compound_name <> 'Test'
    ORDER BY test_result_start_time DESC
    LIMIT {limit}
    """
    with connect_datamart(creds) as conn:
        return pd.read_sql(sql, conn)


def query_rolling_history(creds, days_limit=30):
    """Query recent MS1+3 tests from Redshift for rolling features"""
    sql = f"""
    SELECT
        order_id,
        sample_id,
        test_result_start_time,
        test_result,
        compound_name,
        mixer
    FROM he_datamarts.compound_excellence_datamart
    WHERE test_variable = 'MS1+3'
      AND test_result_start_time >= CURRENT_DATE - INTERVAL '{days_limit} day'
      AND test_status <> 'Failed'
      AND test_status_name <> 'D'
      AND test_result IS NOT NULL
      AND plant_id = '9200'
      AND equipment_id = 'MV5'
      AND compound_name NOT LIKE '%M1-X%'
    ORDER BY test_result_start_time ASC
    """
    with connect_datamart(creds) as conn:
        return pd.read_sql(sql, conn)

def main():
    parser = argparse.ArgumentParser(description='Automatically predict and validate recent unseen tested batches.')
    parser.add_argument('--days', type=int, default=14, help='Query test results from the last N days')
    parser.add_argument('--max-tests', type=int, default=15, help='Maximum number of unseen mapped batches to validate')
    parser.add_argument('--training-csv', default=DEFAULT_TRAINING_CSV, help='Path to training feature CSV to exclude already trained batches')
    parser.add_argument('--bypass-exclude', action='store_true', help='Bypass training batch exclusion for testing calibration')
    args = parser.parse_args()

    print("======================================================================")
    print("      AUTOMATIC VALIDATION FOR RECENT UNSEEN TESTED BATCHES")
    print("======================================================================\n")

    # 1. Load credentials
    print("Step 1: Loading database credentials...")
    try:
        creds = load_db_credentials()
    except Exception as e:
        print(f"Error loading credentials.json: {e}")
        return

    # 2. Load trained keys
    print(f"Step 2: Loading trained batch keys from '{args.training_csv}'...")
    training_keys = set()
    if os.path.exists(args.training_csv):
        training_keys = load_training_batch_keys(args.training_csv)
        print(f"  -> Loaded {len(training_keys)} trained batch keys to exclude.")
    else:
        print(f"  -> Warning: '{args.training_csv}' not found. No training batch exclusion applied.")

    # Load family medians from model bundles
    print("Loading GBDT model bundles to extract baseline & family medians...")
    try:
        with_oil_bundle = joblib.load(os.path.join(PARENT_DIR, 'models', 'results_with_oil/mooney_model_bundle.joblib'))
        without_oil_bundle = joblib.load(os.path.join(PARENT_DIR, 'models', 'results_without_oil/mooney_model_bundle.joblib'))
        family_medians_all = {}
        family_medians_all.update(with_oil_bundle.get('family_medians', {}))
        family_medians_all.update(without_oil_bundle.get('family_medians', {}))
        track_median_oil = float(with_oil_bundle.get('track_median', 45.0))
        track_median_no_oil = float(without_oil_bundle.get('track_median', 45.0))
        print("Model bundles loaded successfully.")
    except Exception as e:
        print(f"Error loading model bundles: {e}")
        family_medians_all = {}
        track_median_oil = 45.0
        track_median_no_oil = 45.0

    # Load training history to handle cold-start compounds
    print("Loading training history for rolling features...")
    try:
        train_hist = pd.read_csv('stage_statistics_enriched_all_features_weather_v4.csv', 
                                 usecols=['OrderID', 'OrderStartTime', 'test_result_start_time', 'MNY', 'CompoundName', 'MixerLine'],
                                 low_memory=False)
        train_hist = train_hist.rename(columns={'CompoundName': 'compound_name', 'MixerLine': 'mixer', 'MNY': 'test_result'})
    except Exception as e:
        print(f"Warning: Could not load training history: {e}")
        train_hist = pd.DataFrame(columns=['OrderID', 'OrderStartTime', 'test_result_start_time', 'test_result', 'compound_name', 'mixer'])

    print("Querying 30-day Redshift history for rolling features...")
    try:
        redshift_hist = query_rolling_history(creds, days_limit=30)
        redshift_hist = redshift_hist.rename(columns={'order_id': 'OrderID'})
    except Exception as e:
        print(f"Warning: Could not query Redshift history: {e}")
        redshift_hist = pd.DataFrame(columns=['OrderID', 'test_result_start_time', 'test_result', 'compound_name', 'mixer'])
        
    # Combine training and recent redshift history
    all_history = pd.concat([train_hist, redshift_hist]).drop_duplicates(subset=['OrderID'])
    all_history['test_result_start_time'] = pd.to_datetime(all_history['test_result_start_time'])
    all_history['OrderStartTime'] = pd.to_datetime(all_history['OrderStartTime'])
    all_history['OrderStartTime'] = all_history['OrderStartTime'].fillna(all_history['test_result_start_time'] - pd.Timedelta(hours=2))
    all_history['visible_time'] = all_history['test_result_start_time'].fillna(all_history['OrderStartTime'] + pd.Timedelta(hours=6))
    all_history = all_history.sort_values('OrderStartTime').reset_index(drop=True)
    print(f"Chronological rolling history prepared with {len(all_history)} records.")

    # 3. Query Redshift
    print(f"Step 3: Querying MS1+3 test results from Redshift (Last {args.days} days)...")
    try:
        ce_rows = query_recent_tests(creds, days_limit=args.days)
        print(f"  -> Retrieved {len(ce_rows)} lab tests.")
        if not ce_rows.empty:
            # Pass 1: keep latest retest of the exact same (order_id, sample_id)
            ce_rows = ce_rows.sort_values(by='test_result_start_time')
            ce_rows = ce_rows.drop_duplicates(subset=['order_id', 'sample_id'], keep='last')

            # Pass 2: within each order, collapse consecutive / adjacent sample_ids.
            # Two samples are "adjacent" when their sample_ids differ by <= 2.
            # They represent the same physical batch being retested: keep only the
            # latest sample_id (= the retest) and discard the earlier one.
            def keep_latest_per_consecutive_group(grp):
                """Within one order's samples, group consecutive sample_ids and
                keep only the row with the highest sample_id in each group."""
                grp = grp.sort_values('sample_id')
                sids = grp['sample_id'].tolist()
                # Assign a group label: increment whenever gap between consecutive ids > 2
                group_labels = [0]
                for i in range(1, len(sids)):
                    gap = abs(sids[i] - sids[i - 1])
                    group_labels.append(group_labels[-1] if gap <= 2 else group_labels[-1] + 1)
                grp = grp.copy()
                grp['_consec_group'] = group_labels
                # From each consecutive group keep the row with the max sample_id
                return grp.loc[grp.groupby('_consec_group')['sample_id'].idxmax()]

            ce_rows['sample_id'] = pd.to_numeric(ce_rows['sample_id'], errors='coerce')
            ce_rows = (
                ce_rows
                .groupby('order_id', group_keys=True)
                .apply(keep_latest_per_consecutive_group, include_groups=False)
                .reset_index(level=0)
                .drop(columns=['_consec_group'], errors='ignore')
                .reset_index(drop=True)
            )

            # Sort ascending by test time so we process chronologically
            ce_rows = ce_rows.sort_values(by='test_result_start_time', ascending=True)
            print(f"  -> Deduplicated to {len(ce_rows)} unique latest retests (consecutive groups merged).")
    except Exception as e:
        print(f"Error querying Redshift: {e}")
        return

    if ce_rows.empty:
        print("No recent tests found in the specified date range.")
        return

    # 4. Filter for base compounds (M1/R1)
    ce_rows['compound_name'] = ce_rows['compound_name'].astype(str)
    base_ce = ce_rows[
        ce_rows['compound_name'].str.startswith('M1-', na=False) |
        ce_rows['compound_name'].str.startswith('m1-', na=False) |
        ce_rows['compound_name'].str.startswith('R1-', na=False) |
        ce_rows['compound_name'].str.startswith('r1-', na=False)
    ].copy()
    print(f"Step 4: Filtered to M1/R1 base compounds: {len(base_ce)} tests.")
    
    if base_ce.empty:
        print("No recent M1/R1 tests found.")
        return

    # 5. Loop and process
    print("\nStep 5: Matching SFEPLANT curves and running predictions...")
    reports = []
    failed_attempts = []
    processed_batch_keys = set()  # guard against same batch being processed twice
    
    # Initialize online calibration states at family level
    kf_state = {}
    ewma_state = {}
    
    Q = 0.01
    R = 1.0
    alpha = 0.2
    
    for idx, ce in base_ce.iterrows():
        if len(reports) >= args.max_tests:
            print(f"Reached maximum limit of {args.max_tests} validated batches.")
            break
            
        order_id = str(ce['order_id']).strip()
        sample_id = int(float(ce['sample_id']))
        comp_name = ce['compound_name']
        actual_mny = float(ce['test_result'])
        test_time = ce['test_result_start_time']
        
        print(f"\nProcessing Order: {order_id} | Sample: {sample_id} | Compound: {comp_name} | Lab MNY: {actual_mny}")
        
        mapped = False
        # Search SFEPLANT and SFEPLANT_ARCHIVE
        for database in ['SFEPLANT', 'SFEPLANT_ARCHIVE']:
            try:
                mapping = query_sample_batch_mapping(order_id, sample_id, database)
                if mapping.empty:
                    continue
                    
                batch_number = int(mapping.iloc[0]['batch_number_pred'])
                batch_key = f"{order_id}_{batch_number:04d}"

                if batch_key in processed_batch_keys:
                    print(f"  -> Skipping: Batch {batch_key} already processed in this run (consecutive sample duplicate).")
                    mapped = True
                    break

                if not args.bypass_exclude and batch_key in training_keys:
                    print(f"  -> Skipping: Batch {batch_key} already used in model training.")
                    mapped = True
                    break
                    
                # Fetch curves
                batch_row_df = query_order_batch_row(order_id, batch_number, database)
                if batch_row_df.empty:
                    failed_attempts.append({'order_id': order_id, 'sample_id': sample_id, 'reason': f"No curve data found in {database} for batch {batch_number}"})
                    continue
                
                db_row = batch_row_df.iloc[0]
                
                # Extract base features
                feature_row = extract_features_for_row(db_row)
                
                # ----------------------------------------------------
                # V2.0: Chronological Rolling Features Calculation
                # ----------------------------------------------------
                t_current = pd.to_datetime(db_row.get('OrderStartTime', test_time - pd.Timedelta(hours=2)))
                
                # Get visible history
                visible_hist = all_history[all_history['visible_time'] < t_current].copy()
                
                # Compute rolling features
                comp_vis = visible_hist[visible_hist['compound_name'] == comp_name]
                
                roll_mny_mean_3b = np.nan
                roll_mny_std_10b = np.nan
                if len(comp_vis) >= 1:
                    roll_mny_mean_3b = float(comp_vis['test_result'].iloc[-3:].mean())
                if len(comp_vis) >= 2:
                    roll_mny_std_10b = float(comp_vis['test_result'].iloc[-10:].std()) if len(comp_vis['test_result'].iloc[-10:]) > 1 else 0.0
                    
                roll_resid_mean_5b = np.nan
                roll_resid_mean_10b = np.nan
                
                if len(visible_hist) > 0:
                    visible_hist['compound_family'] = visible_hist['compound_name'].apply(get_compound_family)
                    
                    # Compute residuals using family_medians
                    def get_baseline(row_hist):
                        c = row_hist['compound_name']
                        is_oil_val = 'T' in str(c)
                        fallback = track_median_oil if is_oil_val else track_median_no_oil
                        return family_medians_all.get(c, fallback)
                        
                    visible_hist['baseline'] = visible_hist.apply(get_baseline, axis=1)
                    visible_hist['y_resid'] = visible_hist['test_result'] - visible_hist['baseline']
                    
                    # Family rolling residual
                    fam_name = get_compound_family(comp_name)
                    fam_vis = visible_hist[visible_hist['compound_family'] == fam_name]
                    if len(fam_vis) >= 1:
                        roll_resid_mean_5b = float(fam_vis['y_resid'].iloc[-5:].mean())
                        
                    # Mixer rolling residual
                    mixer_vis = visible_hist[visible_hist['mixer'] == db_row.get('MixerLine', 'MV5')]
                    if len(mixer_vis) >= 1:
                        roll_resid_mean_10b = float(mixer_vis['y_resid'].iloc[-10:].mean())
                
                # Inject rolling features into GBDT feature row
                feature_row['roll_mny_mean_3b_same_comp'] = roll_mny_mean_3b
                feature_row['roll_mny_std_10b_same_comp'] = roll_mny_std_10b
                feature_row['roll_resid_mean_5b_family'] = roll_resid_mean_5b
                feature_row['roll_resid_mean_10b_mixer'] = roll_resid_mean_10b
                
                # Override temp CSV paths inside function using scratch to avoid conflict
                os.makedirs('scratch', exist_ok=True)
                feature_df = pd.DataFrame([feature_row])
                feature_df.to_csv(TEMP_FEATURE_CSV, index=False, encoding='utf-8-sig')
                
                is_oil = float(feature_row['is_oil_loading_present']) == 1.0
                silica_phr_val = float(feature_row.get('silica_phr', 0.0))
                weight_pct_silian_val = float(feature_row.get('weight_pct_silian', 0.0))
                is_silica_system = silica_phr_val >= 25.0 and weight_pct_silian_val > 0.0
                
                if '15760' in str(comp_name):
                    bundle_path = os.path.join(PARENT_DIR, 'models', 'results_15760/mooney_model_bundle_15760.joblib')
                    print("  -> Using dedicated M1-T15760 model bundle.")
                else:
                    if is_oil:
                        if is_silica_system:
                            folder = "results_with_oil_high_silica"
                        else:
                            folder = "results_with_oil_carbon_black"
                    else:
                        if is_silica_system:
                            folder = "results_without_oil_high_silica"
                        else:
                            folder = "results_without_oil_carbon_black"
                    bundle_path = os.path.join(PARENT_DIR, 'models', folder, 'mooney_model_bundle.joblib')
                
                if not os.path.exists(bundle_path):
                    print(f"  -> Error: Model bundle '{bundle_path}' not found. Train GBDT models first.")
                    continue
                    
                # Predict GBDT
                predict_new_compound(bundle_path, TEMP_FEATURE_CSV, TEMP_PRED_CSV)
                pred_row = pd.read_csv(TEMP_PRED_CSV).iloc[0]
                pred_mny = float(pred_row['predicted_MNY_base'])
                conf_score = float(pred_row.get('confidence_score', 1.0))
                conf_label = str(pred_row.get('confidence_label', 'Green'))
                is_similarity_applied = bool(pred_row.get('is_similarity_applied', False))
                is_ood_fallback_applied = bool(pred_row.get('is_ood_fallback_applied', False))
                gap = pred_mny - actual_mny
                
                # Extract NN features and run NN predictions
                try:
                    nn_features, nn_recipe = extract_nn_features_for_row(db_row, feature_row)
                    pred_nn_mlp, pred_nn_ae = get_nn_predictions(is_oil, nn_features, nn_recipe)
                except Exception as ex:
                    print(f"  -> Warning: NN feature extraction or prediction failed: {ex}")
                    pred_nn_mlp, pred_nn_ae = np.nan, np.nan
                
                # ----------------------------------------------------
                # V2.0: Online EWMA and Kalman Filter Calibration (Family level)
                # ----------------------------------------------------
                fam_name = get_compound_family(comp_name)
                if fam_name not in kf_state:
                    kf_state[fam_name] = {'bias': 0.0, 'P': 1.0}
                if fam_name not in ewma_state:
                    ewma_state[fam_name] = {'bias': 0.0}
                    
                # Fetch current biases
                bias_kf = kf_state[fam_name]['bias']
                bias_ewma = ewma_state[fam_name]['bias']
                
                # Apply correction
                calibrated_gbdt_kf = pred_mny + bias_kf
                calibrated_gbdt_ewma = pred_mny + bias_ewma
                
                # Update calibration state using GBDT error
                error_gbdt = actual_mny - pred_mny
                
                # Kalman Filter Update
                P_pred = kf_state[fam_name]['P'] + Q
                innovation = error_gbdt - bias_kf
                S = P_pred + R
                K = P_pred / S
                bias_kf_new = bias_kf + K * innovation
                P_new = (1.0 - K) * P_pred
                
                bias_kf_new = np.clip(bias_kf_new, -6.0, 6.0)
                kf_state[fam_name]['bias'] = bias_kf_new
                kf_state[fam_name]['P'] = P_new
                
                # EWMA Update
                bias_ewma_new = (1.0 - alpha) * bias_ewma + alpha * error_gbdt
                bias_ewma_new = np.clip(bias_ewma_new, -6.0, 6.0)
                ewma_state[fam_name]['bias'] = bias_ewma_new
                
                # Also apply same calibration to Neural Network models
                calibrated_nn_mlp_kf = pred_nn_mlp + bias_kf if pd.notna(pred_nn_mlp) else np.nan
                calibrated_nn_ae_kf = pred_nn_ae + bias_kf if pd.notna(pred_nn_ae) else np.nan
                
                report = {
                    'order_id': order_id,
                    'sample_id': sample_id,
                    'mapped_batch_number': batch_number,
                    'batch_key': batch_key,
                    'database': database,
                    'compound_name': comp_name,
                    'test_result_start_time': test_time,
                    'lab_MNY': actual_mny,
                    
                    # Tree Model (GBDT) Predictions
                    'predicted_MNY_gbdt': pred_mny,
                    'gbdt_gap': gap,
                    'gbdt_abs_gap': abs(gap),
                    
                    # Online Calibrated GBDT
                    'predicted_MNY_gbdt_kf': calibrated_gbdt_kf,
                    'predicted_MNY_gbdt_ewma': calibrated_gbdt_ewma,
                    'bias_kf': bias_kf,
                    'bias_ewma': bias_ewma,
                    
                    # Neural Network Predictions
                    'predicted_MNY_nn_mlp': pred_nn_mlp,
                    'nn_mlp_gap': (pred_nn_mlp - actual_mny) if pd.notna(pred_nn_mlp) else np.nan,
                    'nn_mlp_abs_gap': abs(pred_nn_mlp - actual_mny) if pd.notna(pred_nn_mlp) else np.nan,
                    
                    'predicted_MNY_nn_mlp_kf': calibrated_nn_mlp_kf,
                    
                    'predicted_MNY_nn_ae': pred_nn_ae,
                    'nn_ae_gap': (pred_nn_ae - actual_mny) if pd.notna(pred_nn_ae) else np.nan,
                    'nn_ae_abs_gap': abs(pred_nn_ae - actual_mny) if pd.notna(pred_nn_ae) else np.nan,
                    
                    'predicted_MNY_nn_ae_kf': calibrated_nn_ae_kf,
                    
                    'is_oil_loading_present': feature_row['is_oil_loading_present'],
                    'gbdt_model_bundle': bundle_path,
                    'reliability': pred_row['applicability_reliability'],
                    'applicability_distance': pred_row['applicability_distance'],
                    'confidence_score': conf_score,
                    'confidence_label': conf_label,
                    'is_similarity_applied': is_similarity_applied,
                    'is_ood_fallback_applied': is_ood_fallback_applied
                }
                reports.append(report)
                processed_batch_keys.add(batch_key)
                
                nn_mlp_str = f"{pred_nn_mlp:.2f}" if pd.notna(pred_nn_mlp) else "N/A"
                nn_ae_str = f"{pred_nn_ae:.2f}" if pd.notna(pred_nn_ae) else "N/A"
                print(f"  -> SUCCESS! MNY Lab: {actual_mny:.2f} | GBDT: {pred_mny:.2f} (KF Cal: {calibrated_gbdt_kf:.2f}) | Conf: {conf_label} ({conf_score:.2f}) | SimApplied: {is_similarity_applied} | OOD Reverted: {is_ood_fallback_applied}")
                mapped = True
                break
                
            except Exception as e:
                failed_attempts.append({'order_id': order_id, 'sample_id': sample_id, 'reason': f"Error in {database}: {str(e)}"})
        
        if not mapped:
            print(f"  -> Failed to map/process Order {order_id} Sample {sample_id} to physical curves.")

    # 6. Save and output summary
    if reports:
        report_df = pd.DataFrame(reports)
        
        # --- AUTOMATED FEW-SHOT CALIBRATION FOR ALL MODELS ---
        print("\nStep 6: Calculating few-shot calibration biases for all models...")
        
        report_df['compound_family'] = report_df['compound_name'].apply(get_compound_family)
        
        # Calculate raw biases
        report_df['bias_gbdt'] = report_df['lab_MNY'] - report_df['predicted_MNY_gbdt']
        report_df['bias_nn_mlp'] = report_df['lab_MNY'] - report_df['predicted_MNY_nn_mlp']
        report_df['bias_nn_ae'] = report_df['lab_MNY'] - report_df['predicted_MNY_nn_ae']
        
        biases = {
            'gbdt': {'families': {}, 'tracks': {}},
            'nn_mlp': {'families': {}, 'tracks': {}},
            'nn_ae': {'families': {}, 'tracks': {}}
        }
        
        # Compute family-level and track-level biases for each model
        for model_key in ['gbdt', 'nn_mlp', 'nn_ae']:
            bias_col = f'bias_{model_key}'
            
            # Family-level biases (require at least 2 samples for stability)
            for family, group in report_df.groupby('compound_family'):
                valid_group = group.dropna(subset=[bias_col])
                if len(valid_group) >= 2:
                    biases[model_key]['families'][family] = float(valid_group[bias_col].mean())
                    
            # Track-level biases (With-Oil vs Without-Oil)
            for oil_val, group in report_df.groupby('is_oil_loading_present'):
                track_name = "With-Oil" if oil_val == 1.0 else "Without-Oil"
                valid_group = group.dropna(subset=[bias_col])
                if len(valid_group) > 0:
                    biases[model_key]['tracks'][track_name] = float(valid_group[bias_col].mean())
            
        # Write GBDT calibration biases to JSON for backward compatibility
        bias_path = os.path.join(PARENT_DIR, 'models', 'results_m2_analysis/calibration_biases.json')
        os.makedirs(os.path.dirname(bias_path), exist_ok=True)
        with open(bias_path, 'w', encoding='utf-8') as f:
            json.dump(biases['gbdt'], f, ensure_ascii=False, indent=2)
        print(f"Successfully saved GBDT calibration biases to: '{bias_path}'")

        # --- APPLY BIASES TO EVALUATE CALIBRATION PERFORMANCE ---
        for model_key in ['gbdt', 'nn_mlp', 'nn_ae']:
            calibrated_preds = []
            applied_biases = []
            bias_sources = []
            
            for idx, row in report_df.iterrows():
                family = row['compound_family']
                oil_val = row['is_oil_loading_present']
                track_name = "With-Oil" if oil_val == 1.0 else "Without-Oil"
                
                pred_val = row[f'predicted_MNY_{model_key}']
                if pd.isna(pred_val):
                    calibrated_preds.append(np.nan)
                    applied_biases.append(np.nan)
                    bias_sources.append("None")
                    continue
                    
                bias_val = 0.0
                bias_src = "None"
                
                if family in biases[model_key]['families']:
                    bias_val = biases[model_key]['families'][family]
                    bias_src = f"Family:{family}"
                elif track_name in biases[model_key]['tracks']:
                    bias_val = biases[model_key]['tracks'][track_name]
                    bias_src = f"Track:{track_name}"
                    
                calibrated_preds.append(pred_val + bias_val)
                applied_biases.append(bias_val)
                bias_sources.append(bias_src)
                
            report_df[f'predicted_MNY_{model_key}_calibrated'] = calibrated_preds
            report_df[f'applied_bias_{model_key}'] = applied_biases
            report_df[f'bias_source_{model_key}'] = bias_sources
            report_df[f'gap_{model_key}_calibrated'] = report_df[f'predicted_MNY_{model_key}_calibrated'] - report_df['lab_MNY']
            report_df[f'abs_gap_{model_key}_calibrated'] = report_df[f'gap_{model_key}_calibrated'].abs()
        
        # --- DETECT AND FILTER OUTLIERS FROM VALIDATION METRICS ---
        print("\nStep 7: Identifying and filtering validation outliers...")
        try:
            train_df = pd.read_csv(args.training_csv, usecols=['CompoundName', 'MNY'], low_memory=False)
            train_df['MNY'] = pd.to_numeric(train_df['MNY'], errors='coerce')
            train_df['compound_family'] = train_df['CompoundName'].apply(get_compound_family)
            family_medians = train_df.groupby('compound_family')['MNY'].median().to_dict()
        except Exception as e:
            print(f"Warning: Could not compute family medians from training CSV for validation filtering: {e}")
            family_medians = {}
            
        drop_indices = []
        for idx, row in report_df.iterrows():
            family = row['compound_family']
            family_med = family_medians.get(family, np.nan)
            if pd.notna(family_med):
                if abs(row['lab_MNY'] - family_med) > 20.0:
                    drop_indices.append(idx)
                    print(f"  -> WARNING: Order {row['order_id']} Batch {row['mapped_batch_number']} ({row['compound_name']}) flagged as outlier (Lab MNY={row['lab_MNY']:.2f}, Family Median={family_med:.2f}). Excluded from validation metrics.")
        
        # Create cleaned dataframe for metrics calculation
        report_df_stats = report_df.drop(index=drop_indices) if drop_indices else report_df.copy()
        
        # Save to CSV (overwriting base CSV with calibrated details)
        try:
            report_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        except PermissionError:
            print(f"\n[Warning] Permission denied when writing to '{OUTPUT_CSV}'. Please close the file if it is open in Excel and try again.")
        
        print("\n=======================================================")
        print("                 VALIDATION RESULTS SUMMARY")
        print("=======================================================")
        print(f"Saved detailed comparison report to: '{OUTPUT_CSV}'")
        
        pd.set_option('display.max_columns', None)
        print("\nPredictions vs Actuals:")
        cols_to_print = ['order_id', 'mapped_batch_number', 'compound_name', 'lab_MNY', 'predicted_MNY_gbdt', 'predicted_MNY_gbdt_kf', 'predicted_MNY_nn_ae', 'predicted_MNY_nn_ae_kf', 'reliability']
        print(report_df[[c for c in cols_to_print if c in report_df.columns]].to_string(index=False))
        
        # Calculate stats for all three models and calibration methods
        stats_data = []
        for model_key, model_label in [('gbdt', 'GBDT Stacked'), ('nn_mlp', 'NN Direct MLP'), ('nn_ae', 'NN AE + MLP')]:
            kf_col = f'predicted_MNY_{model_key}_kf'
            if kf_col in report_df_stats.columns:
                valid_df = report_df_stats.dropna(subset=[f'predicted_MNY_{model_key}', kf_col])
            else:
                valid_df = report_df_stats.dropna(subset=[f'predicted_MNY_{model_key}'])
                
            if len(valid_df) > 0:
                base_mae = (valid_df[f'predicted_MNY_{model_key}'] - valid_df['lab_MNY']).abs().mean()
                base_rmse = np.sqrt(((valid_df[f'predicted_MNY_{model_key}'] - valid_df['lab_MNY'])**2).mean())
                base_bias = (valid_df[f'predicted_MNY_{model_key}'] - valid_df['lab_MNY']).mean()
                
                cal_mae = valid_df[f'abs_gap_{model_key}_calibrated'].mean() if f'abs_gap_{model_key}_calibrated' in valid_df.columns else np.nan
                cal_rmse = np.sqrt((valid_df[f'gap_{model_key}_calibrated']**2).mean()) if f'gap_{model_key}_calibrated' in valid_df.columns else np.nan
                cal_bias = valid_df[f'gap_{model_key}_calibrated'].mean() if f'gap_{model_key}_calibrated' in valid_df.columns else np.nan
                
                kf_mae = (valid_df[kf_col] - valid_df['lab_MNY']).abs().mean() if kf_col in valid_df.columns else np.nan
                kf_rmse = np.sqrt(((valid_df[kf_col] - valid_df['lab_MNY'])**2).mean()) if kf_col in valid_df.columns else np.nan
                kf_bias = (valid_df[kf_col] - valid_df['lab_MNY']).mean() if kf_col in valid_df.columns else np.nan
                
                stats_data.append({
                    'Model': model_label,
                    'Base MAE': base_mae,
                    'Cal MAE': cal_mae,
                    'Kalman MAE': kf_mae,
                    'Base RMSE': base_rmse,
                    'Cal RMSE': cal_rmse,
                    'Kalman RMSE': kf_rmse,
                    'Base Bias': base_bias,
                    'Cal Bias': cal_bias,
                    'Kalman Bias': kf_bias
                })
        
        stats_summary_df = pd.DataFrame(stats_data)
        
        print("\nOverall Performance Comparison:")
        print(f"  Total Unseen Batches Validated: {len(report_df_stats)} (Excluded {len(drop_indices)} outliers)")
        print(stats_summary_df.to_string(index=False))
        
        print("\nPerformance Comparison by Track:")
        for oil_val, group in report_df_stats.groupby('is_oil_loading_present'):
            track = "With-Oil" if oil_val == 1.0 else "Without-Oil"
            print(f"\n  Track: {track} ({len(group)} batches)")
            t_stats = []
            for model_key, model_label in [('gbdt', 'GBDT Stacked'), ('nn_mlp', 'NN Direct MLP'), ('nn_ae', 'NN AE + MLP')]:
                kf_col = f'predicted_MNY_{model_key}_kf'
                if kf_col in group.columns:
                    valid_g = group.dropna(subset=[f'predicted_MNY_{model_key}', kf_col])
                else:
                    valid_g = group.dropna(subset=[f'predicted_MNY_{model_key}'])
                    
                if len(valid_g) > 0:
                    t_base_mae = (valid_g[f'predicted_MNY_{model_key}'] - valid_g['lab_MNY']).abs().mean()
                    t_base_rmse = np.sqrt(((valid_g[f'predicted_MNY_{model_key}'] - valid_g['lab_MNY'])**2).mean())
                    t_base_bias = (valid_g[f'predicted_MNY_{model_key}'] - valid_g['lab_MNY']).mean()
                    
                    t_cal_mae = valid_g[f'abs_gap_{model_key}_calibrated'].mean() if f'abs_gap_{model_key}_calibrated' in valid_g.columns else np.nan
                    t_cal_rmse = np.sqrt((valid_g[f'gap_{model_key}_calibrated']**2).mean()) if f'gap_{model_key}_calibrated' in valid_g.columns else np.nan
                    t_cal_bias = valid_g[f'gap_{model_key}_calibrated'].mean() if f'gap_{model_key}_calibrated' in valid_g.columns else np.nan
                    
                    t_kf_mae = (valid_g[kf_col] - valid_g['lab_MNY']).abs().mean() if kf_col in valid_g.columns else np.nan
                    t_kf_rmse = np.sqrt(((valid_g[kf_col] - valid_g['lab_MNY'])**2).mean()) if kf_col in valid_g.columns else np.nan
                    t_kf_bias = (valid_g[kf_col] - valid_g['lab_MNY']).mean() if kf_col in valid_g.columns else np.nan
                    
                    t_stats.append({
                        'Model': model_label,
                        'Base MAE': t_base_mae,
                        'Cal MAE': t_cal_mae,
                        'Kalman MAE': t_kf_mae,
                        'Base RMSE': t_base_rmse,
                        'Cal RMSE': t_cal_rmse,
                        'Kalman RMSE': t_kf_rmse,
                        'Base Bias': t_base_bias,
                        'Cal Bias': t_cal_bias,
                        'Kalman Bias': t_kf_bias
                    })
            t_stats_df = pd.DataFrame(t_stats)
            print(t_stats_df.to_string(index=False))
            
    else:
        print("\nNo unseen tested batches were successfully processed.")
        if failed_attempts:
            print("\nRecent failed attempts details:")
            for fa in failed_attempts[:5]:
                print(f"  Order: {fa['order_id']}, Sample: {fa['sample_id']} | Reason: {fa['reason']}")

if __name__ == '__main__':
    main()
