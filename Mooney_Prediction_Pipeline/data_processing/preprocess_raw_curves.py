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

import pandas as pd
import numpy as np
import os
import joblib
import sys
from sklearn.ensemble import IsolationForest

# Configure stdout to use utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Input / Output paths
INPUT_CSV = os.path.join(WORKSPACE_ROOT, "stage_statistics_enriched.csv")
OUTPUT_JOBLIB = "scratch/neural_network_dataset.joblib"
ANOMALY_CSV = os.path.join(PARENT_DIR, "models", "results_nn_ae", "isolation_forest_anomalies.csv")

HEX_STEP = 4
MAX_VALID_VALUE = 10000
K_POINTS = 20
IF_CONTAMINATION = 0.03

def hex_to_series(hex_str):
    """Parse hex string to numeric series (every 4 characters as a 16-bit integer)"""
    if pd.isna(hex_str) or str(hex_str).strip() in ['nan', '', 'NaN']:
        return np.array([], dtype=float)
    hex_str = str(hex_str).strip()
    if len(hex_str) == 0:
        return np.array([], dtype=float)
    values = []
    for i in range(0, len(hex_str), HEX_STEP):
        try:
            val = int(hex_str[i:i+HEX_STEP], 16)
            values.append(val)
        except ValueError:
            continue
    return np.array(values, dtype=float)

def find_ram_first_drop(ram):
    """Find the index of the first sudden drop in WayofRam"""
    if len(ram) < 10:
        return 0
    diff = np.diff(ram)
    for i in range(len(diff)):
        if diff[i] < -5:
            return i + 1
    return 0

def find_temp_pid_start(temp, target_temp, oil_end):
    """Find the index after oil_end where Temperature reaches target_temp - 3 and remains stable"""
    target = target_temp - 3
    for i in range(int(oil_end), len(temp)):
        if not np.isnan(temp[i]) and temp[i] >= target:
            if i + 5 < len(temp):
                future_temps = temp[i:i+5]
                if np.all(future_temps >= target - 10):
                    return i
            else:
                return i
    return min(oil_end + 10, len(temp) - 1)

def find_temp_discharge_start(temp, pid_start):
    """Find the index of the TOP-MIXING DISCHARGE after pid_start"""
    if pid_start >= len(temp) - 2:
        return len(temp) - 1
    diff = np.diff(temp)
    best_i = -1
    best_drop = -15  # must drop at least 15°C in one step to count
    for i in range(pid_start + 1, len(diff)):
        if not np.isnan(diff[i]) and diff[i] < best_drop:
            best_drop = diff[i]
            best_i = i
    if best_i >= 0:
        return best_i
    for i in range(pid_start + 1, len(diff)):
        if not np.isnan(diff[i]) and diff[i] < -5:
            return i
    return max(pid_start + 1, len(temp) - 5)

def find_bottom_mixer_start(temp, t5):
    """Find the index where BOTTOM MIXING starts"""
    total_len = len(temp)
    if t5 >= total_len - 3:
        return total_len - 1
    search_limit = min(t5 + 80, total_len)
    min_val = temp[t5] if not np.isnan(temp[t5]) else np.inf
    min_idx = t5
    for i in range(t5, search_limit):
        val = temp[i]
        if not np.isnan(val) and val < min_val:
            min_val = val
            min_idx = i
    rise_search_limit = min(min_idx + 100, total_len)
    bot_start = min_idx
    for i in range(min_idx + 1, rise_search_limit):
        if not np.isnan(temp[i]) and temp[i] >= min_val + 3:
            bot_start = i
            break
    return max(t5 + 1, min(bot_start, total_len - 1))

def find_bottom_mixer_end(temp, power, t_bot_start):
    """Find the index where BOTTOM MIXING ends"""
    total_len = len(temp)
    if t_bot_start >= total_len - 5:
        return total_len
    power_end = total_len
    if power is not None and len(power) >= total_len:
        bot_power = power[t_bot_start:total_len]
        power_threshold = 50
        last_active = -1
        for i in range(len(bot_power)):
            if not np.isnan(bot_power[i]) and bot_power[i] >= power_threshold:
                last_active = i
        if last_active >= 0:
            power_end = t_bot_start + last_active + 1
    bot_section_temp = temp[t_bot_start:total_len]
    diff = np.diff(bot_section_temp)
    best_i = -1
    best_drop = -5
    for i in range(len(diff)):
        if not np.isnan(diff[i]) and diff[i] < best_drop:
            best_drop = diff[i]
            best_i = i
    temp_end = (t_bot_start + best_i + 1) if best_i >= 0 else total_len
    candidates = [x for x in [power_end, temp_end] if x > t_bot_start + 5]
    if candidates:
        return min(min(candidates), total_len)
    return total_len

def remove_dropout_zeros(arr, threshold):
    """Replace isolated drops to 0.0 (consecutive runs of length <= 3) with NaN"""
    if len(arr) == 0:
        return arr
    is_zero = (arr < threshold) | np.isnan(arr)
    cleaned = arr.copy()
    n = len(arr)
    i = 0
    while i < n:
        if is_zero[i]:
            j = i
            while j < n and is_zero[j]:
                j += 1
            run_length = j - i
            if run_length <= 3:
                # Isolated zero run (dropout anomaly) -> set to NaN for interpolation
                cleaned[i:j] = np.nan
            i = j
        else:
            i += 1
    return cleaned

def fill_nan_series(arr):
    """Interpolate and forward/backward fill single time-series arrays"""
    if len(arr) == 0:
        return arr
    s = pd.Series(arr)
    s = s.interpolate(method='linear', limit_direction='both').ffill().bfill().fillna(0.0)
    return s.values

def resample_segment(segment, num_points):
    """Resample a 1D array to a fixed number of points using linear interpolation"""
    n = len(segment)
    if n == 0:
        return np.zeros(num_points)
    xp = np.arange(n)
    x_new = np.linspace(0, n - 1, num_points)
    return np.interp(x_new, xp, segment)

def is_curve_physically_valid(temp, power, torque, speed, ram, stages):
    """Check physical properties of the mixing curves to detect anomalies"""
    # 1. Stage duration checks
    for s, e, stage_name in stages:
        duration = e - s
        if duration < 3:
            return False, f"Stage {stage_name} duration too short: {duration}s"
        if duration > 300:
            return False, f"Stage {stage_name} duration too long: {duration}s"

    # 2. Temperature sanity checks (must be in range [15°C, 250°C])
    if np.any(temp < 15.0) or np.any(temp > 250.0):
        return False, f"Temperature out of range: min={np.min(temp):.1f}°C, max={np.max(temp):.1f}°C"
        
    # Temperature should rise from loading to PID
    stage_temps = {}
    for s, e, stage_name in stages:
        stage_temps[stage_name] = np.mean(temp[s:e])
    if 'Stage5_PID' in stage_temps and 'Stage1_Loading' in stage_temps:
        if stage_temps['Stage5_PID'] <= stage_temps['Stage1_Loading'] + 10.0:
            return False, f"Temperature does not rise properly: Loading mean={stage_temps['Stage1_Loading']:.1f}, PID mean={stage_temps['Stage5_PID']:.1f}"

    # 3. Rotor Speed checks in active mixing stages (should be > 10 RPM)
    for s, e, stage_name in stages:
        if stage_name in ["Stage2_DryMixing", "Stage3_OilLoading", "Stage4_WetMixing", "Stage5_PID"]:
            sub_speed = speed[s:e]
            if np.mean(sub_speed) < 10.0:
                return False, f"Rotor speed in {stage_name} is too low (mean={np.mean(sub_speed):.1f} RPM)"

    # 4. Power and Torque checks in active mixing stages
    for s, e, stage_name in stages:
        if stage_name in ["Stage2_DryMixing", "Stage4_WetMixing", "Stage5_PID"]:
            sub_power = power[s:e]
            sub_torque = torque[s:e]
            if np.mean(sub_power) < 10.0 or np.mean(sub_torque) < 10.0:
                return False, f"Power or torque in {stage_name} too low (mean power={np.mean(sub_power):.1f}kW, mean torque={np.mean(sub_torque):.1f})"

    # 5. WayofRam checks in stages 1 to 5 (should not be stuck at flat zero)
    for s, e, stage_name in stages:
        if stage_name in ["Stage1_Loading", "Stage2_DryMixing", "Stage3_OilLoading", "Stage4_WetMixing", "Stage5_PID"]:
            sub_ram = ram[s:e]
            if np.max(sub_ram) < 10.0:
                return False, f"WayofRam sensor in {stage_name} stuck at zero (max={np.max(sub_ram):.1f}mm)"
            if np.std(sub_ram) < 0.01 and np.mean(sub_ram) < 20.0:
                return False, f"WayofRam sensor in {stage_name} flatlined (std={np.std(sub_ram):.3f})"

    return True, "Valid"

def remove_mny_anomalies(df):
    """
    Remove MNY target outliers:
    1. Within-Order Variance check: If an OrderID has multiple batches, and the MNY range (max - min) 
       within the same order is > 10.0 MNY, we drop ALL batches in that order.
    2. Compound Family Global Outlier check: For each compound family, we compute the median MNY 
       and Median Absolute Deviation (MAD). We define a robust standard deviation = 1.4826 * MAD (capped at min 1.5).
       We drop any batch whose MNY deviates from the family median by more than:
       threshold = max(5.0, min(2.5 * robust_std, 10.0))
    """
    initial_count = len(df)
    
    def get_compound_family(name):
        if not isinstance(name, str):
            return 'Unknown'
        prefix = name.split()[0] if name.split() else name
        return prefix.rstrip('-')
        
    df = df.copy().reset_index(drop=True)
    df['compound_family'] = df['CompoundName'].apply(get_compound_family)
    
    # 1. Within-Order Variance Check - Drop entire orders with range > 10.0 MNY
    order_ranges = df.groupby('OrderID')['MNY'].agg(lambda x: x.max() - x.min())
    anomalous_orders = order_ranges[order_ranges > 10.0].index.tolist()
    
    drop_indices_order = df[df['OrderID'].isin(anomalous_orders)].index.tolist()
    for idx in drop_indices_order:
        row = df.loc[idx]
        print(f"OrderID {row['OrderID']} has huge MNY fluctuation (>10 MNY). Dropping batch (BatchNumber={row['BatchNumber']}, MNY={row['MNY']:.2f})")
        
    df_step1 = df.drop(index=drop_indices_order).copy()
    
    # 2. Compound Family Robust Global Outlier Check
    drop_indices_family = []
    family_groups = df_step1.groupby('compound_family')
    for family, group in family_groups:
        if len(group) == 0:
            continue
        median_val = group['MNY'].median()
        mad = np.median(np.abs(group['MNY'] - median_val))
        robust_std = 1.4826 * mad
        robust_std = max(robust_std, 1.5)
        
        threshold = max(5.0, min(2.5 * robust_std, 10.0))
        
        for idx, row in group.iterrows():
            if abs(row['MNY'] - median_val) > threshold:
                drop_indices_family.append(idx)
                print(f"Compound Family {family} Global Outlier (threshold={threshold:.2f}, robust_std={robust_std:.2f}). Dropping batch (OrderID={row['OrderID']}, BatchNumber={row['BatchNumber']}, MNY={row['MNY']:.2f}, Family Median={median_val:.2f})")
                
    all_drop_indices = drop_indices_order + drop_indices_family
    df_clean = df.drop(index=all_drop_indices).reset_index(drop=True)
    
    print(f"\n[MNY Label Cleaning Summary] Cleaned {len(all_drop_indices)} anomalous MNY rows. Labeled rows reduced from {initial_count} to {len(df_clean)}.\n")
    return df_clean

def main():
    print(f"Reading dataset: {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} does not exist!")
        return

    cols_to_use = [
        'OrderID', 'BatchNumber', 'CompoundName', 'Target_Temperature', 'OrderStartTime', 'test_result_start_time',
        'weight_pct_oil', 'temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam',
        'CurrentValue', 'PrevStepValue', 'MNY', 'batch_information_fk_final',
        'weight_pct_solid_elastomer', 'weight_pct_natural_rubber', 'weight_pct_silica',
        'weight_pct_silian', 'weight_pct_carbon_black', 'supplier_rubber_viscosity_avg',
        'supplier_silica_moisture_avg', 'supplier_silica_surface_area_avg',
        'supplier_carbon_black_structure_avg', 'supplier_carbon_black_surface_area_avg',
        'supplier_carbon_black_moisture_avg',
        'Top_Fill_Factor', 'Bot_Fill_Factor'
    ]
    df = pd.read_csv(INPUT_CSV, usecols=cols_to_use, low_memory=False)
    print(f"Loaded {len(df)} total rows from CSV.")

    # Filter to labeled rows only
    df_labeled = df.dropna(subset=['MNY']).copy()
    print(f"Filtered to {len(df_labeled)} labeled rows.")

    # Filter ONLY for M1 batches
    df_labeled = df_labeled[df_labeled['CompoundName'].str.contains('M1', na=False)].copy()
    print(f"Filtered to {len(df_labeled)} M1-only labeled rows.")

    # USER INSTRUCTION: Remove MNY target outliers (Intra-Order & Family-level Global Outliers)
    df_labeled = remove_mny_anomalies(df_labeled)

    # 1. Physical Sanity Data Cleaning
    valid_batches = []
    anomaly_counts = {}
    
    count = 0
    total_labeled = len(df_labeled)
    
    for idx, row in df_labeled.iterrows():
        count += 1
        if count % 2000 == 0 or count == total_labeled:
            print(f"Physically validating batch {count}/{total_labeled}...")
            
        temp = hex_to_series(row['temp'])
        power = hex_to_series(row['power'])
        torque = hex_to_series(row['Torque'])
        speed = hex_to_series(row['RotorSpeed'])
        ram = hex_to_series(row['WayofRam'])
        
        tp_len = min(len(temp), len(power))
        if tp_len < 10:
            anomaly_counts["Curve too short"] = anomaly_counts.get("Curve too short", 0) + 1
            continue
            
        temp = temp[:tp_len]
        power = power[:tp_len]
        
        torque_full = np.full(tp_len, np.nan)
        torque_full[:min(len(torque), tp_len)] = torque[:min(len(torque), tp_len)]
        torque = torque_full
        
        speed_full = np.full(tp_len, np.nan)
        speed_full[:min(len(speed), tp_len)] = speed[:min(len(speed), tp_len)]
        speed = speed_full
        
        ram_full = np.full(tp_len, np.nan)
        ram_full[:min(len(ram), tp_len)] = ram[:min(len(ram), tp_len)]
        ram = ram_full

        # Set values > MAX_VALID_VALUE to NaN
        for arr in [temp, power, torque, speed, ram]:
            arr[arr > MAX_VALID_VALUE] = np.nan
            
        # Clean sudden dropout drops (exact zeros/near-zeros of short duration)
        temp = remove_dropout_zeros(temp, 15.0)
        power = remove_dropout_zeros(power, 5.0)
        torque = remove_dropout_zeros(torque, 5.0)
        speed = remove_dropout_zeros(speed, 2.0)
        ram = remove_dropout_zeros(ram, 5.0)
        
        # Clean series by interpolating and forward-backward filling
        temp = fill_nan_series(temp)
        power = fill_nan_series(power)
        torque = fill_nan_series(torque)
        speed = fill_nan_series(speed)
        ram = fill_nan_series(ram)
        
        target_temp = float(row['Target_Temperature']) if pd.notna(row['Target_Temperature']) else 140.0
        w_oil = float(row['weight_pct_oil']) if pd.notna(row['weight_pct_oil']) else 0.0
        has_oil = pd.notna(row['PrevStepValue']) and pd.notna(row['CurrentValue']) and w_oil > 0.0
        is_oil_loading_present = 1.0 if has_oil else 0.0
        
        t1 = find_ram_first_drop(ram)
        
        if has_oil:
            oil_start = int(round(row['PrevStepValue']))
            oil_end = int(round(row['CurrentValue']))
            
            if oil_start >= tp_len - 15: oil_start = tp_len - 15
            if oil_end >= tp_len - 10: oil_end = tp_len - 10
            
            t2 = oil_start
            t3 = oil_end
            
            if t2 <= t1: t2 = t1 + 5
            if t3 <= t2: t3 = t2 + 5
            
            t4 = find_temp_pid_start(temp, target_temp, t3)
            if t4 <= t3: t4 = t3 + 5
            
            t5 = find_temp_discharge_start(temp, t4)
            if t5 <= t4: t5 = t4 + 5
            
            t_bot_start = find_bottom_mixer_start(temp, t5)
            
            # Backwards capping
            if t5 > tp_len: t5 = tp_len
            if t4 > t5 - 2: t4 = t5 - 2
            if t3 > t4 - 2: t3 = t4 - 2
            if t2 > t3 - 2: t2 = t3 - 2
            if t1 > t2 - 2: t1 = t2 - 2
            
            if t1 < 0: t1 = 0
            if t2 < t1: t2 = t1
            if t3 < t2: t3 = t2
            if t4 < t3: t4 = t3
            if t5 < t4: t5 = t4
            
            if t_bot_start < t5: t_bot_start = t5
            if t_bot_start > tp_len: t_bot_start = tp_len
            
            t_bot_end = find_bottom_mixer_end(temp, power, t_bot_start)
            if t_bot_end <= t_bot_start: t_bot_end = tp_len
            if t_bot_end > tp_len: t_bot_end = tp_len
            
            stages = [
                (0, t1, "Stage1_Loading"),
                (t1, t2, "Stage2_DryMixing"),
                (t2, t3, "Stage3_OilLoading"),
                (t3, t4, "Stage4_WetMixing"),
                (t4, t5, "Stage5_PID"),
                (t_bot_start, t_bot_end, "Stage6_BottomMixing")
            ]
        else:
            t_pid = find_temp_pid_start(temp, target_temp, t1)
            if t_pid <= t1: t_pid = t1 + 5
            
            t5 = find_temp_discharge_start(temp, t_pid)
            if t5 <= t_pid: t5 = t_pid + 5
            
            t_bot_start = find_bottom_mixer_start(temp, t5)
            
            if t5 > tp_len: t5 = tp_len
            if t_pid > t5 - 2: t_pid = t5 - 2
            if t1 > t_pid - 2: t1 = t_pid - 2
            
            if t1 < 0: t1 = 0
            if t_pid < t1: t_pid = t1
            if t5 < t_pid: t5 = t_pid
            
            if t_bot_start < t5: t_bot_start = t5
            if t_bot_start > tp_len: t_bot_start = tp_len
            
            t_bot_end = find_bottom_mixer_end(temp, power, t_bot_start)
            if t_bot_end <= t_bot_start: t_bot_end = tp_len
            if t_bot_end > tp_len: t_bot_end = tp_len
            
            stages = [
                (0, t1, "Stage1_Loading"),
                (t1, t_pid, "Stage2_DryMixing"),
                (t_pid, t5, "Stage5_PID"),
                (t_bot_start, t_bot_end, "Stage6_BottomMixing")
            ]
            
        # Curve segment validation
        is_valid, reason = is_curve_physically_valid(temp, power, torque, speed, ram, stages)
        if not is_valid:
            key = reason.split(":")[0]
            anomaly_counts[key] = anomaly_counts.get(key, 0) + 1
            continue
            
        # Resample each stage to K_POINTS
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
                
        # Stack features
        var_list = ['temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam']
        feature_vector = []
        for s_info in stages:
            stage_name = s_info[2]
            for var in var_list:
                feature_vector.append(resampled_features[f"{stage_name}_{var}"])
                
        feature_vector = np.concatenate(feature_vector)
        
        # Static recipe features
        recipe_dict = {
            'weight_pct_solid_elastomer': float(row['weight_pct_solid_elastomer']) if pd.notna(row['weight_pct_solid_elastomer']) else 0.0,
            'weight_pct_natural_rubber': float(row['weight_pct_natural_rubber']) if pd.notna(row['weight_pct_natural_rubber']) else 0.0,
            'weight_pct_silica': float(row['weight_pct_silica']) if pd.notna(row['weight_pct_silica']) else 0.0,
            'weight_pct_oil': w_oil,
            'weight_pct_silian': float(row['weight_pct_silian']) if pd.notna(row['weight_pct_silian']) else 0.0,
            'weight_pct_carbon_black': float(row['weight_pct_carbon_black']) if pd.notna(row['weight_pct_carbon_black']) else 0.0,
            'supplier_rubber_viscosity_avg': float(row['supplier_rubber_viscosity_avg']) if pd.notna(row['supplier_rubber_viscosity_avg']) else 0.0,
            'supplier_silica_moisture_avg': float(row['supplier_silica_moisture_avg']) if pd.notna(row['supplier_silica_moisture_avg']) else 0.0,
            'supplier_silica_surface_area_avg': float(row['supplier_silica_surface_area_avg']) if pd.notna(row['supplier_silica_surface_area_avg']) else 0.0,
            'supplier_carbon_black_structure_avg': float(row['supplier_carbon_black_structure_avg']) if pd.notna(row['supplier_carbon_black_structure_avg']) else 0.0,
            'supplier_carbon_black_surface_area_avg': float(row['supplier_carbon_black_surface_area_avg']) if pd.notna(row['supplier_carbon_black_surface_area_avg']) else 0.0,
            'supplier_carbon_black_moisture_avg': float(row['supplier_carbon_black_moisture_avg']) if pd.notna(row['supplier_carbon_black_moisture_avg']) else 0.0,
            'Top_Fill_Factor': float(row['Top_Fill_Factor']) if pd.notna(row['Top_Fill_Factor']) else 0.0,
            'Bot_Fill_Factor': float(row['Bot_Fill_Factor']) if pd.notna(row['Bot_Fill_Factor']) else 0.0,
        }
        
        batch_info = {
            'OrderID': str(row['OrderID']),
            'BatchNumber': int(row['BatchNumber']),
            'CompoundName': row['CompoundName'],
            'batch_information_fk_final': str(row['batch_information_fk_final']),
            'OrderStartTime': str(row['OrderStartTime']),
            'test_result_start_time': str(row['test_result_start_time']) if pd.notnull(row['test_result_start_time']) else None,
            'MNY': float(row['MNY']),
            'weight_pct_oil': w_oil,
            'is_oil_loading_present': is_oil_loading_present,
            'features': feature_vector,
            'recipe_features': np.array(list(recipe_dict.values()))
        }
        valid_batches.append(batch_info)
        
    print(f"\n[Validation Step] M1 physically valid batches: {len(valid_batches)} / {total_labeled}")

    # 2. Isolation Forest Outlier Detection (Separate by Track)
    with_oil_valid = [b for b in valid_batches if b['is_oil_loading_present'] == 1]
    without_oil_valid = [b for b in valid_batches if b['is_oil_loading_present'] == 0]
    
    cleaned_batches = []
    dropped_by_if = []
    
    # Run Isolation Forest on With-Oil track
    if len(with_oil_valid) > 10:
        print(f"Fitting Isolation Forest on With-Oil track ({len(with_oil_valid)} batches)...")
        X_with = np.array([b['features'] for b in with_oil_valid])
        clf_with = IsolationForest(contamination=IF_CONTAMINATION, random_state=42)
        if_preds_with = clf_with.fit_predict(X_with)
        scores_with = clf_with.decision_function(X_with)
        
        for i, b in enumerate(with_oil_valid):
            if if_preds_with[i] == -1:
                dropped_by_if.append({
                    'OrderID': b['OrderID'],
                    'BatchNumber': b['BatchNumber'],
                    'CompoundName': b['CompoundName'],
                    'Track': 'With-Oil',
                    'IF_Score': scores_with[i]
                })
            else:
                cleaned_batches.append(b)
    else:
        cleaned_batches.extend(with_oil_valid)
        
    # Run Isolation Forest on Without-Oil track
    if len(without_oil_valid) > 10:
        print(f"Fitting Isolation Forest on Without-Oil track ({len(without_oil_valid)} batches)...")
        X_without = np.array([b['features'] for b in without_oil_valid])
        clf_without = IsolationForest(contamination=IF_CONTAMINATION, random_state=42)
        if_preds_without = clf_without.fit_predict(X_without)
        scores_without = clf_without.decision_function(X_without)
        
        for i, b in enumerate(without_oil_valid):
            if if_preds_without[i] == -1:
                dropped_by_if.append({
                    'OrderID': b['OrderID'],
                    'BatchNumber': b['BatchNumber'],
                    'CompoundName': b['CompoundName'],
                    'Track': 'Without-Oil',
                    'IF_Score': scores_without[i]
                })
            else:
                cleaned_batches.append(b)
    else:
        cleaned_batches.extend(without_oil_valid)

    # 3. Output reports
    print(f"\n==================== DATA CLEANING SUMMARY (M1 ONLY) ====================")
    print(f"Total M1 labeled input rows: {total_labeled}")
    print(f"Discarded by physical rules: {total_labeled - len(valid_batches)} batches")
    print("Physical drop reasons breakdown:")
    for reason, count_val in anomaly_counts.items():
        print(f"  - {reason:40s}: {count_val} batches")
        
    print(f"Discarded by Isolation Forest: {len(dropped_by_if)} batches")
    print(f"Final training dataset size (M1): {len(cleaned_batches)} batches")
    print("=========================================================================\n")
    
    # Save anomalies report
    os.makedirs(os.path.dirname(ANOMALY_CSV), exist_ok=True)
    if dropped_by_if:
        dropped_df = pd.DataFrame(dropped_by_if)
        dropped_df.to_csv(ANOMALY_CSV, index=False)
        print(f"Saved Isolation Forest anomaly list to: {ANOMALY_CSV}")
        
    # Save cleaned features to joblib
    print(f"Saving cleaned & preprocessed dataset to: {OUTPUT_JOBLIB}")
    joblib.dump(cleaned_batches, OUTPUT_JOBLIB)
    print("Done!")

if __name__ == "__main__":
    main()
