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
import sys
import json
import pyodbc


# Configure stdout to use utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
INPUT_CSV = os.path.join(WORKSPACE_ROOT, "stage_statistics_enriched.csv")
OUTPUT_CSV = "stage_statistics_enriched_all_features_weather_v4.csv"

HEX_STEP = 4
MAX_VALID_VALUE = 10000
PROCESS_LABELED_ONLY = True  # Set to True to only decode and segment batches with MNY test labels (faster)

# ================== Tool Functions ==================


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
    """
    Find the index of the TOP-MIXING DISCHARGE after pid_start.
    This is the point where the temperature drops sharply and DEEPLY (>= 15°C drop in a few seconds),
    marking the end of top mixing and rubber falling out.
    We look for the LARGEST single-step drop >= 15°C after pid_start to avoid false positives
    from small oscillations during mixing.
    """
    if pid_start >= len(temp) - 2:
        return len(temp) - 1
    diff = np.diff(temp)
    # Find the biggest drop after pid_start — that's the real discharge
    best_i = -1
    best_drop = -15  # must drop at least 15°C in one step to count
    for i in range(pid_start + 1, len(diff)):
        if not np.isnan(diff[i]) and diff[i] < best_drop:
            best_drop = diff[i]
            best_i = i
    if best_i >= 0:
        return best_i
    # Fallback: first drop >= 5°C
    for i in range(pid_start + 1, len(diff)):
        if not np.isnan(diff[i]) and diff[i] < -5:
            return i
    return max(pid_start + 1, len(temp) - 5)


def find_bottom_mixer_start(temp, t5):
    """
    Find the index where BOTTOM MIXING starts — when temperature begins rising after discharge.
    After top discharge (t5), temperature drops to a minimum (transfer period).
    When temperature starts rising again, bottom mixing has begun.
    Strategy:
      1. Find the temperature minimum in [t5, t5+80] (transfer lasts up to ~60-80s)
      2. From the minimum, find the first index where temp >= min + 3 (rising has started)
    """
    total_len = len(temp)
    if t5 >= total_len - 3:
        return total_len - 1

    # Extended window to find the true minimum after discharge (up to 80s)
    search_limit = min(t5 + 80, total_len)
    min_val = temp[t5] if not np.isnan(temp[t5]) else np.inf
    min_idx = t5
    for i in range(t5, search_limit):
        val = temp[i]
        if not np.isnan(val) and val < min_val:
            min_val = val
            min_idx = i

    # From the minimum, find the first point where temp rises >= 3°C above min
    rise_search_limit = min(min_idx + 100, total_len)
    bot_start = min_idx  # default: minimum itself
    for i in range(min_idx + 1, rise_search_limit):
        if not np.isnan(temp[i]) and temp[i] >= min_val + 3:
            bot_start = i
            break

    return max(t5 + 1, min(bot_start, total_len - 1))


def find_bottom_mixer_end(temp, power, t_bot_start):
    """
    Find the index where BOTTOM MIXING ends.
    Key signals:
      1. POWER: After bottom discharge, power drops to near zero (machine stops doing work).
                Find the last index in [t_bot_start, end] where power is still active (>= 50).
      2. TEMPERATURE: Fallback — find the largest temperature drop >= 5°C in [t_bot_start, end].
    The bottom mixing end is the EARLIER of the two signals (whichever fires first from the right).
    """
    total_len = len(temp)
    if t_bot_start >= total_len - 5:
        return total_len

    # ---- Signal 1: Power drops to near zero (primary) ----
    # Find the last index where power is meaningfully active (>= 50 kW or units)
    power_end = total_len  # default
    if power is not None and len(power) >= total_len:
        bot_power = power[t_bot_start:total_len]
        # Use a threshold of 50 (power below this = machine stopped)
        power_threshold = 50
        last_active = -1
        for i in range(len(bot_power)):
            if not np.isnan(bot_power[i]) and bot_power[i] >= power_threshold:
                last_active = i
        if last_active >= 0:
            power_end = t_bot_start + last_active + 1  # end = just after last active point

    # ---- Signal 2: Largest temperature drop >= 5°C in bottom section (secondary) ----
    bot_section_temp = temp[t_bot_start:total_len]
    diff = np.diff(bot_section_temp)
    best_i = -1
    best_drop = -5
    for i in range(len(diff)):
        if not np.isnan(diff[i]) and diff[i] < best_drop:
            best_drop = diff[i]
            best_i = i
    temp_end = (t_bot_start + best_i + 1) if best_i >= 0 else total_len

    # Use the earlier of the two (but ensure it's after t_bot_start by a meaningful margin)
    candidates = [x for x in [power_end, temp_end] if x > t_bot_start + 5]
    if candidates:
        return min(min(candidates), total_len)
    return total_len





def fill_anomalies_by_compound(decoded_data, column_name):
    """Fill anomalies (> MAX_VALID_VALUE) by the compound-wise average at the same time step"""
    compound_groups = {}
    for i, d in enumerate(decoded_data):
        compound = d['Compound']
        if compound not in compound_groups:
            compound_groups[compound] = []
        compound_groups[compound].append(i)
    
    for compound, indices in compound_groups.items():
        if len(indices) < 2:
            continue
        max_len = max(len(decoded_data[i][column_name]) for i in indices)
        for time_idx in range(max_len):
            values = []
            for i in indices:
                arr = decoded_data[i][column_name]
                if time_idx < len(arr) and not np.isnan(arr[time_idx]) and arr[time_idx] <= MAX_VALID_VALUE:
                    values.append(arr[time_idx])
            
            if values:
                mean_val = np.mean(values)
                for i in indices:
                    arr = decoded_data[i][column_name]
                    if time_idx < len(arr) and (np.isnan(arr[time_idx]) or arr[time_idx] > MAX_VALID_VALUE):
                        arr[time_idx] = mean_val


# ================== Main Program ==================

def get_silica_phr_mapping():
    """Query SFEPLANT and SFEPLANT_ARCHIVE for silica phr in recipe materials"""
    credentials_path = os.path.join(PARENT_DIR, 'data_processing', 'credentials.json')
    if not os.path.exists(credentials_path):
        print("Warning: credentials.json not found in 'data processing/', using default silica_phr = 0.0")
        return pd.DataFrame(columns=['RecipeID', 'silica_phr'])
        
    with open(credentials_path, 'r', encoding='utf-8') as f:
        creds = json.load(f)
        
    c = creds['HF_MMS']
    
    def connect(db):
        conn_str = (
            f"DRIVER={c['driver']};"
            f"SERVER={c['server']};"
            f"DATABASE={db};"
            f"UID={c['username']};"
            f"PWD={c['password']};"
            "Encrypt=yes;"
            "TrustServerCertificate=yes;"
        )
        return pyodbc.connect(conn_str)
        
    sql = """
    SELECT RecipeID, SUM(CAST(Pphr AS FLOAT)) AS silica_phr
    FROM (
        SELECT RecipeID, Pphr 
        FROM SFEPLANT.dbo.RecipeMaterials 
        WHERE MaterialCode LIKE 'CS10%' OR MaterialCode LIKE 'CS12%'
        UNION ALL
        SELECT RecipeID, Pphr 
        FROM SFEPLANT_ARCHIVE.dbo.RecipeMaterials 
        WHERE MaterialCode LIKE 'CS10%' OR MaterialCode LIKE 'CS12%'
    ) as sub
    GROUP BY RecipeID
    """
    
    try:
        conn = connect('SFEPLANT')
        df = pd.read_sql(sql, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error querying silica phr mapping: {e}")
        return pd.DataFrame(columns=['RecipeID', 'silica_phr'])


def run_segmentation():
    print(f"Reading enriched dataset from: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} does not exist!")
        return
        
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Loaded {len(df)} total rows.")
    
    # Query and merge silica phr mapping
    print("Querying silica phr mapping for recipes...")
    silica_phr_df = get_silica_phr_mapping()
    print(f"Retrieved {len(silica_phr_df)} recipes with silica phr.")
    
    if len(silica_phr_df) > 0 and 'CompoundDescription' in df.columns:
        df = pd.merge(df, silica_phr_df.rename(columns={'RecipeID': 'CompoundDescription'}), on='CompoundDescription', how='left')
    
    if 'silica_phr' in df.columns:
        df['silica_phr'] = df['silica_phr'].fillna(0.0)
    else:
        df['silica_phr'] = 0.0
        
    # Keep all compounds (no filtering of silane or silica phr)
    df_all = df.copy()
    print(f"Keeping all compounds for modeling: {len(df_all)} rows.")

    if PROCESS_LABELED_ONLY:
        df_all = df_all.dropna(subset=['MNY'])
        print(f"Filtered to labeled rows only for modeling: {len(df_all)} rows.")
    
    # Do not drop rows where oil loading times are missing since we support non-oil recipes now
    print(f"Prepared {len(df_all)} rows for segmenting.")
    
    # Extract OrderStartTime locally from input dataset and get Hefei Weather data
    print("Extracting OrderStartTime from input dataset...")
    try:
        order_times_df = df_all[['OrderID', 'OrderStartTime']].dropna().drop_duplicates(subset=['OrderID'])
        print(f"Extracted {len(order_times_df)} OrderStartTimes.")
        
        # Fetch weather data
        sys.path.append('data processing')
        from function_definitions_M1 import get_hefei_weather
        print("Fetching Hefei High-tech Zone weather data from Open-Meteo...")
        weather_df = get_hefei_weather(order_times_df['OrderStartTime'])
        
        # Merge weather data into order times
        order_times_df['date'] = pd.to_datetime(order_times_df['OrderStartTime']).dt.date
        weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date
        order_times_with_weather = pd.merge(order_times_df, weather_df, on='date', how='left')
        order_times_with_weather.drop(columns=['date', 'OrderStartTime'], inplace=True, errors='ignore')
        
        # Merge into df_all
        order_times_with_weather['OrderID'] = order_times_with_weather['OrderID'].astype(str).str.strip()
        df_all['OrderID'] = df_all['OrderID'].astype(str).str.strip()
        df_all = pd.merge(df_all, order_times_with_weather, on='OrderID', how='left')
        print("Successfully merged weather data (temperature & humidity) into dataset.")
    except Exception as e:
        print(f"Error processing weather or order times: {e}")
        
    if len(df_all) == 0:
        print("No rows to process after filtering.")
        return
        
    # 2. Decode hex curves
    print("Decoding curves to series...")
    decoded = []
    for _, row in df_all.iterrows():
        batch_id = str(row['batch_information_fk_final'])
        
        temp = hex_to_series(row['temp'])
        power = hex_to_series(row['power'])
        torque = hex_to_series(row['Torque'])
        speed = hex_to_series(row['RotorSpeed'])
        ram = hex_to_series(row['WayofRam'])
        
        # temp and power contain the FULL curve (top mixing + bottom mixing concatenated).
        # WayofRam, Torque, RotorSpeed only cover the TOP MIXING section (ram is not used 
        # in bottom mixing), so we must NOT truncate temp/power to their shorter lengths.
        # Use temp and power together as the full curve length.
        tp_len = min(len(temp), len(power))  # full curve length (top + bottom mixing)
        if tp_len == 0:
            continue
        
        # Other signals are only valid for the top mixing portion — keep their own length
        # but cap them at tp_len to avoid index errors
        torque_len = min(len(torque), tp_len)
        speed_len = min(len(speed), tp_len)
        ram_len = min(len(ram), tp_len)
        
        temp = temp[:tp_len]
        power = power[:tp_len]
        torque_full = np.full(tp_len, np.nan)
        torque_full[:torque_len] = torque[:torque_len]
        torque = torque_full
        speed_full = np.full(tp_len, np.nan)
        speed_full[:speed_len] = speed[:speed_len]
        speed = speed_full
        ram_full = np.full(tp_len, np.nan)
        ram_full[:ram_len] = ram[:ram_len]
        ram = ram_full
        
        min_len = tp_len  # keep for compatibility

        
        # Determine target temperature
        target_temp = float(row['Target_Temperature']) if ('Target_Temperature' in row and pd.notna(row['Target_Temperature'])) else 140.0
        
        data = {
            "row_data": row.to_dict(),
            "batch_information_fk_final": batch_id,
            "OrderID": row['OrderID'],
            "BatchNumber": row['BatchNumber'],
            "Compound": row['CompoundName'],
            "Target_Temperature": target_temp,
            "temp": temp,
            "power": power,
            "Torque": torque,
            "RotorSpeed": speed,
            "WayofRam": ram,

            "PrevStepValue": row['PrevStepValue'],
            "CurrentValue": row['CurrentValue']
        }
        decoded.append(data)
        
    # Process anomalies (> 10000)
    print("Processing anomalies...")
    for d in decoded:
        for col in ['temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam']:
            arr = d[col]
            if len(arr) == 0:
                continue
            arr[arr > MAX_VALID_VALUE] = np.nan
            
    # Compound-wise average filling
    print("Filling anomalies by compound averages...")
    for col in ['temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam']:
        fill_anomalies_by_compound(decoded, col)
        
    # 3. Stage boundary calculation and physical feature extraction
    print("Segmenting curves and extracting physical features...")
    feature_rows = []
    
    for idx, d in enumerate(decoded):
        temp = d['temp']
        power = d['power']
        torque = d['Torque']
        speed = d['RotorSpeed']
        ram = d['WayofRam']
        target_temp = d['Target_Temperature']
        total_len = len(temp)
        
        if total_len < 10:
            continue
            
        # Determine if recipe has oil loading
        w_oil = float(d['row_data'].get('weight_pct_oil', 0.0)) if pd.notna(d['row_data'].get('weight_pct_oil')) else 0.0
        has_oil = pd.notna(d['PrevStepValue']) and pd.notna(d['CurrentValue']) and w_oil > 0.0
        
        t1 = find_ram_first_drop(ram)
        
        if has_oil:
            is_oil_loading_present = 1.0
            oil_start = int(round(d['PrevStepValue']))
            oil_end = int(round(d['CurrentValue']))
            
            # Cap database values if they exceed total curve length
            if oil_start >= total_len - 15: oil_start = total_len - 15
            if oil_end >= total_len - 10: oil_end = total_len - 10
            
            t2 = oil_start
            t3 = oil_end
            
            # Forward constraints
            if t2 <= t1: t2 = t1 + 5
            if t3 <= t2: t3 = t2 + 5
            
            t4 = find_temp_pid_start(temp, target_temp, t3)
            if t4 <= t3: t4 = t3 + 5
            
            t5 = find_temp_discharge_start(temp, t4)
            if t5 <= t4: t5 = t4 + 5
            
            # Find bottom mixer start index (B1 start = temperature rising after discharge dip)
            t_bot_start = find_bottom_mixer_start(temp, t5)
            
            # Backwards capping to ensure all boundaries fit within total_len
            if t5 > total_len: t5 = total_len
            if t4 > t5 - 2: t4 = t5 - 2
            if t3 > t4 - 2: t3 = t4 - 2
            if t2 > t3 - 2: t2 = t3 - 2
            if t1 > t2 - 2: t1 = t2 - 2
            
            # Ensure none goes below 0
            if t1 < 0: t1 = 0
            if t2 < t1: t2 = t1
            if t3 < t2: t3 = t2
            if t4 < t3: t4 = t3
            if t5 < t4: t5 = t4
            
            # Ensure t_bot_start is after t5 and doesn't exceed total_len
            if t_bot_start < t5: t_bot_start = t5
            if t_bot_start > total_len: t_bot_start = total_len
            
            # Find bottom mixer end (power drop to zero = primary signal, temp drop = secondary)
            t_bot_end = find_bottom_mixer_end(temp, power, t_bot_start)
            if t_bot_end <= t_bot_start: t_bot_end = total_len
            if t_bot_end > total_len: t_bot_end = total_len
            
            stages = {
                "Stage1_Loading": (0, t1),
                "Stage2_DryMixing": (t1, t2),
                "Stage3_OilLoading": (t2, t3),
                "Stage4_WetMixing": (t3, t4),
                "Stage5_PID": (t4, t5),
                "Stage6_BottomMixing": (t_bot_start, t_bot_end)
            }
        else:
            is_oil_loading_present = 0.0
            # For recipes without oil loading, there is no Oil Loading or Wet Mixing stage.
            # Dry Mixing goes all the way from t1 to the PID start temperature search.
            t_pid = find_temp_pid_start(temp, target_temp, t1)
            if t_pid <= t1: t_pid = t1 + 5
            
            t5 = find_temp_discharge_start(temp, t_pid)
            if t5 <= t_pid: t5 = t_pid + 5
            
            # Find bottom mixer start index (B1 start = temperature rising after discharge dip)
            t_bot_start = find_bottom_mixer_start(temp, t5)
            
            # Backwards capping
            if t5 > total_len: t5 = total_len
            if t_pid > t5 - 2: t_pid = t5 - 2
            if t1 > t_pid - 2: t1 = t_pid - 2
            
            if t1 < 0: t1 = 0
            if t_pid < t1: t_pid = t1
            if t5 < t_pid: t5 = t_pid
            
            # Ensure t_bot_start is after t5 and doesn't exceed total_len
            if t_bot_start < t5: t_bot_start = t5
            if t_bot_start > total_len: t_bot_start = total_len
            
            # Find bottom mixer end (power drop to zero = primary signal, temp drop = secondary)
            t_bot_end = find_bottom_mixer_end(temp, power, t_bot_start)
            if t_bot_end <= t_bot_start: t_bot_end = total_len
            if t_bot_end > total_len: t_bot_end = total_len
            
            # For non-oil recipes, set boundaries so OilLoading and WetMixing have duration 0
            t2 = t1
            t3 = t1
            t4 = t_pid
            
            stages = {
                "Stage1_Loading": (0, t1),
                "Stage2_DryMixing": (t1, t_pid),
                "Stage3_OilLoading": (t1, t1),
                "Stage4_WetMixing": (t1, t1),
                "Stage5_PID": (t_pid, t5),
                "Stage6_BottomMixing": (t_bot_start, t_bot_end)
            }
        
        # Build features dict starting with original row metadata
        features = d['row_data'].copy()
        
        # Remove raw hex curve strings to keep file clean
        for col in ['temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam']:
            features.pop(col, None)
            
        # Add boundary indices and oil loading present flag
        features.update({
            "is_oil_loading_present": is_oil_loading_present,
            "idx_t1_Loading": t1,
            "idx_t2_DryMixing": t2,
            "idx_t3_OilLoading": t3,
            "idx_t4_PID_Start": t4,
            "idx_t5_Discharge": t5,
            "idx_t6_BottomMixing_Start": t_bot_start,
            "idx_t7_BottomMixing_End": t_bot_end,
            "idx_total_length": total_len
        })
        
        # A. Basic stage-level statistics and integrals
        for stage_name, (s, e) in stages.items():
            s_idx = int(s)
            e_idx = int(min(e, total_len))
            duration = e_idx - s_idx
            
            features[f"{stage_name}_Duration"] = duration
            
            if duration <= 0:
                # Fill defaults if stage is invalid
                for col in ['temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam', 'eta_torque']:
                    features[f"{stage_name}_{col}_Mean"] = np.nan
                    features[f"{stage_name}_{col}_Std"] = np.nan
                features[f"{stage_name}_RotorSpeed_Integral"] = np.nan
                features[f"{stage_name}_Torque_Integral"] = np.nan
                features[f"{stage_name}_power_Integral"] = np.nan
                continue
                
            seg_temp = temp[s_idx:e_idx]
            seg_power = power[s_idx:e_idx]
            seg_torque = torque[s_idx:e_idx]
            seg_speed = speed[s_idx:e_idx]
            seg_ram = ram[s_idx:e_idx]
            
            # Apparent viscosity for this segment (eta_torque = torque / speed)
            speed_nonzero_seg = np.where(seg_speed > 0, seg_speed, np.nan)
            seg_eta_torque = seg_torque / speed_nonzero_seg
            
            for col_name, arr in [('temp', seg_temp), ('power', seg_power), ('Torque', seg_torque), ('RotorSpeed', seg_speed), ('WayofRam', seg_ram), ('eta_torque', seg_eta_torque)]:
                valid = arr[~np.isnan(arr)]
                features[f"{stage_name}_{col_name}_Mean"] = np.nanmean(arr) if len(valid) > 0 else np.nan
                features[f"{stage_name}_{col_name}_Std"] = np.nanstd(arr) if len(valid) > 0 else np.nan
                
            # Integrals (Shear history and Torque history in this stage)
            valid_speed = seg_speed[~np.isnan(seg_speed)]
            features[f"{stage_name}_RotorSpeed_Integral"] = np.trapezoid(valid_speed, dx=1.0) if len(valid_speed) > 1 else (valid_speed[0] if len(valid_speed) == 1 else 0.0)
            
            valid_torque = seg_torque[~np.isnan(seg_torque)]
            features[f"{stage_name}_Torque_Integral"] = np.trapezoid(valid_torque, dx=1.0) if len(valid_torque) > 1 else (valid_torque[0] if len(valid_torque) == 1 else 0.0)
            
            valid_power = seg_power[~np.isnan(seg_power)]
            features[f"{stage_name}_power_Integral"] = np.trapezoid(valid_power, dx=1.0) if len(valid_power) > 1 else (valid_power[0] if len(valid_power) == 1 else 0.0)

                
        # B. Advanced physical parameters
        
        # 1. Temperature Dimension
        discharge_temp = temp[int(t5)] if t5 < len(temp) else (temp[-1] if len(temp) > 0 else np.nan)
        max_temp = np.nanmax(temp) if len(temp[~np.isnan(temp)]) > 0 else np.nan
        t_max_temp = np.nanargmax(temp) if len(temp[~np.isnan(temp)]) > 0 else np.nan
        init_temp = temp[int(t1)] if t1 < len(temp) else (temp[0] if len(temp) > 0 else np.nan)
        temp_rise_rate = (discharge_temp - init_temp) / total_len if (total_len > 0 and pd.notna(discharge_temp) and pd.notna(init_temp)) else np.nan
        
        valid_temp = temp[~np.isnan(temp)]
        temp_integral = np.trapezoid(valid_temp, dx=1.0) if len(valid_temp) > 1 else 0.0
        
        temp_diff = np.diff(temp)
        temp_change_rate_std = np.nanstd(temp_diff) if len(temp_diff[~np.isnan(temp_diff)]) > 0 else np.nan
        
        features.update({
            "phys_discharge_temp": discharge_temp,
            "phys_max_temp": max_temp,
            "phys_t_max_temp": t_max_temp,
            "phys_init_temp": init_temp,
            "phys_temp_rise_rate": temp_rise_rate,
            "phys_temp_integral": temp_integral,
            "phys_temp_change_rate_std": temp_change_rate_std
        })
        
        # 2. Power & Mechanical Dimension
        valid_power = power[~np.isnan(power)]
        avg_power = np.nanmean(power) if len(valid_power) > 0 else np.nan
        power_integral = np.trapezoid(valid_power, dx=1.0) if len(valid_power) > 1 else 0.0 # total kW-s
        
        peak_power = np.nanmax(power) if len(valid_power) > 0 else np.nan
        t_peak_power = np.nanargmax(power) if len(valid_power) > 0 else np.nan
        
        # Power stability in PID stage (coefficient of variation)
        pid_power = power[int(t4):int(t5)]
        pid_power_valid = pid_power[~np.isnan(pid_power)]
        power_stability_pid = np.nanstd(pid_power) / np.nanmean(pid_power) if len(pid_power_valid) > 2 and np.nanmean(pid_power) != 0 else np.nan
        
        # Max instantaneous drop rate of power
        power_diff = np.diff(power)
        max_power_drop_rate = -np.nanmin(power_diff) if len(power_diff[~np.isnan(power_diff)]) > 0 else np.nan
        
        features.update({
            "phys_avg_power": avg_power,
            "phys_power_integral": power_integral,
            "phys_peak_power": peak_power,
            "phys_t_peak_power": t_peak_power,
            "phys_power_stability_pid": power_stability_pid,
            "phys_max_power_drop_rate": max_power_drop_rate
        })
        
        # Apparent viscosity (eta_app = power / speed^2)
        # Avoid division by zero by filtering speed > 0
        speed_nonzero = np.where(speed > 0, speed, np.nan)
        eta_app = power / (speed_nonzero ** 2)
        eta_app_valid = eta_app[~np.isnan(eta_app)]
        
        features["phys_eta_app_overall_mean"] = np.nanmean(eta_app) if len(eta_app_valid) > 0 else np.nan
        features["phys_eta_app_wetmix_mean"] = np.nanmean(eta_app[int(t3):int(t4)]) if (has_oil and t4 > t3 and len(eta_app[int(t3):int(t4)][~np.isnan(eta_app[int(t3):int(t4)])]) > 0) else np.nan
        features["phys_eta_app_pid_mean"] = np.nanmean(eta_app[int(t4):int(t5)]) if (t5 > t4 and len(eta_app[int(t4):int(t5)][~np.isnan(eta_app[int(t4):int(t5)])]) > 0) else np.nan
        features["phys_eta_app_discharge"] = eta_app[int(t5)] if (t5 < len(eta_app) and not np.isnan(eta_app[int(t5)])) else np.nan
        
        # 3. Time Dimension
        features.update({
            "time_total_mixing": total_len,
            "time_pct_Loading": (t1) / total_len,
            "time_pct_DryMixing": (t2 - t1) / total_len if has_oil else (t4 - t1) / total_len,
            "time_pct_OilLoading": (t3 - t2) / total_len if has_oil else 0.0,
            "time_pct_WetMixing": (t4 - t3) / total_len if has_oil else 0.0,
            "time_pct_PID": (t5 - t4) / total_len,
            "time_pct_Discharge": (total_len - t5) / total_len,
            "time_reach_discharge": t5
        })
        
        # Shear history (speed integral)
        valid_speed = speed[~np.isnan(speed)]
        shear_history = np.trapezoid(valid_speed, dx=1.0) if len(valid_speed) > 1 else 0.0
        features["phys_shear_history_total"] = shear_history
        
        # 4. Process setting and control parameters
        # Ram lifts count
        ram_diff = np.diff(ram)
        ram_lifts = np.sum(ram_diff > 10) # ram lifts are associated with positive change in ram position
        features["setting_ram_lifts_count"] = ram_lifts
        
        # Stable ram position at discharge
        stable_ram_discharge = np.nanmean(ram[int(t5):]) if len(ram[int(t5):][~np.isnan(ram[int(t5):])]) > 0 else np.nan
        features["setting_stable_ram_discharge"] = stable_ram_discharge
        
        # ================== L1-L2-L3 Recipe Fingerprints & Rheology Features ==================
        
        # L1: Recipe weight percentage ratios
        w_solid = float(features.get('weight_pct_solid_elastomer', 0.0)) if pd.notna(features.get('weight_pct_solid_elastomer')) else 0.0
        w_nr = float(features.get('weight_pct_natural_rubber', 0.0)) if pd.notna(features.get('weight_pct_natural_rubber')) else 0.0
        w_cb = float(features.get('weight_pct_carbon_black', 0.0)) if pd.notna(features.get('weight_pct_carbon_black')) else 0.0
        w_silica = float(features.get('weight_pct_silica', 0.0)) if pd.notna(features.get('weight_pct_silica')) else 0.0
        w_oil = float(features.get('weight_pct_oil', 0.0)) if pd.notna(features.get('weight_pct_oil')) else 0.0
        
        total_rubber = w_solid + w_nr
        if total_rubber > 0:
            ratio_nr_rubber = w_nr / total_rubber
            ratio_filler_polymer = (w_cb + w_silica) / total_rubber
            ratio_oil_polymer = w_oil / total_rubber
        else:
            ratio_nr_rubber = 0.0
            ratio_filler_polymer = 0.0
            ratio_oil_polymer = 0.0
            
        total_filler = w_cb + w_silica
        ratio_oil_filler = w_oil / total_filler if total_filler > 0 else 0.0
        
        features.update({
            "ratio_nr_rubber": ratio_nr_rubber,
            "ratio_filler_polymer": ratio_filler_polymer,
            "ratio_oil_polymer": ratio_oil_polymer,
            "ratio_oil_filler": ratio_oil_filler
        })
        
        # L2: Online rheology
        s1_start, s1_end = int(stages["Stage1_Loading"][0]), int(stages["Stage1_Loading"][1])
        if s1_end > s1_start:
            s1_power = power[s1_start:s1_end]
            s1_power_valid = s1_power[~np.isnan(s1_power)]
            s1_power_max = np.nanmax(s1_power_valid) if len(s1_power_valid) > 0 else np.nan
            
            s1_torque = torque[s1_start:s1_end]
            s1_torque_valid = s1_torque[~np.isnan(s1_torque)]
            s1_torque_max = np.nanmax(s1_torque_valid) if len(s1_torque_valid) > 0 else np.nan
        else:
            s1_power_max = np.nan
            s1_torque_max = np.nan
            
        s2_start, s2_end = int(stages["Stage2_DryMixing"][0]), int(stages["Stage2_DryMixing"][1])
        if s2_end > s2_start + 3:
            s2_power = power[s2_start:s2_end]
            s2_power_valid = s2_power[~np.isnan(s2_power)]
            if len(s2_power_valid) > 3:
                x_time = np.arange(len(s2_power_valid))
                slope, _ = np.polyfit(x_time, s2_power_valid, 1)
            else:
                slope = np.nan
                
            s2_torque = torque[s2_start:s2_end]
            s2_speed = speed[s2_start:s2_end]
            s2_speed_nonzero = np.where(s2_speed > 0, s2_speed, np.nan)
            s2_eta_torque = s2_torque / s2_speed_nonzero
            s2_eta_torque_valid = s2_eta_torque[~np.isnan(s2_eta_torque)]
            s2_eta_torque_end = s2_eta_torque_valid[-1] if len(s2_eta_torque_valid) > 0 else np.nan
        else:
            slope = np.nan
            s2_eta_torque_end = np.nan
            
        features.update({
            "Stage1_power_Max": s1_power_max,
            "Stage1_Torque_Max": s1_torque_max,
            "Stage2_power_decay_slope": slope,
            "Stage2_eta_torque_End": s2_eta_torque_end
        })
        
        # L3: Specific energy per stage and thermal history above 100°C
        total_solids = w_solid + w_nr + w_cb + w_silica
        for stage_name in stages.keys():
            power_integral = features.get(f"{stage_name}_power_Integral", 0.0)
            if pd.isna(power_integral):
                power_integral = 0.0
            features[f"{stage_name}_Specific_Energy"] = power_integral / total_solids if total_solids > 0 else 0.0
            
        temp_above_100 = np.maximum(0.0, temp - 100.0)
        temp_above_100_valid = temp_above_100[~np.isnan(temp_above_100)]
        phys_temp_integral_above_100 = np.trapezoid(temp_above_100_valid, dx=1.0) if len(temp_above_100_valid) > 1 else 0.0
        
        features["phys_temp_integral_above_100"] = phys_temp_integral_above_100
        
        # ======================================================================================
        
        feature_rows.append(features)
        
        if (idx + 1) % 500 == 0:
            print(f"Segmented {idx + 1}/{len(decoded)} batches...")
            
    # Save to wide CSV
    out_df = pd.DataFrame(feature_rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Successfully saved feature wide table of {len(out_df)} rows to: {OUTPUT_CSV}")
    print("Final columns:", list(out_df.columns)[:20], "...")

if __name__ == '__main__':
    run_segmentation()
