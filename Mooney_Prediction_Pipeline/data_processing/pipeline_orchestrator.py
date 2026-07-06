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
import json
import warnings
import time
import pyodbc
import psycopg2 as psy
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# 1. Load credentials
CREDENTIALS_PATH = os.path.join(PARENT_DIR, 'data_processing', 'credentials.json')
with open(CREDENTIALS_PATH, 'r', encoding='utf-8') as f:
    creds = json.load(f)

# 2. Database connection functions with SSL encryption enabled
def connect_mms_encrypted(database='SFEPLANT', timeout=120):
    """Connect to SQL Server with encryption and timeout enabled"""
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
    return pyodbc.connect(conn_str, timeout=timeout)

def query_with_retry(sql, max_retries=5, base_delay=20, database=None):
    """Execute SQL query with retry logic, exponential backoff, and database fallback.
    
    Tries the specified database first (default: SFEPLANT_ARCHIVE), then falls back to
    the other database if the first is unavailable.
    """
    databases_to_try = [database or 'SFEPLANT_ARCHIVE', 'SFEPLANT']
    databases_to_try = list(dict.fromkeys(databases_to_try))
    
    last_error = None
    for attempt in range(1, max_retries + 1):
        for db in databases_to_try:
            try:
                conn = connect_mms_encrypted(database=db, timeout=180)
                df = pd.read_sql(sql, conn)
                conn.close()
                return df
            except Exception as e:
                last_error = e
                err_str = str(e)
                if 'Cannot open database' in err_str or 'Login failed' in err_str:
                    continue
                break
        
        if attempt == max_retries:
            print(f"      [FATAL] Failed after {max_retries} attempts: {last_error}")
            raise last_error
        delay = min(base_delay * (2 ** (attempt - 1)), 120)
        print(f"      [RETRY {attempt}/{max_retries}] Error: {str(last_error)[:100]}... Retrying in {delay}s")
        time.sleep(delay)

def connect_datamart_encrypted(datamart="mustangmaster"):
    """Connect to Redshift with SSL requirement"""
    c = creds[datamart]
    return psy.connect(
        host=c['host'],
        port=c['port'],
        database=c['database'],
        user=c['user'],
        password=c['password'],
        sslmode='require'
    )

# 3. Dynamic patch of function_definitions_M1 connections
import function_definitions_M1 as fdef
fdef.connect_mms = connect_mms_encrypted
fdef.connect_datamart = connect_datamart_encrypted

def run_pipeline():
    print("=== STARTING OPTIMIZED PIPELINE V2 (ACTUAL PALLET SCOPE) ===")
    
    # ----------------------------------------------------
    # Step 1: Query Lab MNY Results from Redshift
    # ----------------------------------------------------
    print("\n[Step 1] Querying lab MNY results...")
    with open(os.path.join(SCRIPT_DIR, 'get master MNY test.sql'), 'r', encoding='utf-8') as f:
        mny_sql = f.read()
    
    conn_redshift = connect_datamart_encrypted("mustangmaster")
    mny_df = pd.read_sql(mny_sql, conn_redshift)
    print(f"-> Retrieved {len(mny_df)} lab MNY rows.")
    
    # ----------------------------------------------------
    # Step 2: Extract and clean compound codes from MNY tests
    # ----------------------------------------------------
    print("\n[Step 2] Extracting unique compound codes from MNY tests...")
    compounds = mny_df['compound_name'].dropna().unique()
    clean_comp_list = []
    for c in compounds:
        c_str = str(c).strip()
        if not (c_str.startswith('M1-') or c_str.startswith('R1-')):
            continue
        parts = c_str.split('-')
        if len(parts) >= 2:
            clean_code = parts[0] + '-' + parts[1].replace('-', '').strip()
            clean_comp_list.append(clean_code)
        else:
            clean_comp_list.append(c_str)
    clean_comp_set = sorted(list(set(clean_comp_list)))
    print(f"-> Extracted {len(clean_comp_set)} unique clean compound codes.")
    
    # Map compound names to clean codes in mny_df for OrderID filtering
    mny_df['order_id_clean'] = mny_df['order_id'].astype(str).str.strip()
    mny_df['clean_code'] = mny_df['compound_name'].apply(
        lambda x: (str(x).split('-')[0] + '-' + str(x).split('-')[1].replace('-', '').strip()) if len(str(x).split('-')) >= 2 else str(x)
    )

    all_matching_orders = sorted(list(set(mny_df[mny_df['clean_code'].isin(clean_comp_set)]['order_id_clean'].dropna().unique().tolist())))
    print(f"-> Found {len(all_matching_orders)} unique labeled OrderIDs across all active compounds.")

    # ----------------------------------------------------
    # Step 3: Run Sample-to-Batch Mapping (Metadata query)
    # ----------------------------------------------------
    print("\n[Step 3] Querying Pallet & Sample metadata from SQL Server...")
    fmf_order_ids = all_matching_orders
    
    f1_list = []
    chunk_size = 500
    for idx in range(0, len(fmf_order_ids), chunk_size):
        chunk = fmf_order_ids[idx:idx + chunk_size]
        expanded_chunk = []
        for oid in chunk:
            if pd.notnull(oid):
                oid_str = str(oid).strip()
                expanded_chunk.append(oid_str)
                expanded_chunk.append(f"2023~{oid_str}")
                expanded_chunk.append(f"2024~{oid_str}")
                expanded_chunk.append(f"2025~{oid_str}")
                expanded_chunk.append(f"2026~{oid_str}")
                
        order_id_str = ", ".join([f"'{oid}'" for oid in expanded_chunk])
        
        pallet_samples_sql = f"""
        select SampleID, AtWeight, samples.PalletID, pallets.EstimatedWeight, headers.OrderID, headers.BatchNumber, headers.BatchWeight
        from SFEPLANT.dbo.PalletSample as samples WITH (NOLOCK)
        join SFEPLANT.dbo.Pallets as pallets WITH (NOLOCK) on pallets.PalletID=samples.PalletID 
        join SFEPLANT.dbo.BatchHeader as headers WITH (NOLOCK) on (samples.PalletID = headers.PalletID)
        where pallets.OrderID IN ({order_id_str})
        
        union all
        
        select SampleID, AtWeight, samples.PalletID, pallets.EstimatedWeight, headers.OrderID, headers.BatchNumber, headers.BatchWeight
        from SFEPLANT_ARCHIVE.dbo.PalletSample as samples WITH (NOLOCK)
        join SFEPLANT_ARCHIVE.dbo.Pallets as pallets WITH (NOLOCK) on pallets.PalletID=samples.PalletID 
        join SFEPLANT_ARCHIVE.dbo.BatchHeader as headers WITH (NOLOCK) on (samples.PalletID = headers.PalletID)
        where pallets.OrderID IN ({order_id_str})
        """
        chunk_df = query_with_retry(pallet_samples_sql, max_retries=3, base_delay=10)
        f1_list.append(chunk_df)
        
    f1_df = pd.concat(f1_list, ignore_index=True)
    print(f"-> Retrieved {len(f1_df)} total pallet/batch mapping rows.")

    if len(f1_df) == 0:
        raise ValueError("Failed to retrieve any pallet samples matching labeled curves!")

    # ----------------------------------------------------
    # Step 4: Map and filter exact labeled (OrderID, PalletID, BatchNumber) groups
    # ----------------------------------------------------
    print("\n[Step 4] Mapping Mooney test results to actual pallet batches...")
    mny_df['order_id_clean'] = mny_df['order_id'].astype(str).str.strip()
    mny_df['sample_id_int'] = pd.to_numeric(mny_df['sample_id'], errors='coerce')
    
    f1_df['OrderID_clean'] = f1_df['OrderID'].astype(str).str.strip().apply(
        lambda x: x.split('~')[1] if '~' in x else x
    )
    f1_df['SampleID_int'] = pd.to_numeric(f1_df['SampleID'], errors='coerce')
    
    # Filter valid rows
    f1_df = f1_df.dropna(subset=['SampleID_int', 'BatchNumber'])
    f1_df['BatchNumber'] = f1_df['BatchNumber'].astype(int)
    f1_df['SampleID_int'] = f1_df['SampleID_int'].astype(int)
    
    mny_df = mny_df.dropna(subset=['sample_id_int'])
    mny_df['sample_id_int'] = mny_df['sample_id_int'].astype(int)
    
    # Inner join: matches lab test to ALL batches in the corresponding PalletID
    mny_mapped = pd.merge(
        mny_df[['order_id_clean', 'sample_id_int', 'test_result', 'test_target', 'test_result_start_time']], 
        f1_df[['OrderID', 'OrderID_clean', 'SampleID_int', 'PalletID', 'BatchNumber']], 
        left_on=['order_id_clean', 'sample_id_int'], 
        right_on=['OrderID_clean', 'SampleID_int'], 
        how='inner'
    )
    
    # Group by (OrderID, PalletID, BatchNumber) to aggregate retests cleanly
    mny_labels = mny_mapped.groupby(['OrderID', 'PalletID', 'BatchNumber']).agg({
        'test_result': 'mean',
        'test_target': 'first',
        'test_result_start_time': 'first'
    }).reset_index().rename(columns={
        'test_result': 'MNY', 
        'test_target': 'MNY target'
    })
    mny_labels['OrderID'] = mny_labels['OrderID'].astype(str).str.strip()
    mny_labels['PalletID'] = mny_labels['PalletID'].astype(str).str.strip()
    mny_labels['BatchNumber'] = mny_labels['BatchNumber'].astype(int)
    
    # Unique query pairs (OrderID, BatchNumber) to pull curves for
    query_pairs = mny_labels[['OrderID', 'BatchNumber']].drop_duplicates()
    print(f"-> Total unique group batches to query: {len(query_pairs)} (across {mny_labels['PalletID'].nunique()} pallets).")

    # ----------------------------------------------------
    # Step 5: Query Mixing Curves and Recipes (FMF) for group batches
    # ----------------------------------------------------
    print("\n[Step 5] Querying mixing curves and recipes for group batches since 2024...")
    
    chunk_size = 200
    fmf_df_list = []
    total_chunks = (len(query_pairs) - 1) // chunk_size + 1
    total_rows = 0
    print(f"-> Querying mixing curves in chunks of {chunk_size} group batches...")
    
    def build_curve_query_exact(order_ids_str, where_clause, db_prefix):
        return f"""
        WITH order_mat_totals AS (
            SELECT
                om.OrderID,
                SUM(CASE WHEN om.MaterialCode LIKE 'CE%' 
                          AND om.MaterialCode NOT LIKE 'CE19%'
                     THEN CAST(om.BatchWeight AS FLOAT) 
                END) AS sum_solid_elastomer,
                SUM(CASE WHEN om.MaterialCode LIKE 'CN%'
                     THEN CAST(om.BatchWeight AS FLOAT) 
                END) AS sum_natural_rubber,
                SUM(CASE WHEN om.MaterialCode LIKE 'CS100%'
                     THEN CAST(om.BatchWeight AS FLOAT) 
                END) AS sum_silica,
                SUM(CASE WHEN om.MaterialCode LIKE 'CS%' 
                          AND om.MaterialCode NOT LIKE 'CS100%'
                     THEN CAST(om.BatchWeight AS FLOAT) 
                END) AS sum_oil,
                SUM(CASE WHEN om.MaterialCode LIKE 'CA551%'
                     THEN CAST(om.BatchWeight AS FLOAT) 
                END) AS sum_silian,
                SUM(CASE WHEN om.MaterialName LIKE 'CTP%'
                     THEN CAST(om.BatchWeight AS FLOAT) 
                END) AS sum_CTP,
                SUM(CASE WHEN om.MaterialCode LIKE 'CC%'
                     THEN CAST(om.BatchWeight AS FLOAT) 
                END) AS sum_carbon_black
            FROM [{db_prefix}].[dbo].Orders o2 WITH (NOLOCK)
            JOIN [{db_prefix}].[dbo].OrderMaterials om WITH (NOLOCK)
                ON o2.OrderID = om.OrderID
            WHERE o2.OrderID IN ({order_ids_str})
            GROUP BY om.OrderID
        )
        SELECT 
            a.CompoundDescription, a.OrderID, a.CompoundName,
            a.Equipment AS MixerLine, a.OrderStartTime,
            oil.StepNo AS StepNumber, oil.Value AS CurrentValue, oil.PrevStepValue,
            bh.BatchNumber,
            d.ParameterValue  AS Top_Fill_Factor,
            d2.ParameterValue AS Bot_Fill_Factor,
            d3.ParameterValue AS Target_Temperature,
            bc.Curve1 AS temp, bc.Curve2 AS power,
            bc.Curve5 AS Torque, bc.Curve6 AS RotorSpeed, bc.Curve7 AS WayofRam,
            CAST(m.sum_solid_elastomer AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 AS weight_pct_solid_elastomer,
            CAST(m.sum_natural_rubber AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 AS weight_pct_natural_rubber,
            CAST(m.sum_silica AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 AS weight_pct_silica,
            CAST(m.sum_oil AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 AS weight_pct_oil,
            CAST(m.sum_silian AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 AS weight_pct_silian,
            CAST(m.sum_carbon_black AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 AS weight_pct_carbon_black
        FROM [{db_prefix}].[dbo].[Orders] a WITH (NOLOCK)
        JOIN [{db_prefix}].[dbo].[BatchHeader] bh WITH (NOLOCK)
            ON a.OrderID = bh.OrderID
        OUTER APPLY (
            SELECT TOP 1 bd.StepNo, bd.Value, bd_prev.Value AS PrevStepValue
            FROM [{db_prefix}].[dbo].BatchData bd WITH (NOLOCK)
            LEFT JOIN [{db_prefix}].[dbo].BatchData bd_prev WITH (NOLOCK)
                ON bd_prev.OrderID = bd.OrderID
               AND bd_prev.BatchNumber = bd.BatchNumber
               AND bd_prev.EquipmentID = bd.EquipmentID
               AND bd_prev.StepNo = bd.StepNo - 1
               AND bd_prev.VariablePath = 'SCP-1-Step-Time-rel-s'
               AND bd_prev.GroupName = 'AVR_MST'
            WHERE bd.OrderID = a.OrderID
              AND bd.BatchNumber = bh.BatchNumber
              AND bd.EquipmentID = bh.EquipmentID
              AND bd.VariablePath = 'SCP-1-Step-Time-rel-s'
              AND bd.GroupName = 'AVR_MST'
              AND EXISTS (
                  SELECT 1 
                  FROM [{db_prefix}].[dbo].[RecipeMaterials] rm WITH (NOLOCK)
                  WHERE rm.RecipeID = a.CompoundDescription
                    AND rm.StepNumber = bd.StepNo
                    AND rm.MaterialCode LIKE '%CS4%'
              )
        ) oil
        OUTER APPLY (
            SELECT TOP 1 Curve1, Curve2, Curve5, Curve6, Curve7
            FROM [{db_prefix}].[dbo].[BatchCurve] bc WITH (NOLOCK)
            WHERE bc.OrderID = a.OrderID
              AND bc.BatchNumber = bh.BatchNumber
              AND bc.EquipmentID = bh.EquipmentID
            ORDER BY bc.Timestamp DESC
        ) bc
        LEFT JOIN [{db_prefix}].[dbo].[RecipeCBS3Parameters] d WITH (NOLOCK)
            ON a.CompoundDescription = d.RecipeID
           AND d.ParameterID = 'MI.1.RHT.Fill-Factor'
        LEFT JOIN [{db_prefix}].[dbo].[RecipeCBS3Parameters] d2 WITH (NOLOCK)
            ON a.CompoundDescription = d2.RecipeID
           AND d2.ParameterID = 'MI.1.RHB.Fill-Factor'
        LEFT JOIN [{db_prefix}].[dbo].[RecipeCBS3Parameters] d3 WITH (NOLOCK)
            ON a.CompoundDescription = d3.RecipeID
           AND d3.ParameterID = 'MI.1.RHT.Target-Temperature'
        LEFT JOIN order_mat_totals m
            ON m.OrderID = a.OrderID
        WHERE ({where_clause})
          AND a.OrderID NOT LIKE '%H%'
          AND a.OrderStartTime >= '2024-01-01'
        """

    for idx in range(0, len(query_pairs), chunk_size):
        chunk_df = query_pairs.iloc[idx:idx + chunk_size]
        unique_oids = chunk_df['OrderID'].dropna().unique().tolist()
        order_ids_str = ", ".join([f"'{oid}'" for oid in unique_oids])
        
        pair_conds = [f"(a.OrderID = '{row['OrderID']}' AND bh.BatchNumber = {row['BatchNumber']})" for _, row in chunk_df.iterrows()]
        where_clause = " OR ".join(pair_conds)
        
        chunk_num = idx // chunk_size + 1
        
        # Query SFEPLANT_ARCHIVE with retry
        archive_sql = build_curve_query_exact(order_ids_str, where_clause, 'SFEPLANT_ARCHIVE')
        chunk_res = query_with_retry(archive_sql, max_retries=3, base_delay=15)
        
        # Query SFEPLANT with retry
        active_sql = build_curve_query_exact(order_ids_str, where_clause, 'SFEPLANT')
        try:
            active_res = query_with_retry(active_sql, max_retries=2, base_delay=10)
            if len(active_res) > 0:
                chunk_res = pd.concat([chunk_res, active_res], ignore_index=True)
                chunk_res = chunk_res.drop_duplicates(subset=['OrderID', 'BatchNumber'], keep='first')
        except Exception:
            pass
            
        fmf_df_list.append(chunk_res)
        total_rows += len(chunk_res)
        print(f"   Chunk {chunk_num}/{total_chunks}: +{len(chunk_res)} rows (cumulative: {total_rows})")
        
    fmf_df = pd.concat(fmf_df_list, ignore_index=True)
    print(f"-> Total curves retrieved for group batches: {len(fmf_df)}")

    # ----------------------------------------------------
    # Step 6: Query Raw Material Charge IDs (Traceability)
    # ----------------------------------------------------
    print("\n[Step 6] Querying raw material Charge IDs for group OrderIDs...")
    active_orders = fmf_df['OrderID'].unique().tolist()
    charge_df_list = []
    chunk_size = 1000
    for i in range(0, len(active_orders), chunk_size):
        chunk = active_orders[i:i + chunk_size]
        order_id_str = ", ".join([f"'{oid}'" for oid in chunk if pd.notnull(oid)])
        
        with open(os.path.join(SCRIPT_DIR, 'raw_mat_chargeID_match.sql'), 'r', encoding='utf-8') as f:
            charge_sql = f.read()
        
        charge_sql = charge_sql.replace("TOP (1000)", "")
        if "WHERE" in charge_sql:
            charge_sql = charge_sql.replace("WHERE", f"WHERE OrderID IN ({order_id_str}) AND")
        else:
            charge_sql += f" WHERE OrderID IN ({order_id_str})"
            
        chunk_charge_df = query_with_retry(charge_sql, max_retries=3, base_delay=10)
        charge_df_list.append(chunk_charge_df)
        
    charge_df = pd.concat(charge_df_list)
    charge_df['OrderID'] = charge_df['OrderID'].astype(str).str.strip()
    charge_df['BatchNumber'] = charge_df['BatchNumber'].astype(int)
    charge_df = charge_df.merge(query_pairs, on=['OrderID', 'BatchNumber'], how='inner')
    print(f"-> Mapped raw material Charge IDs for group batches: {len(charge_df)} rows.")
    
    # ----------------------------------------------------
    # Step 7: Query Raw Material Supplier QM Quality Reports
    # ----------------------------------------------------
    print("\n[Step 7] Querying supplier raw material QM properties...")
    with open(os.path.join(SCRIPT_DIR, 'get_raw_material_properties.sql'), 'r', encoding='utf-8') as f:
        qm_sql = f.read()
        
    qm_sql = qm_sql.replace('"""+plant+"""', "'9200'")
    qm_df = pd.read_sql(qm_sql, conn_redshift)
    print(f"-> Retrieved supplier QM database: {len(qm_df)} rows.")
    
    # ----------------------------------------------------
    # Step 8: Processing and Pivoting Datasets by Batch/车次
    # ----------------------------------------------------
    print("\n[Step 8] Processing and pivoting datasets...")
    
    # A. Extract FMF Curves
    fmf_meta = fmf_df[[
        'OrderID', 'BatchNumber', 'CompoundDescription', 'CompoundName', 'MixerLine',
        'Top_Fill_Factor', 'Bot_Fill_Factor', 'Target_Temperature', 'OrderStartTime',
        'weight_pct_solid_elastomer', 'weight_pct_natural_rubber', 'weight_pct_silica',
        'weight_pct_oil', 'weight_pct_silian', 'weight_pct_carbon_black',
        'temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam'
    ]].drop_duplicates(subset=['OrderID', 'BatchNumber'])
    
    val_cols = fmf_df.groupby(['OrderID', 'BatchNumber'])[['CurrentValue', 'PrevStepValue']].first().reset_index()
    
    batch_curves = pd.merge(fmf_meta, val_cols, on=['OrderID', 'BatchNumber'], how='inner')
    batch_curves['OrderID'] = batch_curves['OrderID'].astype(str).str.strip()
    batch_curves['BatchNumber'] = batch_curves['BatchNumber'].astype(int)
    print(f"-> Extracted mixing curves to batch level: {len(batch_curves)} rows.")
    
    # B. Process Raw Material QM Properties
    if len(qm_df) > 0 and len(charge_df) > 0:
        qm_df['mean value / s'] = pd.to_numeric(qm_df['mean value / s'], errors='coerce')
        charge_df['ChargeID_clean'] = charge_df['ChargeID'].astype(str).str.strip().str.upper()
        qm_df['ChargeID_clean'] = qm_df['batch'].astype(str).str.strip().str.upper()
        qm_df['parameter_clean'] = qm_df['short text'].astype(str).str.strip().str.lower()
        
        def map_parameter_category(row):
            param = row['parameter_clean']
            mat = str(row['material'])
            if 'loss on heating' in param and mat.startswith('CS100'):
                return 'silica_moisture'
            elif 'nitrogen surface area' in param and mat.startswith('CS100'):
                return 'silica_surface_area_bet'
            elif 'dbp' in param and mat.startswith('CC'):
                return 'cb_structure_oan'
            elif 'iodine' in param and mat.startswith('CC'):
                return 'cb_surface_area_iodine'
            elif 'stsa' in param and mat.startswith('CC'):
                return 'cb_surface_area_stsa'
            elif 'heating' in param and mat.startswith('CC'):
                return 'supplier_carbon_black_moisture_avg'
            elif 'uml 1+4' in param and (mat.startswith('CE') or mat.startswith('CN')):
                return 'supplier_rubber_viscosity_avg'
            return 'other'
            
        qm_df['cat'] = qm_df.apply(map_parameter_category, axis=1)
        qm_df = qm_df[qm_df['cat'] != 'other']
        
        qm_agg = qm_df.groupby(['ChargeID_clean', 'cat'])['mean value / s'].mean().unstack().reset_index()
        charge_with_qm = pd.merge(charge_df, qm_agg, on='ChargeID_clean', how='inner')
        
        if 'cb_surface_area_stsa' in charge_with_qm.columns and 'cb_surface_area_iodine' in charge_with_qm.columns:
            charge_with_qm['supplier_carbon_black_surface_area_avg'] = charge_with_qm['cb_surface_area_stsa'].fillna(charge_with_qm['cb_surface_area_iodine'])
        elif 'cb_surface_area_iodine' in charge_with_qm.columns:
            charge_with_qm['supplier_carbon_black_surface_area_avg'] = charge_with_qm['cb_surface_area_iodine']
        elif 'cb_surface_area_stsa' in charge_with_qm.columns:
            charge_with_qm['supplier_carbon_black_surface_area_avg'] = charge_with_qm['cb_surface_area_stsa']
        else:
            charge_with_qm['supplier_carbon_black_surface_area_avg'] = np.nan
            
        if 'cb_structure_oan' in charge_with_qm.columns:
            charge_with_qm['supplier_carbon_black_structure_avg'] = charge_with_qm['cb_structure_oan']
        else:
            charge_with_qm['supplier_carbon_black_structure_avg'] = np.nan
            
        if 'silica_moisture' in charge_with_qm.columns:
            charge_with_qm['supplier_silica_moisture_avg'] = charge_with_qm['silica_moisture']
        else:
            charge_with_qm['supplier_silica_moisture_avg'] = np.nan
            
        if 'silica_surface_area_bet' in charge_with_qm.columns:
            charge_with_qm['supplier_silica_surface_area_avg'] = charge_with_qm['silica_surface_area_bet']
        else:
            charge_with_qm['supplier_silica_surface_area_avg'] = np.nan
            
        features_to_agg = [
            'supplier_rubber_viscosity_avg',
            'supplier_silica_moisture_avg',
            'supplier_silica_surface_area_avg',
            'supplier_carbon_black_structure_avg',
            'supplier_carbon_black_surface_area_avg',
            'supplier_carbon_black_moisture_avg'
        ]
        
        for col in features_to_agg:
            if col not in charge_with_qm.columns:
                charge_with_qm[col] = np.nan
                
        batch_qm_features = charge_with_qm.groupby(['OrderID', 'BatchNumber'])[features_to_agg].mean().reset_index()
        batch_qm_features['OrderID'] = batch_qm_features['OrderID'].astype(str).str.strip()
        batch_qm_features['BatchNumber'] = batch_qm_features['BatchNumber'].astype(int)
        print(f"-> Prepared supplier QM properties for {len(batch_qm_features)} batches.")
    else:
        features_to_agg = [
            'supplier_rubber_viscosity_avg',
            'supplier_silica_moisture_avg',
            'supplier_silica_surface_area_avg',
            'supplier_carbon_black_structure_avg',
            'supplier_carbon_black_surface_area_avg',
            'supplier_carbon_black_moisture_avg'
        ]
        batch_qm_features = pd.DataFrame(columns=['OrderID', 'BatchNumber'] + features_to_agg)

    # ----------------------------------------------------
    # Step 9: Final Inner Merge & Pallet-Level Mapping
    # ----------------------------------------------------
    print("\n[Step 9] Constructing final merged dataset at group scope...")
    
    # Pre-merge batch curves and QM features
    batch_data = pd.merge(batch_curves, batch_qm_features, on=['OrderID', 'BatchNumber'], how='inner')
    
    # Merge with MNY labels (keep all batches in the query scope, linked to their physical PalletID)
    final_dataset = pd.merge(batch_data, mny_labels, on=['OrderID', 'BatchNumber'], how='inner')
    
    final_dataset['batch_information_fk_final'] = (
        final_dataset['OrderID'].astype(str) + '_' + 
        final_dataset['BatchNumber'].apply(lambda x: str(int(x)).zfill(4))
    )
    
    output_filename = 'stage_statistics_enriched.csv'
    final_dataset.to_csv(output_filename, index=False)
    print(f"\n[Done] Successfully saved pivoted dataset to: {output_filename}")
    print(f"Final shape: {final_dataset.shape}")
    print(f"Batches WITH MNY label in group: {final_dataset['MNY'].notnull().sum()}")
    
if __name__ == '__main__':
    run_pipeline()
