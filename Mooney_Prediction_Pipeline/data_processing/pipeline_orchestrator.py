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
def connect_mms_encrypted(database='SFEPLANT'):
    """Connect to SQL Server with encryption enabled"""
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

def connect_datamart_encrypted(datamart="mustangmaster"):
    """Connect to Redshift with SSL requirement"""
    c = creds[datamart]
    return psy.connect(
        host=c['host'],
        port=c['port'],
        database=c['database'],
        user=c['user'],
        password=c['password'],
        sslmode='require'  # Enforce SSL encryption
    )

# 3. Dynamic patch of function_definitions_M1 connections
import function_definitions_M1 as fdef
fdef.connect_mms = connect_mms_encrypted
fdef.connect_datamart = connect_datamart_encrypted

def run_pipeline():
    print("=== STARTING OPTIMIZED PIPELINE V2 ===")
    
    # ----------------------------------------------------
    # Step 1: Query Lab MNY Results from Redshift
    # ----------------------------------------------------
    print("\n[Step 1] Querying lab MNY results...")
    with open('data processing/get master MNY test.sql', 'r', encoding='utf-8') as f:
        mny_sql = f.read()
    
    conn_redshift = connect_datamart_encrypted("mustangmaster")
    mny_df = pd.read_sql(mny_sql, conn_redshift)
    print(f"-> Retrieved {len(mny_df)} lab MNY rows.")
    
    # ----------------------------------------------------
    # Step 2: Extract and clean compound names for SQL Server query optimization
    # ----------------------------------------------------
    print("\n[Step 2] Extracting unique compound codes from MNY tests...")
    compounds = mny_df['compound_name'].dropna().unique()
    clean_comp_list = []
    for c in compounds:
        c_str = str(c).strip()
        # Only process M1 and R1 compounds
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
    print("Sample compounds:", clean_comp_set[:10])
    
    # Format list for SQL IN clause
    compounds_list_str = ", ".join([f"'{c}'" for c in clean_comp_set])
    
    # Map compound names to clean codes in mny_df for OrderID filtering
    mny_df['order_id_clean'] = mny_df['order_id'].astype(str).str.strip()
    mny_df['clean_code'] = mny_df['compound_name'].apply(
        lambda x: (str(x).split('-')[0] + '-' + str(x).split('-')[1].replace('-', '').strip()) if len(str(x).split('-')) >= 2 else str(x)
    )

    # ----------------------------------------------------
    # Step 3: Query Mixing Curves and Recipes (FMF) since 2024 (Highly Optimized Chunked)
    # ----------------------------------------------------
    print("\n[Step 3] Querying mixing curves and recipes since 2024...")
    
    # Collect all unique matching OrderIDs across all clean compounds
    all_matching_orders = sorted(list(set(mny_df[mny_df['clean_code'].isin(clean_comp_set)]['order_id_clean'].dropna().unique().tolist())))
    print(f"-> Found {len(all_matching_orders)} unique labeled OrderIDs across all active compounds.")
    
    chunk_size = 150
    fmf_df_list = []
    print(f"-> Querying mixing curves in chunks of {chunk_size} OrderIDs directly...")
    for idx in range(0, len(all_matching_orders), chunk_size):
        order_chunk = all_matching_orders[idx:idx + chunk_size]
        order_ids_chunk_str = ", ".join([f"'{oid}'" for oid in order_chunk if pd.notnull(oid)])
        
        chunk_sql = f"""
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

            FROM [SFEPLANT_ARCHIVE].[dbo].Orders o2 WITH (NOLOCK)
            JOIN [SFEPLANT_ARCHIVE].[dbo].OrderMaterials om WITH (NOLOCK)
                ON o2.OrderID = om.OrderID
            WHERE o2.OrderID IN ({order_ids_chunk_str})
            GROUP BY om.OrderID
        )

        SELECT 
            a.CompoundDescription,
            a.OrderID,
            a.CompoundName,
            a.Equipment AS MixerLine,
            a.OrderStartTime,

            oil.StepNo AS StepNumber,
            oil.Value AS CurrentValue,
            oil.PrevStepValue,

            bh.BatchNumber,

            d.ParameterValue  AS Top_Fill_Factor,
            d2.ParameterValue AS Bot_Fill_Factor,
            d3.ParameterValue AS Target_Temperature,

            bc.Curve1 AS temp,
            bc.Curve2 AS power,
            bc.Curve5 AS Torque,
            bc.Curve6 AS RotorSpeed,
            bc.Curve7 AS WayofRam,

            CAST(m.sum_solid_elastomer AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 AS weight_pct_solid_elastomer,
            CAST(m.sum_natural_rubber AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 AS weight_pct_natural_rubber,
            CAST(m.sum_silica AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 AS weight_pct_silica,
            CAST(m.sum_oil AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 AS weight_pct_oil,
            CAST(m.sum_silian AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 AS weight_pct_silian,
            CAST(m.sum_carbon_black AS FLOAT) / NULLIF(CAST(bh.BatchWeight AS FLOAT), 0) * 100 AS weight_pct_carbon_black

        FROM [SFEPLANT_ARCHIVE].[dbo].[Orders] a WITH (NOLOCK)
        JOIN [SFEPLANT_ARCHIVE].[dbo].[BatchHeader] bh WITH (NOLOCK)
            ON a.OrderID = bh.OrderID

        OUTER APPLY (
            SELECT TOP 1 
                bd.StepNo, 
                bd.Value, 
                bd_prev.Value AS PrevStepValue
            FROM [SFEPLANT_ARCHIVE].[dbo].BatchData bd WITH (NOLOCK)
            LEFT JOIN [SFEPLANT_ARCHIVE].[dbo].BatchData bd_prev WITH (NOLOCK)
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
                  FROM [SFEPLANT_ARCHIVE].[dbo].[RecipeMaterials] rm WITH (NOLOCK)
                  WHERE rm.RecipeID = a.CompoundDescription
                    AND rm.StepNumber = bd.StepNo
                    AND rm.MaterialCode LIKE '%CS4%'
              )
        ) oil

        OUTER APPLY (
            SELECT TOP 1 Curve1, Curve2, Curve5, Curve6, Curve7
            FROM [SFEPLANT_ARCHIVE].[dbo].[BatchCurve] bc WITH (NOLOCK)
            WHERE bc.OrderID = a.OrderID
              AND bc.BatchNumber = bh.BatchNumber
              AND bc.EquipmentID = bh.EquipmentID
            ORDER BY bc.Timestamp DESC
        ) bc

        LEFT JOIN [SFEPLANT_ARCHIVE].[dbo].[RecipeCBS3Parameters] d WITH (NOLOCK)
            ON a.CompoundDescription = d.RecipeID
           AND d.ParameterID = 'MI.1.RHT.Fill-Factor'

        LEFT JOIN [SFEPLANT_ARCHIVE].[dbo].[RecipeCBS3Parameters] d2 WITH (NOLOCK)
            ON a.CompoundDescription = d2.RecipeID
           AND d2.ParameterID = 'MI.1.RHB.Fill-Factor'

        LEFT JOIN [SFEPLANT_ARCHIVE].[dbo].[RecipeCBS3Parameters] d3 WITH (NOLOCK)
            ON a.CompoundDescription = d3.RecipeID
           AND d3.ParameterID = 'MI.1.RHT.Target-Temperature'

        LEFT JOIN order_mat_totals m
            ON m.OrderID = a.OrderID

        WHERE a.OrderID IN ({order_ids_chunk_str})
          AND a.OrderID NOT LIKE '%H%'
          AND a.OrderStartTime >= '2024-01-01'
        """
        chunk_df = pd.read_sql(chunk_sql, connect_mms_encrypted())
        fmf_df_list.append(chunk_df)
        if (idx // chunk_size) % 5 == 0 or (idx + chunk_size) >= len(all_matching_orders):
            print(f"   Processed chunk {idx // chunk_size + 1}/{(len(all_matching_orders) - 1) // chunk_size + 1}: retrieved {len(chunk_df)} rows.")
            
    fmf_df = pd.concat(fmf_df_list, ignore_index=True)
    print(f"-> Total curve/recipe rows retrieved: {len(fmf_df)}")
    
    # ----------------------------------------------------
    # Step 4: Run Sample-to-Batch Mapping (120% sampling) (Optimized Split Query)
    # ----------------------------------------------------
    print("\n[Step 4] Mapping samples to batch numbers using Optimized Split Query...")
    
    fmf_order_ids = sorted(list(set(fmf_df['OrderID'].dropna().unique().tolist())))
    print(f"-> Found {len(fmf_order_ids)} labeled OrderIDs from Step 3 curves. Querying pallet samples in chunks...")
    
    # Query Pallet Samples for these OrderIDs in chunks to avoid full table scans
    f1_list = []
    chunk_size = 1000
    for idx in range(0, len(fmf_order_ids), chunk_size):
        chunk = fmf_order_ids[idx:idx + chunk_size]
        order_id_str = ", ".join([f"'{oid}'" for oid in chunk if pd.notnull(oid)])
        
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
        
        chunk_df = pd.read_sql(pallet_samples_sql, connect_mms_encrypted())
        f1_list.append(chunk_df)
        if (idx // chunk_size) % 5 == 0 or (idx + chunk_size) >= len(fmf_order_ids):
            print(f"   Processed chunk {idx // chunk_size + 1}/{(len(fmf_order_ids) - 1) // chunk_size + 1}: retrieved {len(chunk_df)} rows.")
            
    f1_df = pd.concat(f1_list, ignore_index=True)
    print(f"-> Retrieved {len(f1_df)} total pallet sample rows matching labeled curves.")

    if len(f1_df) == 0:
        raise ValueError("Failed to retrieve any pallet samples matching labeled curves!")
        
    # Query 2: Fetch BatchHeader counts/weights for active OrderIDs in chunks
    active_order_ids = f1_df['OrderID'].dropna().unique().tolist()
    print(f"-> Found {len(active_order_ids)} unique active OrderIDs. Querying BatchHeaders in chunks...")
    
    f2_list = []
    chunk_size = 1000
    for idx in range(0, len(active_order_ids), chunk_size):
        chunk = active_order_ids[idx:idx + chunk_size]
        order_id_str = ", ".join([f"'{oid}'" for oid in chunk if pd.notnull(oid)])
        
        sql = f"""
        select OrderID, BatchNumber, BatchWeight
        from SFEPLANT.dbo.BatchHeader WITH (NOLOCK)
        where OrderID IN ({order_id_str})
        union all
        select OrderID, BatchNumber, BatchWeight
        from SFEPLANT_ARCHIVE.dbo.BatchHeader WITH (NOLOCK)
        where OrderID IN ({order_id_str})
        """
        f2_chunk = pd.read_sql(sql, connect_mms_encrypted())
        f2_list.append(f2_chunk)
        
    f2_df_raw = pd.concat(f2_list)
    print(f"-> Retrieved {len(f2_df_raw)} BatchHeader rows for active orders.")
    
    # Aggregate in Python
    f2_df = f2_df_raw.groupby('OrderID').agg(
        Nr_of_Total_Batches=('BatchNumber', 'max'),
        total_order_weight=('BatchWeight', 'sum')
    ).reset_index()
    
    # Merge
    sample_batch_df = pd.merge(f1_df, f2_df, on='OrderID', how='left')
    
    # Predict batch number using Vectorized Closest-to-Sample-ID Pallet Selection
    sample_ids = pd.to_numeric(sample_batch_df['SampleID'], errors='coerce').fillna(0).astype(int)
    batch_nums = pd.to_numeric(sample_batch_df['BatchNumber'], errors='coerce').fillna(0).astype(int)
    sample_batch_df['SampleID_num'] = sample_ids
    sample_batch_df['BatchNumber_num'] = batch_nums
    
    # 1. Count unique batches per (OrderID, SampleID)
    unique_counts = sample_batch_df.groupby(['OrderID', 'SampleID'])['BatchNumber_num'].transform('nunique')
    
    # 2. Get the unique batch for groups with only 1 unique batch
    single_batch = sample_batch_df.groupby(['OrderID', 'SampleID'])['BatchNumber_num'].transform('min')
    
    # 3. For groups with multiple batches, compute max batch per pallet
    pallet_maxs = sample_batch_df.groupby(['OrderID', 'SampleID', 'PalletID'])['BatchNumber_num'].transform('max')
    sample_batch_df['pallet_max_batch'] = pallet_maxs
    
    # Calculate distance to sample
    sample_batch_df['dist_to_sample'] = (sample_batch_df['pallet_max_batch'] - sample_batch_df['SampleID_num']).abs()
    
    # Find the minimum distance per (OrderID, SampleID)
    min_dists = sample_batch_df.groupby(['OrderID', 'SampleID'])['dist_to_sample'].transform('min')
    
    # Identify which rows correspond to this minimum distance
    is_best_pallet = (sample_batch_df['dist_to_sample'] == min_dists)
    
    # For rows that match the best pallet, get the pallet_max_batch
    sample_batch_df['best_batch_multi'] = np.where(is_best_pallet, sample_batch_df['pallet_max_batch'], np.nan)
    
    # Propagate the best batch to the entire group
    best_multi = sample_batch_df.groupby(['OrderID', 'SampleID'])['best_batch_multi'].transform('max')
    
    # Assign prediction: if unique_counts == 1, use single_batch; else use best_multi
    pred = np.where(unique_counts == 1, single_batch, best_multi)
    
    # Fill any remaining NaNs with single_batch
    nan_mask = np.isnan(pred)
    pred[nan_mask] = single_batch[nan_mask]
    
    # Cap by total batches in order
    sample_batch_df['batch_number_pred'] = np.minimum(pred, sample_batch_df['Nr_of_Total_Batches'].fillna(9999)).astype(int)
    
    # Cleanup temp columns
    sample_batch_df = sample_batch_df.drop(columns=['SampleID_num', 'BatchNumber_num', 'pallet_max_batch', 'dist_to_sample', 'best_batch_multi'])
    
    sample_batch_df['batch_information_fk_final'] = sample_batch_df['OrderID'].astype(str) + '_' + sample_batch_df['batch_number_pred'].apply(lambda x: str(int(x)).zfill(4)) 
    
    print(f"-> Mapped sample-to-batch mapping with {len(sample_batch_df)} rows.")



    # ----------------------------------------------------
    # Step 5: Query Raw Material Charge IDs (Traceability)
    # ----------------------------------------------------
    print("\n[Step 5] Querying raw material Charge IDs for active OrderIDs...")
    active_orders = fmf_df['OrderID'].unique().tolist()
    # Batch query in chunks of 1000 OrderIDs to avoid SQL expression limit
    charge_df_list = []
    chunk_size = 1000
    for i in range(0, len(active_orders), chunk_size):
        chunk = active_orders[i:i + chunk_size]
        order_id_str = ", ".join([f"'{oid}'" for oid in chunk if pd.notnull(oid)])
        
        with open('data processing/raw_mat_chargeID_match.sql', 'r', encoding='utf-8') as f:
            charge_sql = f.read()
        
        charge_sql = charge_sql.replace("TOP (1000)", "")
        if "WHERE" in charge_sql:
            charge_sql = charge_sql.replace("WHERE", f"WHERE OrderID IN ({order_id_str}) AND")
        else:
            charge_sql += f" WHERE OrderID IN ({order_id_str})"
            
        chunk_charge_df = pd.read_sql(charge_sql, connect_mms_encrypted())
        charge_df_list.append(chunk_charge_df)
        
    charge_df = pd.concat(charge_df_list)
    print(f"-> Mapped raw material Charge IDs: {len(charge_df)} rows.")
    
    # ----------------------------------------------------
    # Step 6: Query Raw Material Supplier QM Quality Reports
    # ----------------------------------------------------
    print("\n[Step 6] Querying supplier raw material QM properties...")
    with open('data processing/get_raw_material_properties.sql', 'r', encoding='utf-8') as f:
        qm_sql = f.read()
        
    qm_sql = qm_sql.replace('"""+plant+"""', "'9200'")
    qm_df = pd.read_sql(qm_sql, conn_redshift)
    print(f"-> Retreived supplier QM database: {len(qm_df)} rows.")
    
    # ----------------------------------------------------
    # Step 7: Processing and Pivoting Datasets by Batch/车次
    # ----------------------------------------------------
    print("\n[Step 7] Processing and pivoting datasets...")
    
    # A. Extract FMF Curves (Curves are at batch level, so keep as single columns)
    fmf_meta = fmf_df[[
        'OrderID', 'BatchNumber', 'CompoundDescription', 'CompoundName', 'MixerLine',
        'Top_Fill_Factor', 'Bot_Fill_Factor', 'Target_Temperature', 'OrderStartTime',
        'weight_pct_solid_elastomer', 'weight_pct_natural_rubber', 'weight_pct_silica',
        'weight_pct_oil', 'weight_pct_silian', 'weight_pct_carbon_black',
        'temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam'
    ]].drop_duplicates(subset=['OrderID', 'BatchNumber'])
    
    # For CurrentValue and PrevStepValue, keep as simple columns (take first matching step)
    val_cols = fmf_df.groupby(['OrderID', 'BatchNumber'])[['CurrentValue', 'PrevStepValue']].first().reset_index()
    
    # Merge metadata with value columns
    batch_curves = pd.merge(fmf_meta, val_cols, on=['OrderID', 'BatchNumber'], how='inner')
    
    # Ensure datatypes match
    batch_curves['OrderID'] = batch_curves['OrderID'].astype(str).str.strip()
    batch_curves['BatchNumber'] = batch_curves['BatchNumber'].astype(int)
    print(f"-> Extracted mixing curves to batch level: {len(batch_curves)} rows (one row per batch).")
    
    # B. Process Lab MNY Test Results with ONE-TO-ONE matching using batch_number_pred
    mny_df['order_id_clean'] = mny_df['order_id'].astype(str).str.strip()
    mny_df['sample_id_int'] = pd.to_numeric(mny_df['sample_id'], errors='coerce')
    
    sample_batch_df['OrderID_clean'] = sample_batch_df['OrderID'].astype(str).str.strip()
    sample_batch_df['SampleID_int'] = pd.to_numeric(sample_batch_df['SampleID'], errors='coerce')
    
    # Clean and keep unique OrderID + batch_number_pred MNY results
    sample_batch_df = sample_batch_df.dropna(subset=['batch_number_pred', 'SampleID_int'])
    sample_batch_df['batch_number_pred'] = sample_batch_df['batch_number_pred'].astype(int)
    sample_batch_df['SampleID_int'] = sample_batch_df['SampleID_int'].astype(int)
    
    mny_df = mny_df.dropna(subset=['sample_id_int'])
    mny_df['sample_id_int'] = mny_df['sample_id_int'].astype(int)
    
    mny_mapped = pd.merge(
        mny_df[['order_id_clean', 'sample_id_int', 'test_result', 'test_target', 'test_result_start_time']], 
        sample_batch_df[['OrderID_clean', 'SampleID_int', 'batch_number_pred']], 
        left_on=['order_id_clean', 'sample_id_int'], 
        right_on=['OrderID_clean', 'SampleID_int'], 
        how='inner'
    ).rename(columns={'OrderID_clean': 'OrderID'})
    
    # Group by OrderID + batch_number_pred (mapped batch) to avoid duplication
    mny_labels = mny_mapped.groupby(['OrderID', 'batch_number_pred']).agg({
        'test_result': 'mean',
        'test_target': 'first',
        'test_result_start_time': 'first'
    }).reset_index().rename(columns={
        'batch_number_pred': 'BatchNumber', 
        'test_result': 'MNY', 
        'test_target': 'MNY target'
    })
    mny_labels['OrderID'] = f_ord = mny_labels['OrderID'].astype(str).str.strip()
    mny_labels['BatchNumber'] = mny_labels['BatchNumber'].astype(int)
    print(f"-> Prepared MNY labels: {len(mny_labels)} unique batches with lab MNY values.")
    
    # C. Process Raw Material QM Properties
    if len(qm_df) > 0:
        qm_df['mean value / s'] = pd.to_numeric(qm_df['mean value / s'], errors='coerce')
        charge_df['ChargeID_clean'] = charge_df['ChargeID'].astype(str).str.strip().str.upper()
        qm_df['ChargeID_clean'] = qm_df['batch'].astype(str).str.strip().str.upper()
        
        # Categorize parameters
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
        
        # Group by ChargeID and cat, take average of mean value / s
        qm_agg = qm_df.groupby(['ChargeID_clean', 'cat'])['mean value / s'].mean().unstack().reset_index()
        
        # Merge with charge_df
        charge_with_qm = pd.merge(charge_df, qm_agg, on='ChargeID_clean', how='inner')
        
        # Map/combine CB surface area STSA and Iodine
        if 'cb_surface_area_stsa' in charge_with_qm.columns and 'cb_surface_area_iodine' in charge_with_qm.columns:
            charge_with_qm['supplier_carbon_black_surface_area_avg'] = charge_with_qm['cb_surface_area_stsa'].fillna(charge_with_qm['cb_surface_area_iodine'])
        elif 'cb_surface_area_iodine' in charge_with_qm.columns:
            charge_with_qm['supplier_carbon_black_surface_area_avg'] = charge_with_qm['cb_surface_area_iodine']
        elif 'cb_surface_area_stsa' in charge_with_qm.columns:
            charge_with_qm['supplier_carbon_black_surface_area_avg'] = charge_with_qm['cb_surface_area_stsa']
        else:
            charge_with_qm['supplier_carbon_black_surface_area_avg'] = np.nan
            
        # Map CB structure OAN
        if 'cb_structure_oan' in charge_with_qm.columns:
            charge_with_qm['supplier_carbon_black_structure_avg'] = charge_with_qm['cb_structure_oan']
        else:
            charge_with_qm['supplier_carbon_black_structure_avg'] = np.nan
            
        # Map Silica moisture and surface area
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
        
        # Ensure all columns exist in charge_with_qm
        for col in features_to_agg:
            if col not in charge_with_qm.columns:
                charge_with_qm[col] = np.nan
                
        # Group by OrderID + BatchNumber and take mean
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
    # Step 8: Final Left Merge & Dataset Construction
    # ----------------------------------------------------
    print("\n[Step 8] Constructing final merged dataset...")
    # LEFT join: keep all active batches from 2024 (>= 100k rows), fill labels if available, keep blank otherwise
    final_dataset = pd.merge(batch_curves, mny_labels, on=['OrderID', 'BatchNumber'], how='left')
    final_dataset = pd.merge(final_dataset, batch_qm_features, on=['OrderID', 'BatchNumber'], how='left')
    
    # Generate unique foreign key column batch_information_fk_final
    final_dataset['batch_information_fk_final'] = (
        final_dataset['OrderID'].astype(str) + '_' + 
        final_dataset['BatchNumber'].apply(lambda x: str(int(x)).zfill(4))
    )
    
    # Save the output
    output_filename = 'stage_statistics_enriched.csv'
    final_dataset.to_csv(output_filename, index=False)
    print(f"\n[Done] Successfully saved pivoted dataset to: {output_filename}")
    print(f"Final shape: {final_dataset.shape}")
    print(f"Batches WITH MNY label: {final_dataset['MNY'].notnull().sum()}")
    print(f"Batches WITHOUT MNY label (blank): {final_dataset['MNY'].isnull().sum()}")
    
if __name__ == '__main__':
    run_pipeline()
