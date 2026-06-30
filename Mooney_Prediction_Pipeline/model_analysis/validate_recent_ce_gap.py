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

import json
import os
import sys
import argparse

import joblib
import numpy as np
import pandas as pd
import psycopg2 as psy

sys.path.insert(0, 'data processing')
from new_compound_inference import predict_new_compound
from predict_latest_sfe_order import connect_mms, extract_features_for_row


FEATURE_CSV = 'scratch/recent_ce_gap_features.csv'
PREDICTION_CSV = 'scratch/recent_ce_gap_prediction.csv'
REPORT_CSV = 'scratch/recent_ce_gap_report.csv'
DEFAULT_TRAINING_FEATURE_CSV = 'stage_statistics_enriched_all_features_weather_v4.csv'


def connect_datamart():
    with open(os.path.join(PARENT_DIR, 'data_processing', 'credentials.json'), 'r', encoding='utf-8') as file:
        creds = json.load(file)
    c = creds['mustangmaster']
    return psy.connect(
        host=c['host'],
        port=c['port'],
        database=c['database'],
        user=c['user'],
        password=c['password'],
        sslmode='require',
    )


def query_recent_ms13(limit=200):
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
      AND test_result_start_time >= '2024-01-01'
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
    with connect_datamart() as conn:
        return pd.read_sql(sql, conn)


def load_training_batch_keys(training_feature_csv):
    if not training_feature_csv or not os.path.exists(training_feature_csv):
        return set()
    usecols = ['OrderID', 'BatchNumber']
    training_df = pd.read_csv(training_feature_csv, usecols=lambda col: col in usecols, low_memory=False)
    if not set(usecols).issubset(training_df.columns):
        return set()
    training_df = training_df.dropna(subset=usecols)
    training_df['training_key'] = (
        training_df['OrderID'].astype(str).str.strip() + '_' +
        pd.to_numeric(training_df['BatchNumber'], errors='coerce').fillna(-1).astype(int).astype(str).str.zfill(4)
    )
    return set(training_df['training_key'])


def query_sample_batch_mapping(order_id, sample_id, database):
    sql_samples = """
    SELECT
        CAST(samples.SampleID AS INT) AS SampleID,
        samples.AtWeight,
        samples.PalletID,
        pallets.EstimatedWeight,
        headers.OrderID,
        headers.BatchNumber,
        headers.BatchWeight
    FROM dbo.PalletSample AS samples
    JOIN dbo.Pallets AS pallets
        ON pallets.PalletID = samples.PalletID
    JOIN dbo.BatchHeader AS headers
        ON samples.PalletID = headers.PalletID
    WHERE CAST(headers.OrderID AS VARCHAR) = ?
      AND CAST(samples.SampleID AS INT) = ?
    """
    sql_headers = """
    SELECT OrderID, BatchNumber, BatchWeight
    FROM dbo.BatchHeader
    WHERE CAST(OrderID AS VARCHAR) = ?
    """
    with connect_mms(database) as conn:
        f1 = pd.read_sql(sql_samples, conn, params=[str(order_id), int(sample_id)])
        f2_raw = pd.read_sql(sql_headers, conn, params=[str(order_id)])
    if len(f1) == 0 or len(f2_raw) == 0:
        return pd.DataFrame()

    f2 = f2_raw.groupby('OrderID').agg(
        Nr_of_Total_Batches=('BatchNumber', 'max'),
        total_order_weight=('BatchWeight', 'sum'),
    ).reset_index()
    df = pd.merge(f1, f2, on='OrderID', how='left')

    unique_batches = df['BatchNumber'].dropna().unique()

    # ── Case 1: only one BatchNumber came back → trust it unconditionally ──────
    if len(unique_batches) == 1:
        df['batch_number_pred'] = int(unique_batches[0])
        return df

    # ── Case 2: multiple BatchNumbers across one or more Pallets ───────────────
    #    Problem: the JOIN can pull in extra pallets whose PalletID happens to
    #    match a different pallet record (e.g. pallet 7753081047 with batch 142
    #    appearing alongside pallet 7753081017 with batches 47/48/49 for sample 47).
    #
    #    Fix: for each PalletID, compute max(BatchNumber).
    #    Then pick the Pallet whose max batch is NUMERICALLY CLOSEST to sample_id.
    #    Rationale: lab sample numbers roughly track production batch numbers —
    #    a spurious far-away pallet (batch 142 vs sample 47) will always lose
    #    to the real pallet (batch 49 vs sample 47, distance = 2 vs 95).
    pallet_max = (
        df.groupby(['OrderID', 'PalletID'])['BatchNumber']
        .max()
        .reset_index()
        .rename(columns={'BatchNumber': 'pallet_max_batch'})
    )
    pallet_max['dist_to_sample'] = abs(pallet_max['pallet_max_batch'] - int(sample_id))
    best_row   = pallet_max.loc[pallet_max['dist_to_sample'].idxmin()]
    best_batch = int(best_row['pallet_max_batch'])

    df['batch_number_pred'] = best_batch
    return df


def query_order_batch_row(order_id, batch_number, database):
    sql = """
    WITH selected_batch AS (
        SELECT
            a.CompoundDescription,
            a.OrderID,
            a.CompoundName,
            a.OrderStartTime,
            a.Equipment AS MixerLine,
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
        WHERE CAST(a.OrderID AS VARCHAR) = ?
          AND bh.BatchNumber = ?
          AND bc.Curve1 IS NOT NULL
          AND bc.Curve2 IS NOT NULL
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
        WHERE CAST(om.OrderID AS VARCHAR) = ?
          AND bh.BatchNumber = ?
        GROUP BY om.OrderID, bh.BatchNumber
    )
    SELECT
        sb.CompoundDescription,
        sb.OrderID,
        sb.CompoundName,
        sb.OrderStartTime,
        sb.MixerLine,
        oil.StepNo AS StepNumber,
        oil.Value AS CurrentValue,
        oil.PrevStepValue,
        sb.BatchNumber,
        d.ParameterValue AS Top_Fill_Factor,
        d2.ParameterValue AS Bot_Fill_Factor,
        d3.ParameterValue AS Target_Temperature,
        sb.temp,
        sb.power,
        sb.Torque,
        sb.RotorSpeed,
        sb.WayofRam,
        m.weight_pct_solid_elastomer,
        m.weight_pct_natural_rubber,
        m.weight_pct_silica,
        m.weight_pct_oil,
        m.weight_pct_silian,
        m.weight_pct_carbon_black
    FROM selected_batch sb
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
        WHERE bd.OrderID = sb.OrderID
          AND bd.BatchNumber = sb.BatchNumber
          AND bd.VariablePath = 'SCP-1-Step-Time-rel-s'
          AND bd.GroupName = 'AVR_MST'
          AND EXISTS (
              SELECT 1
              FROM dbo.RecipeMaterials rm
              WHERE rm.RecipeID = sb.CompoundDescription
                AND rm.StepNumber = bd.StepNo
                AND rm.MaterialCode LIKE '%CS4%'
          )
    ) oil
    LEFT JOIN dbo.RecipeCBS3Parameters d
        ON sb.CompoundDescription = d.RecipeID
       AND d.ParameterID = 'MI.1.RHT.Fill-Factor'
    LEFT JOIN dbo.RecipeCBS3Parameters d2
        ON sb.CompoundDescription = d2.RecipeID
       AND d2.ParameterID = 'MI.1.RHB.Fill-Factor'
    LEFT JOIN dbo.RecipeCBS3Parameters d3
        ON sb.CompoundDescription = d3.RecipeID
       AND d3.ParameterID = 'MI.1.RHT.Target-Temperature'
    LEFT JOIN mat_pivot m
        ON m.OrderID = sb.OrderID
       AND m.BatchNumber = sb.BatchNumber
    """
    with connect_mms(database) as conn:
        return pd.read_sql(sql, conn, params=[str(order_id), int(batch_number), str(order_id), int(batch_number)])


def predict_feature_row(feature_row):
    os.makedirs('scratch', exist_ok=True)
    feature_df = pd.DataFrame([feature_row])
    feature_df.to_csv(FEATURE_CSV, index=False, encoding='utf-8-sig')
    bundle = os.path.join(PARENT_DIR, 'models', 'results_with_oil/mooney_model_bundle.joblib') if float(feature_row['is_oil_loading_present']) == 1.0 else os.path.join(PARENT_DIR, 'models', 'results_without_oil/mooney_model_bundle.joblib')
    if not os.path.exists(bundle):
        raise FileNotFoundError(f'Model bundle not found: {bundle}')
    predict_new_compound(bundle, FEATURE_CSV, PREDICTION_CSV)
    pred = pd.read_csv(PREDICTION_CSV).iloc[0]
    return bundle, pred


def main(max_tests=5, ce_limit=300, training_feature_csv=DEFAULT_TRAINING_FEATURE_CSV):
    os.makedirs('scratch', exist_ok=True)
    training_keys = load_training_batch_keys(training_feature_csv)
    print(f'Loaded {len(training_keys)} trained batch keys from {training_feature_csv}')

    ce_rows = query_recent_ms13(limit=ce_limit)
    if ce_rows.empty:
        raise RuntimeError('No recent Compound Excellence MS1+3 rows found.')

    attempts = []
    reports = []
    for _, ce in ce_rows.iterrows():
        if len(reports) >= max_tests:
            break
        order_id = str(ce['order_id']).strip()
        sample_id = int(float(ce['sample_id']))
        for database in ['SFEPLANT', 'SFEPLANT_ARCHIVE']:
            try:
                mapping = query_sample_batch_mapping(order_id, sample_id, database)
                if mapping.empty:
                    attempts.append((order_id, sample_id, database, 'no_sample_batch_mapping'))
                    continue
                batch_number = int(mapping.iloc[0]['batch_number_pred'])
                batch_key = f'{order_id}_{batch_number:04d}'
                if batch_key in training_keys:
                    attempts.append((order_id, sample_id, database, f'skip_already_in_training::{batch_key}'))
                    continue
                batch_row_df = query_order_batch_row(order_id, batch_number, database)
                if batch_row_df.empty:
                    attempts.append((order_id, sample_id, database, f'no_curve_for_batch_{batch_number}'))
                    continue
                feature_row = extract_features_for_row(batch_row_df.iloc[0])
                bundle, pred = predict_feature_row(feature_row)
                actual = float(ce['test_result'])
                predicted = float(pred['predicted_MNY_base'])
                gap = predicted - actual
                report = {
                    'order_id': order_id,
                    'sample_id': sample_id,
                    'mapped_batch_number': batch_number,
                    'batch_key': batch_key,
                    'database': database,
                    'compound_name_ce': ce['compound_name'],
                    'compound_name_sfe': batch_row_df.iloc[0]['CompoundName'],
                    'test_result_start_time': ce['test_result_start_time'],
                    'lab_MS1_3': actual,
                    'predicted_MNY': predicted,
                    'gap_pred_minus_lab': gap,
                    'abs_gap': abs(gap),
                    'test_target': ce['test_target'],
                    'uom': ce['uom'],
                    'model_bundle': bundle,
                    'reliability': pred['applicability_reliability'],
                    'applicability_distance': pred['applicability_distance'],
                    'out_of_reference_range_feature_count': pred['out_of_reference_range_feature_count'],
                    'feature_csv': FEATURE_CSV,
                    'prediction_csv': PREDICTION_CSV,
                }
                reports.append(report)
                print(f'Validated unseen CE MS1+3 row {len(reports)}/{max_tests}:')
                for key, value in report.items():
                    print(f'  {key}: {value}')
                print('')
                break
            except Exception as exc:
                attempts.append((order_id, sample_id, database, str(exc)[:200]))

    if reports:
        report_df = pd.DataFrame(reports)
        report_df.to_csv(REPORT_CSV, index=False, encoding='utf-8-sig')
        print('\nBatch CE MS1+3 validation completed:')
        print(report_df[['order_id', 'sample_id', 'mapped_batch_number', 'compound_name_ce', 'lab_MS1_3', 'predicted_MNY', 'gap_pred_minus_lab', 'abs_gap', 'reliability']].to_string(index=False))
        print(f'\nSaved report to {REPORT_CSV}')
        return

    attempts_df = pd.DataFrame(attempts, columns=['order_id', 'sample_id', 'database', 'reason'])
    attempts_df.to_csv('scratch/recent_ce_gap_failed_attempts.csv', index=False, encoding='utf-8-sig')
    raise RuntimeError('Could not validate any recent CE MS1+3 row. See scratch/recent_ce_gap_failed_attempts.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate recent Compound Excellence MS1+3 rows that were not in the training feature CSV.')
    parser.add_argument('--max-tests', type=int, default=5, help='Number of unseen mapped batches to validate')
    parser.add_argument('--ce-limit', type=int, default=300, help='Number of recent CE MS1+3 rows to scan')
    parser.add_argument('--training-feature-csv', default=DEFAULT_TRAINING_FEATURE_CSV, help='Training feature CSV used to exclude already-trained batches')
    args = parser.parse_args()
    main(max_tests=args.max_tests, ce_limit=args.ce_limit, training_feature_csv=args.training_feature_csv)