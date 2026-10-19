"""
Mathematical & Data-Driven Discovery Engine
============================================
Discovers mathematical laws and correlations directly from:
  1. Continuous curve time series (BatchCurve): Definite integrals, extrema (max/min), means, variances.
  2. Discrete step data (BatchData): Per-step energy integrals, step duration, step torque, step temp extrema.
  3. Lab physical testing (core.LabCurrent): Mooney Viscosity ML(1+4)@100C.

Strictly mathematical and empirical:
  - No preconceived subjective rules.
  - Formulates explicit mathematical equations with R^2, t-stats, p-values, and F-test.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"


def decode_hex(s: str) -> np.ndarray | None:
    if not isinstance(s, str) or len(s) < 8:
        return None
    try:
        a = np.array([int(s[i:i + 4], 16) for i in range(0, len(s) - len(s) % 4, 4)], dtype=float)
        return np.where(a > 32767, a - 65536, a)
    except Exception:
        return None


def extract_features():
    print("[1/5] Loading cached database records...")
    orders = pd.read_pickle(CACHE / "orders.pkl")
    lab = pd.read_pickle(CACHE / "lab.pkl")
    curves = pd.read_pickle(CACHE / "curves.pkl")
    batchdata = pd.read_pickle(CACHE / "batchdata.pkl")

    curves["OrderID"] = curves["OrderID"].astype(str)
    curves["BatchNumber"] = curves["BatchNumber"].astype(int)
    batchdata["OrderID"] = batchdata["OrderID"].astype(str)
    batchdata["BatchNumber"] = batchdata["BatchNumber"].astype(int)
    batchdata["Value"] = pd.to_numeric(batchdata["Value"], errors="coerce")
    lab["OrderID"] = lab["OrderID"].astype(str)
    lab["SampleBatch"] = lab["SampleBatch"].astype(int)
    lab["Mooney"] = pd.to_numeric(lab["Mooney"], errors="coerce")

    print(f"Loaded {len(curves)} curve records, {len(batchdata)} step records, {len(lab)} lab records.")

    # 1. Process BatchData step features
    print("[2/5] Extracting step-wise mathematical features from BatchData...")
    # Pivot MSB and MST steps
    # We want per-step variables: Energy, Temp, Torque, Time, Speed
    msb_data = batchdata[batchdata.GroupName == "AVR_MSB"]
    mst_data = batchdata[batchdata.GroupName == "AVR_MST"]

    def pivot_steps(df_grp, prefix):
        # Key variables of interest
        var_map = {
            "SCP-1-Step-Time-rel-s": "Time_s",
            "SCP-2-Energy-rel-kWh": "Energy_kWh",
            "SCP-3-Temperature-C-F": "Temp_C",
            "MAX-Batch-Temperature-C-F": "TempMax_C",
            "SCP-5-Torque-Nm-ft/lbs": "Torque_Nm",
            "Rotor-Speed-1/min-Rpm": "Rpm",
            "Motor-Current-A": "Current_A"
        }
        df_sub = df_grp[df_grp.VariablePath.isin(var_map.keys())].copy()
        df_sub["VarName"] = df_sub["VariablePath"].map(var_map)
        df_sub["ColName"] = prefix + "_S" + df_sub["StepNo"].astype(str) + "_" + df_sub["VarName"]
        piv = df_sub.pivot_table(
            index=["OrderID", "BatchNumber"],
            columns="ColName",
            values="Value",
            aggfunc="first"
        ).reset_index()
        return piv

    piv_msb = pivot_steps(msb_data, "MSB")
    piv_mst = pivot_steps(mst_data, "MST")

    # Also extract RHB (overall batch summary from MMS)
    rhb_data = batchdata[batchdata.GroupName == "AVR_RHB"]
    rhb_map = {
        "MAX-Batch-Energy-kWh": "MMS_Total_Energy_kWh",
        "MAX-Batch-spec_-Energy-kWh/kg-kWh/lbs": "MMS_Spec_Energy_kWh_kg",
        "MAX-Batch-Temperature-C-F": "MMS_Max_Temp_C",
        "Mixing-Time-actual-s": "MMS_Mixing_Time_s",
        "Temperature-Batch-Start-C-F": "MMS_Start_Temp_C"
    }
    df_rhb = rhb_data[rhb_data.VariablePath.isin(rhb_map.keys())].copy()
    df_rhb["ColName"] = df_rhb["VariablePath"].map(rhb_map)
    piv_rhb = df_rhb.pivot_table(
        index=["OrderID", "BatchNumber"],
        columns="ColName",
        values="Value",
        aggfunc="first"
    ).reset_index()

    # 2. Extract curve mathematical features
    print("[3/5] Extracting continuous curve mathematical features (integrals, extrema, moments)...")
    curve_records = []
    for idx, row in curves.iterrows():
        oid = row["OrderID"]
        bn = row["BatchNumber"]

        c1 = decode_hex(row.get("Curve1")) # Temp (C)
        c2 = decode_hex(row.get("Curve2")) # Power (kW)
        c3 = decode_hex(row.get("Curve3")) # Energy (kWh)
        c5 = decode_hex(row.get("Curve5")) # Torque (Nm)
        c6 = decode_hex(row.get("Curve6")) # Speed (rpm)

        if c1 is None or c2 is None:
            continue
        lengths = [len(c1), len(c2)]
        if c5 is not None: lengths.append(len(c5))
        n_pts = min(lengths)
        if n_pts < 50:
            continue
        c1 = c1[:n_pts]
        c2 = c2[:n_pts]
        if c5 is not None: c5 = c5[:n_pts]
        if c3 is not None and len(c3) >= n_pts: c3 = c3[:n_pts]
        if c6 is not None and len(c6) >= n_pts: c6 = c6[:n_pts]
        # Find discharge drop index (transition between MST and MSB)
        diff_c1 = np.diff(c1)
        # Drop is marked by negative drop > 20 C
        drop_candidates = np.where(diff_c1 < -20)[0]
        if len(drop_candidates) > 0:
            # Largest drop
            drop_idx = drop_candidates[np.argmin(diff_c1[drop_candidates])] + 1
        else:
            drop_idx = int(n_pts * 0.5)

        # Segment masks
        mst_mask = np.arange(n_pts) < drop_idx
        msb_mask = np.arange(n_pts) >= drop_idx

        # --- Continuous Math Features: Full Cycle ---
        # 1. Definite Integrals
        # Sampling rate is ~ 1 sec / pt. Power integral = sum(P) * dt / 3600 (kWh)
        int_power_full = np.sum(c2) / 3600.0        # Total electrical work
        int_temp_full = np.sum(c1)                  # Cumulative thermal history (degC * s)
        int_torque_full = np.sum(c5) if c5 is not None else np.nan  # Angular impulse (Nm * s)

        # 2. Extrema
        temp_max_full = np.max(c1)
        temp_min_full = np.min(c1)
        power_max_full = np.max(c2)
        power_min_full = np.min(c2)
        torque_max_full = np.max(c5) if c5 is not None else np.nan
        torque_min_full = np.min(c5) if c5 is not None else np.nan

        # 3. Statistical Moments
        temp_mean_full = np.mean(c1)
        temp_std_full = np.std(c1)
        power_mean_full = np.mean(c2)
        power_std_full = np.std(c2)
        torque_mean_full = np.mean(c5) if c5 is not None else np.nan

        # --- Segment MST Continuous Math Features ---
        c1_mst = c1[mst_mask]
        c2_mst = c2[mst_mask]
        c5_mst = c5[mst_mask] if c5 is not None else None

        int_power_mst = np.sum(c2_mst) / 3600.0
        int_temp_mst = np.sum(c1_mst)
        temp_max_mst = np.max(c1_mst) if len(c1_mst) > 0 else np.nan
        temp_min_mst = np.min(c1_mst) if len(c1_mst) > 0 else np.nan
        power_max_mst = np.max(c2_mst) if len(c2_mst) > 0 else np.nan
        power_mean_mst = np.mean(c2_mst) if len(c2_mst) > 0 else np.nan
        torque_max_mst = np.max(c5_mst) if c5_mst is not None and len(c5_mst) > 0 else np.nan
        torque_mean_mst = np.mean(c5_mst) if c5_mst is not None and len(c5_mst) > 0 else np.nan

        # --- Segment MSB Continuous Math Features ---
        c1_msb = c1[msb_mask]
        c2_msb = c2[msb_mask]
        c5_msb = c5[msb_mask] if c5 is not None else None

        int_power_msb = np.sum(c2_msb) / 3600.0
        int_temp_msb = np.sum(c1_msb) if len(c1_msb) > 0 else np.nan
        temp_max_msb = np.max(c1_msb) if len(c1_msb) > 0 else np.nan
        temp_min_msb = np.min(c1_msb) if len(c1_msb) > 0 else np.nan
        power_max_msb = np.max(c2_msb) if len(c2_msb) > 0 else np.nan
        power_min_msb = np.min(c2_msb) if len(c2_msb) > 0 else np.nan
        power_mean_msb = np.mean(c2_msb) if len(c2_msb) > 0 else np.nan
        torque_max_msb = np.max(c5_msb) if c5_msb is not None and len(c5_msb) > 0 else np.nan
        torque_mean_msb = np.mean(c5_msb) if c5_msb is not None and len(c5_msb) > 0 else np.nan

        rec = {
            "OrderID": oid,
            "BatchNumber": bn,
            "Curve_Length_s": n_pts,
            "MST_Length_s": int(drop_idx),
            "MSB_Length_s": int(n_pts - drop_idx),
            # Full Integrals & Extrema
            "Curve_Int_Power_Full_kWh": int_power_full,
            "Curve_Int_Temp_Full_degCs": int_temp_full,
            "Curve_Int_Torque_Full_Nms": int_torque_full,
            "Curve_Temp_Max_Full_C": temp_max_full,
            "Curve_Temp_Min_Full_C": temp_min_full,
            "Curve_Power_Max_Full_kW": power_max_full,
            "Curve_Power_Min_Full_kW": power_min_full,
            "Curve_Torque_Max_Full_Nm": torque_max_full,
            "Curve_Temp_Mean_Full_C": temp_mean_full,
            "Curve_Power_Mean_Full_kW": power_mean_full,
            "Curve_Torque_Mean_Full_Nm": torque_mean_full,
            # MST Integrals & Extrema
            "Curve_Int_Power_MST_kWh": int_power_mst,
            "Curve_Int_Temp_MST_degCs": int_temp_mst,
            "Curve_Temp_Max_MST_C": temp_max_mst,
            "Curve_Temp_Min_MST_C": temp_min_mst,
            "Curve_Power_Max_MST_kW": power_max_mst,
            "Curve_Power_Mean_MST_kW": power_mean_mst,
            "Curve_Torque_Max_MST_Nm": torque_max_mst,
            "Curve_Torque_Mean_MST_Nm": torque_mean_mst,
            # MSB Integrals & Extrema
            "Curve_Int_Power_MSB_kWh": int_power_msb,
            "Curve_Int_Temp_MSB_degCs": int_temp_msb,
            "Curve_Temp_Max_MSB_C": temp_max_msb,
            "Curve_Temp_Min_MSB_C": temp_min_msb,
            "Curve_Power_Max_MSB_kW": power_max_msb,
            "Curve_Power_Min_MSB_kW": power_min_msb,
            "Curve_Power_Mean_MSB_kW": power_mean_msb,
            "Curve_Torque_Max_MSB_Nm": torque_max_msb,
            "Curve_Torque_Mean_MSB_Nm": torque_mean_msb,
        }
        curve_records.append(rec)

    df_curve_features = pd.DataFrame(curve_records)
    print(f"Extracted curve features for {len(df_curve_features)} batches.")

    # 3. Merge everything into master dataset
    print("[4/5] Merging features with Lab Mooney targets...")
    df_all = df_curve_features.merge(piv_msb, on=["OrderID", "BatchNumber"], how="left")
    df_all = df_all.merge(piv_mst, on=["OrderID", "BatchNumber"], how="left")
    df_all = df_all.merge(piv_rhb, on=["OrderID", "BatchNumber"], how="left")

    # Match lab Mooney (using SampleBatch == BatchNumber)
    lab_clean = lab[["OrderID", "SampleBatch", "Mooney"]].dropna().drop_duplicates(subset=["OrderID", "SampleBatch"])
    lab_clean = lab_clean.rename(columns={"SampleBatch": "BatchNumber"})

    df_master = df_all.merge(lab_clean, on=["OrderID", "BatchNumber"], how="left")
    print(f"Master feature table shape: {df_master.shape}")
    print(f"Batches with Lab Mooney: {df_master['Mooney'].notna().sum()}")
    print(f"Batches with Total Energy: {df_master['MMS_Total_Energy_kWh'].notna().sum()}")

    return df_master


if __name__ == "__main__":
    df = extract_features()
    # Save clean dataset
    df.to_pickle(CACHE / "mathematical_dataset.pkl")
    print("Extraction successful, saved to cache/mathematical_dataset.pkl")
