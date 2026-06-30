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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest, StackingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
import joblib

# Configure plotting style
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # Support Chinese characters
plt.rcParams['axes.unicode_minus'] = False            # Support negative signs
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def safe_to_csv(df, path, **kwargs):
    import time
    for i in range(5):
        try:
            df.to_csv(path, **kwargs)
            return
        except PermissionError:
            print(f"\n[WARNING] Permission denied when writing to: {path}")
            print("The file might be open in Excel. Please CLOSE the file now!")
            print("Retrying in 2 seconds...")
            time.sleep(2)
    # Fallback to a file with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dir_name, file_name = os.path.split(path)
    base, ext = os.path.splitext(file_name)
    fallback_path = os.path.join(dir_name, f"{base}_{timestamp}{ext}")
    print(f"[ERROR] Could not write to {path} after retries. Saving fallback to: {fallback_path}")
    df.to_csv(fallback_path, **kwargs)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths
INPUT_CSV = os.path.join(WORKSPACE_ROOT, "stage_statistics_enriched_all_features_weather_v4.csv")
BRAIN_DIR = r"C:\Users\uif35346\.gemini\antigravity\brain\b5dcf5d4-477f-4b95-9fa0-2f369ea05fdb"

MAX_INTERPRETABLE_FEATURES = 80
MAX_MISSING_RATE = 0.70
MAX_DOMINANT_VALUE_RATE = 0.98
MAX_FEATURE_CORRELATION = 0.90

PROCESS_FEATURE_PRIORITY = {
    'supplier_rubber_viscosity_avg': 5.0,
    'supplier_silica_moisture_avg': 5.0,
    'supplier_silica_surface_area_avg': 5.0,
    'supplier_carbon_black_structure_avg': 5.0,
    'supplier_carbon_black_surface_area_avg': 5.0,
    'supplier_carbon_black_moisture_avg': 5.0,
    'roll_mny_': 4.8,
    'roll_resid_': 4.8,
    'weight_pct_': 4.5,
    'silica_phr': 4.5,
    'Fill_Factor': 4.0,
    'Target_Temperature': 4.0,
    'Specific_Energy': 4.0,
    'eta_torque': 4.0,
    'eta_app': 4.0,
    'Duration': 3.5,
    'temp_integral': 3.5,
    'power_integral': 3.5,
    'RotorSpeed_Mean': 3.0,
    'Torque_Mean': 3.0,
    'env_': 1.0,
    'idx_': -2.0,
}

PROCESS_ACTION_HINTS = {
    'Stage2_DryMixing_Duration': '干混时间偏离会改变填料初始分散和胶料塑化程度，优先检查投料后到升温控制前的干混节拍。',
    'Stage4_WetMixing_Specific_Energy': '湿混比能量偏高通常意味着油后分散/剪切历史增强，若预测 MNY 偏高或上升过快，应检查油后转速、功率平台和混炼时间。',
    'Stage5_PID_Duration': 'PID 停留时间主要影响热历史和硅烷化反应，白炭黑体系需要重点看是否过长或过短。',
    'Stage6_Discharge_eta_torque_Mean': '排胶阶段视在粘度是最终流变状态的直接信号，异常升高时应回看排胶前温度、转速和扭矩平台。',
    'phys_eta_app_discharge': '排胶视在粘度偏高说明出料时胶料阻力偏大，可结合排胶温度和总能量判断是否欠塑化或填料分散不足。',
    'phys_temp_integral_above_100': '高温积分代表热反应历史，白炭黑/硅烷体系中过高可能导致反应过度，过低可能反应不足。',
    'supplier_rubber_viscosity_avg': '生胶供应商粘度是上游遗传因素，偏高时不要只调密炼参数，应同步看原材料批次。',
    'supplier_silica_moisture_avg': '白炭黑含水率是硅烷化偶联反应的关键催化剂，过低或过高均会干扰反应程度从而引起门尼异常，需结合硅烷、温升等排查。',
    'supplier_silica_surface_area_avg': '白炭黑比表面积决定了偶联活性位点与填料网络强弱，异常偏高会增强粒子自凝聚进而推高门尼，应核对批次检测。',
    'supplier_carbon_black_structure_avg': '炭黑结构值决定了吸留橡胶比例与填充网络刚性，吸油值偏高会直接导致混炼门尼升高，需核对该批次炭黑指标。',
    'supplier_carbon_black_surface_area_avg': '炭黑表面积或粗细程度会直接影响与橡胶大分子链的物理摩擦及补强阻力，表面积偏高会导致剪切粘度上升。',
    'supplier_carbon_black_moisture_avg': '炭黑含水率过高会在高温段气化或干扰剪切，进而导致门尼粘度升高，应检查原材料防潮与烘干状态。',
    'Top_Fill_Factor': '填充系数影响剪切效率和温升，异常时优先检查称量、批重和上顶栓压力相关设置。',
    'Bot_Fill_Factor': '底部填充相关参数影响下辅机/后段混炼状态，需结合底部混炼扭矩和温升判断。',
    'weight_pct_oil': '油含量影响软化和门尼下降趋势，油料比例或加载阶段异常会直接改变最终粘度。',
    'weight_pct_silica': '白炭黑含量提高会增强填料网络和反应需求，需配合硅烷、温度积分和 PID 时间解释。',
    'weight_pct_carbon_black': '炭黑含量提高会增强补强和粘度，需配合干混能量和转子剪切历史解释。',
}


def physical_priority_score(feature_name):
    score = 0.0
    for token, weight in PROCESS_FEATURE_PRIORITY.items():
        if token in feature_name:
            score += weight
    return score


def add_physics_interactions(df):
    """Add compact, interpretable interaction terms for known process regimes."""
    df = df.copy()
    interaction_specs = [
        ('silica_temp_integral_above_100', 'weight_pct_silica', 'ix_silica_heat_load'),
        ('silica_PID_Duration', 'weight_pct_silica', 'ix_silica_pid_time_load'),
        ('Stage4_WetMixing_Specific_Energy', 'weight_pct_oil', 'ix_oil_wetmix_energy_load'),
        ('Stage4_WetMixing_eta_torque_Mean', 'weight_pct_oil', 'ix_oil_wetmix_eta_load'),
        ('Stage2_DryMixing_Specific_Energy', 'weight_pct_carbon_black', 'ix_cb_drymix_energy_load'),
        ('Stage2_DryMixing_Duration', 'weight_pct_carbon_black', 'ix_cb_drymix_time_load'),
        ('supplier_rubber_viscosity_avg', 'weight_pct_solid_elastomer', 'ix_rubber_viscosity_elastomer_load'),
        ('supplier_silica_moisture_avg', 'weight_pct_silica', 'ix_silica_moisture_load'),
        ('supplier_carbon_black_moisture_avg', 'weight_pct_carbon_black', 'ix_cb_moisture_carbon_black_load'),
        ('supplier_carbon_black_structure_avg', 'weight_pct_carbon_black', 'ix_cb_structure_load'),
    ]
    for left, right, name in interaction_specs:
        if left in df.columns and right in df.columns:
            df[name] = pd.to_numeric(df[left], errors='coerce') * pd.to_numeric(df[right], errors='coerce')
            
    # Phase 2 High-order environmental interactions
    if 'env_humidity_mean' in df.columns and 'weight_pct_silica' in df.columns:
        df['inter_humidity_silica'] = pd.to_numeric(df['env_humidity_mean'], errors='coerce') * pd.to_numeric(df['weight_pct_silica'], errors='coerce')
    if 'env_humidity_mean' in df.columns and 'is_oil_loading_present' in df.columns:
        df['inter_humidity_no_oil'] = pd.to_numeric(df['env_humidity_mean'], errors='coerce') * (1.0 - pd.to_numeric(df['is_oil_loading_present'], errors='coerce'))
    if 'env_temp_mean' in df.columns and 'Stage2_DryMixing_power_Integral' in df.columns:
        df['inter_temp_energy'] = pd.to_numeric(df['env_temp_mean'], errors='coerce') * pd.to_numeric(df['Stage2_DryMixing_power_Integral'], errors='coerce')
        
    return df


def build_interpretable_feature_set(X_train, y_train, feature_cols, max_features=MAX_INTERPRETABLE_FEATURES):
    """Rank and prune features by data quality, target association, physical priority, and redundancy."""
    audit_rows = []
    y_numeric = pd.to_numeric(y_train, errors='coerce')
    for col in feature_cols:
        series = pd.to_numeric(X_train[col], errors='coerce')
        missing_rate = float(series.isna().mean())
        non_na = series.dropna()
        dominant_rate = float(non_na.value_counts(normalize=True).iloc[0]) if len(non_na) else 1.0
        unique_count = int(non_na.nunique())
        valid = pd.concat([series, y_numeric], axis=1).dropna()
        spearman_corr = valid.iloc[:, 0].corr(valid.iloc[:, 1], method='spearman') if len(valid) >= 20 and unique_count > 1 else np.nan
        abs_corr = 0.0 if pd.isna(spearman_corr) else abs(float(spearman_corr))
        priority = physical_priority_score(col)
        quality = (1.0 - missing_rate) * (1.0 - max(0.0, dominant_rate - 0.80))
        selection_score = abs_corr + 0.03 * priority + 0.10 * quality
        audit_rows.append({
            'Feature': col,
            'missing_rate': missing_rate,
            'dominant_value_rate': dominant_rate,
            'unique_count': unique_count,
            'spearman_corr_to_MNY': spearman_corr,
            'abs_spearman_corr_to_MNY': abs_corr,
            'physical_priority': priority,
            'selection_score': selection_score,
            'drop_reason': '',
        })

    audit_df = pd.DataFrame(audit_rows).sort_values('selection_score', ascending=False)
    initial_keep = []
    for _, row in audit_df.iterrows():
        reason = ''
        if row['missing_rate'] > MAX_MISSING_RATE:
            reason = f"missing_rate>{MAX_MISSING_RATE}"
        elif row['dominant_value_rate'] > MAX_DOMINANT_VALUE_RATE:
            reason = f"dominant_value_rate>{MAX_DOMINANT_VALUE_RATE}"
        elif row['unique_count'] <= 1:
            reason = 'constant_or_single_unique_value'
        if reason:
            audit_df.loc[audit_df['Feature'] == row['Feature'], 'drop_reason'] = reason
        else:
            initial_keep.append(row['Feature'])

    selected = []
    dropped_by_corr = set()
    for feature in initial_keep:
        if len(selected) >= max_features:
            audit_df.loc[audit_df['Feature'] == feature, 'drop_reason'] = 'outside_top_ranked_features'
            continue
        if feature in dropped_by_corr:
            continue
        keep_feature = True
        for kept_feature in selected:
            corr_df = X_train[[feature, kept_feature]].apply(pd.to_numeric, errors='coerce').dropna()
            if len(corr_df) < 20:
                continue
            corr_val = corr_df[feature].corr(corr_df[kept_feature], method='spearman')
            if not pd.isna(corr_val) and abs(corr_val) >= MAX_FEATURE_CORRELATION:
                keep_feature = False
                dropped_by_corr.add(feature)
                audit_df.loc[audit_df['Feature'] == feature, 'drop_reason'] = f'high_corr_with::{kept_feature}'
                break
        if keep_feature:
            selected.append(feature)

    audit_df['selected_for_model'] = audit_df['Feature'].isin(selected)
    return selected, audit_df


def build_process_guidance(importance_df, audit_df, X_train_raw, y_train, top_n=25):
    """Create process-oriented interpretation rows for the most influential model features."""
    rows = []
    if importance_df is None or len(importance_df) == 0:
        return pd.DataFrame(rows)

    audit_lookup = audit_df.set_index('Feature') if audit_df is not None and len(audit_df) else pd.DataFrame()
    y_numeric = pd.to_numeric(y_train, errors='coerce')
    for _, row in importance_df.head(top_n).iterrows():
        feature = row['Feature']
        if feature not in X_train_raw.columns:
            continue
        series = pd.to_numeric(X_train_raw[feature], errors='coerce')
        valid = pd.concat([series, y_numeric], axis=1).dropna()
        spearman_corr = valid.iloc[:, 0].corr(valid.iloc[:, 1], method='spearman') if len(valid) >= 20 else np.nan
        direction = '不稳定/非单调'
        if not pd.isna(spearman_corr):
            if spearman_corr > 0.08:
                direction = '该特征升高时，模型/数据倾向于 MNY 升高'
            elif spearman_corr < -0.08:
                direction = '该特征升高时，模型/数据倾向于 MNY 降低'
            else:
                direction = '单调方向较弱，更多体现交互或分群作用'

        q10 = series.quantile(0.10)
        q50 = series.quantile(0.50)
        q90 = series.quantile(0.90)
        hint = ''
        for token, text in PROCESS_ACTION_HINTS.items():
            if token in feature:
                hint = text
                break
        if not hint:
            if 'ix_' in feature:
                hint = '这是物理交互项，用来解释配方负荷与工艺历史共同变化时对 MNY 的影响。'
            elif 'Duration' in feature:
                hint = '这是阶段时间特征，适合结合该阶段温度、扭矩和能量共同判断节拍是否偏离。'
            elif 'Specific_Energy' in feature or 'power_integral' in feature:
                hint = '这是能量历史特征，可用于判断剪切/分散输入是否偏高或偏低。'
            elif 'eta' in feature or 'Torque' in feature:
                hint = '这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。'
            else:
                hint = '建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。'

        audit_values = audit_lookup.loc[feature].to_dict() if len(audit_lookup) and feature in audit_lookup.index else {}
        rows.append({
            'Feature': feature,
            'model_importance': row.get('Importance', np.nan),
            'spearman_corr_to_MNY': spearman_corr,
            'direction_hint': direction,
            'train_p10': q10,
            'train_p50': q50,
            'train_p90': q90,
            'missing_rate': audit_values.get('missing_rate', np.nan),
            'dominant_value_rate': audit_values.get('dominant_value_rate', np.nan),
            'process_guidance': hint,
        })
    return pd.DataFrame(rows)


def build_reference_profile(X_train_raw, y_train, selected_features):
    rows = []
    for feature in selected_features:
        if feature not in X_train_raw.columns:
            continue
        series = pd.to_numeric(X_train_raw[feature], errors='coerce')
        rows.append({
            'Feature': feature,
            'train_mean': series.mean(),
            'train_std': series.std(),
            'train_p01': series.quantile(0.01),
            'train_p05': series.quantile(0.05),
            'train_p10': series.quantile(0.10),
            'train_p50': series.quantile(0.50),
            'train_p90': series.quantile(0.90),
            'train_p95': series.quantile(0.95),
            'train_p99': series.quantile(0.99),
        })
    profile_df = pd.DataFrame(rows)
    target = pd.to_numeric(y_train, errors='coerce')
    metadata = {
        'target_mean': float(target.mean()),
        'target_std': float(target.std()),
        'target_p10': float(target.quantile(0.10)),
        'target_p50': float(target.quantile(0.50)),
        'target_p90': float(target.quantile(0.90)),
    }
    return profile_df, metadata


def save_model_bundle(bundle_path, best_model, preprocessor, compact_features, selected_features,
                      X_train_raw, X_train_model, y_train, subset_name, best_model_name,
                      best_params, best_cv_score, test_metrics, feature_audit_df,
                      guidance_df, importance_df, family_medians=None, track_median=0.0):
    reference_profile_df, target_metadata = build_reference_profile(X_train_raw, y_train, compact_features)
    
    # Precompute covariance for Mahalanobis applicability domain
    try:
        from scipy.linalg import pinv
        cov = np.cov(X_train_model.T)
        inv_cov = pinv(cov)
        mean_vec = np.mean(X_train_model, axis=0)
    except Exception as e:
        print(f"Warning precomputing covariance in save_model_bundle: {e}")
        inv_cov = None
        mean_vec = None
        
    bundle = {
        'model': best_model,
        'preprocessor': preprocessor,
        'compact_features': compact_features,
        'selected_features': selected_features,
        'train_model_matrix': X_train_model,
        'train_target': y_train.reset_index(drop=True),
        'train_raw_df': X_train_raw.copy(),
        'inv_covariance_matrix': inv_cov,
        'mean_vector': mean_vec,
        'reference_profile': reference_profile_df,
        'target_metadata': target_metadata,
        'subset_name': subset_name,
        'model_name': best_model_name,
        'best_params': best_params,
        'best_cv_rmse': best_cv_score,
        'test_metrics': test_metrics,
        'feature_audit': feature_audit_df,
        'process_guidance': guidance_df,
        'feature_importance': importance_df,
        'applicability_distance_quantiles': None,
        'family_medians': family_medians,
        'track_median': track_median,
    }
    joblib.dump(bundle, bundle_path)
    return bundle_path

def fit_filter_anomalies_isolation_forest(df_train, df_test, feature_cols, contamination=0.05, results_dir="results", brain_subset_dir=None):
    print(f"\n--- Running Leakage-Free Curve Anomaly Detection using Isolation Forest ({results_dir}) ---")
    print(f"Initial train shape: {df_train.shape} | test shape: {df_test.shape}")
    
    # Select core curve-derived physical features for anomaly detection (temp, power, torque, duration)
    core_features = [
        col for col in feature_cols 
        if any(kw in col for kw in ['temp_Mean', 'power_Mean', 'Torque_Mean', 'temp_integral', 'power_integral', 'Duration'])
    ]
    print(f"Selected {len(core_features)} core features for Isolation Forest.")
    
    # Initialize anomaly flag columns
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train['is_anomaly_isolation_forest'] = np.nan
    df_test['is_anomaly_isolation_forest'] = np.nan
    
    clean_train_dfs = []
    clean_test_dfs = []
    dropped_train = 0
    dropped_test = 0
    
    # Group by CompoundName on training set
    train_groups = {compound: group for compound, group in df_train.groupby('CompoundName')}
    test_groups = {compound: group for compound, group in df_test.groupby('CompoundName')}
    
    all_compounds = set(list(train_groups.keys()) + list(test_groups.keys()))
    
    for compound in all_compounds:
        train_group = train_groups.get(compound, pd.DataFrame()).copy()
        test_group = test_groups.get(compound, pd.DataFrame()).copy()
        
        # If train group has fewer than 10 samples, we cannot reliably fit Isolation Forest
        if len(train_group) < 10:
            if len(train_group) > 0:
                train_group['is_anomaly_isolation_forest'] = 0.0  # Kept, sample count too low to evaluate
                clean_train_dfs.append(train_group)
                train_groups[compound] = train_group
            if len(test_group) > 0:
                test_group['is_anomaly_isolation_forest'] = 0.0  # Kept, sample count too low to evaluate
                clean_test_dfs.append(test_group)
                test_groups[compound] = test_group
            continue
            
        X_train_group = train_group[core_features].copy()
        # Drop columns that are completely NaN in this specific compound group to prevent imputer error
        nan_cols = X_train_group.columns[X_train_group.isna().all()].tolist()
        group_features = [c for c in core_features if c not in nan_cols]
        
        if len(group_features) == 0:
            train_group['is_anomaly_isolation_forest'] = 0.0
            clean_train_dfs.append(train_group)
            train_groups[compound] = train_group
            if len(test_group) > 0:
                test_group['is_anomaly_isolation_forest'] = 0.0
                clean_test_dfs.append(test_group)
                test_groups[compound] = test_group
            continue
            
        X_train_subset = X_train_group[group_features]
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train_subset)
        
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        train_preds = iso.fit_predict(X_train_imputed)
        
        # Flag anomalies: 1.0 for anomaly (-1 from predict), 0.0 for normal (1 from predict)
        train_group['is_anomaly_isolation_forest'] = np.where(train_preds == -1, 1.0, 0.0)
        
        normal_train_mask = (train_preds == 1)
        anomalies_train_mask = (train_preds == -1)
        
        clean_train_dfs.append(train_group[normal_train_mask])
        dropped_train += np.sum(anomalies_train_mask)
        
        # Update the original group in train_groups to keep track of anomaly status for logging
        train_groups[compound] = train_group
        
        if len(test_group) > 0:
            X_test_group = test_group[group_features].copy()
            X_test_imputed = imputer.transform(X_test_group)
            test_preds = iso.predict(X_test_imputed)
            
            test_group['is_anomaly_isolation_forest'] = np.where(test_preds == -1, 1.0, 0.0)
            
            normal_test_mask = (test_preds == 1)
            anomalies_test_mask = (test_preds == -1)
            
            clean_test_dfs.append(test_group[normal_test_mask])
            dropped_test += np.sum(anomalies_test_mask)
            
            # Update test_groups
            test_groups[compound] = test_group
            
    df_train_cleaned = pd.concat(clean_train_dfs).sort_index() if clean_train_dfs else pd.DataFrame(columns=df_train.columns)
    df_test_cleaned = pd.concat(clean_test_dfs).sort_index() if clean_test_dfs else pd.DataFrame(columns=df_test.columns)
    
    # Output the logging report of all checked batches and their anomaly status
    all_evaluated = []
    for compound in all_compounds:
        tr_g = train_groups.get(compound, pd.DataFrame())
        te_g = test_groups.get(compound, pd.DataFrame())
        if len(tr_g) > 0:
            all_evaluated.append(tr_g)
        if len(te_g) > 0:
            all_evaluated.append(te_g)
            
    if all_evaluated:
        df_all_evaluated = pd.concat(all_evaluated).sort_values(by=['CompoundName', 'OrderID', 'BatchNumber'])
        # Keep key identifier columns for readability
        log_cols = [
            'OrderID', 'BatchNumber', 'CompoundName', 'MNY', 'is_oil_loading_present',
            'is_anomaly_isolation_forest', 'is_silica_system'
        ]
        log_cols = [c for c in log_cols if c in df_all_evaluated.columns]
        
        # Save to CSV
        report_path1 = os.path.join(results_dir, "isolation_forest_anomalies_report.csv")
        report_path2 = os.path.join(brain_subset_dir, "isolation_forest_anomalies_report.csv") if brain_subset_dir else None
        
        safe_to_csv(df_all_evaluated[log_cols], report_path1, index=False, encoding='utf-8-sig')
        if report_path2:
            safe_to_csv(df_all_evaluated[log_cols], report_path2, index=False, encoding='utf-8-sig')
        print(f"Saved Isolation Forest anomaly report to {report_path1} and {report_path2}")
        
    print(f"Dropped {dropped_train} train anomalies using Isolation Forest.")
    print(f"Dropped {dropped_test} test anomalies using Isolation Forest.")
    print(f"Cleaned train shape: {df_train_cleaned.shape} | Cleaned test shape: {df_test_cleaned.shape}")
    
    return df_train_cleaned, df_test_cleaned


def tune_model_optuna(X_train, y_train, model_name, n_trials=25):
    print(f"\n--- Optimizing {model_name} using Optuna ({n_trials} trials, RMSE objective) ---")
    
    def objective(trial):
        if model_name == 'LightGBM':
            params = {
                'objective': 'huber',
                'n_estimators': trial.suggest_int('n_estimators', 200, 900),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 7, 63),
                'min_child_samples': trial.suggest_int('min_child_samples', 30, 220),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 3.0),
                'subsample': trial.suggest_float('subsample', 0.65, 0.95),
                'subsample_freq': 1,
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.45, 0.85),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 100.0, log=True),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            model = lgb.LGBMRegressor(**params)
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 900),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
                'subsample': trial.suggest_float('subsample', 0.65, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.45, 0.85),
                'min_child_weight': trial.suggest_float('min_child_weight', 3.0, 40.0, log=True),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 100.0, log=True),
                'objective': 'reg:squarederror',
                'random_state': 42,
                'n_jobs': -1
            }
            model = xgb.XGBRegressor(**params)
        else:
            return np.inf
            
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        return -np.mean(scores['test_score'])
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"-> Optuna Tuning Complete. Best {model_name} CV RMSE: {study.best_value:.4f}")
    return study.best_params, study.best_value


def compute_rolling_features(df_labeled):
    print("\n--- Computing Leakage-Free Chronological Rolling Features ---")
    df = df_labeled.copy()
    df['OrderStartTime'] = pd.to_datetime(df['OrderStartTime'])
    df['test_result_start_time'] = pd.to_datetime(df['test_result_start_time'])
    
    # visible_time is when the MNY lab result becomes available.
    # Default to OrderStartTime + 6 hours if test_result_start_time is missing.
    df['visible_time'] = df['test_result_start_time'].fillna(df['OrderStartTime'] + pd.Timedelta(hours=6))
    
    # Sort chronologically by OrderStartTime
    df = df.sort_values('OrderStartTime').reset_index(drop=True)
    
    df['roll_mny_mean_3b_same_comp'] = np.nan
    df['roll_mny_std_10b_same_comp'] = np.nan
    
    comp_history = {}
    
    for idx, row in df.iterrows():
        comp = row['CompoundName']
        t_start = row['OrderStartTime']
        
        if comp not in comp_history:
            comp_history[comp] = []
            
        # Filter past records where visible_time < current OrderStartTime
        visible = [h for h in comp_history[comp] if h['visible_time'] < t_start]
        
        if len(visible) >= 1:
            last_3 = visible[-3:]
            df.at[idx, 'roll_mny_mean_3b_same_comp'] = np.mean([h['MNY'] for h in last_3])
        if len(visible) >= 2:
            last_10 = visible[-10:]
            df.at[idx, 'roll_mny_std_10b_same_comp'] = np.std([h['MNY'] for h in last_10]) if len(last_10) > 1 else 0.0
            
        # Append current row
        comp_history[comp].append({
            'visible_time': row['visible_time'],
            'MNY': row['MNY']
        })
        
    return df


def compute_rolling_residuals(df_subset, family_medians, track_median):
    df = df_subset.copy()
    df['OrderStartTime'] = pd.to_datetime(df['OrderStartTime'])
    df['test_result_start_time'] = pd.to_datetime(df['test_result_start_time'])
    df['visible_time'] = df['test_result_start_time'].fillna(df['OrderStartTime'] + pd.Timedelta(hours=6))
    
    # Calculate MNY_residual using the passed family_medians
    df['family_baseline'] = df['CompoundName'].map(family_medians).fillna(track_median)
    df['MNY_residual'] = df['MNY'] - df['family_baseline']
    
    df['roll_resid_mean_5b_family'] = np.nan
    df['roll_resid_mean_10b_mixer'] = np.nan
    
    def get_compound_family(name):
        if not isinstance(name, str):
            return 'Unknown'
        prefix = name.split()[0] if name.split() else name
        return prefix.rstrip('-')
        
    df['compound_family'] = df['CompoundName'].apply(get_compound_family)
    
    family_history = {}
    mixer_history = {}
    
    for idx, row in df.iterrows():
        family = row['compound_family']
        mixer = row['MixerLine']
        t_start = row['OrderStartTime']
        
        # Family rolling residual
        if family not in family_history:
            family_history[family] = []
        visible_fam = [h for h in family_history[family] if h['visible_time'] < t_start]
        if len(visible_fam) >= 1:
            df.at[idx, 'roll_resid_mean_5b_family'] = np.mean([h['y_resid'] for h in visible_fam[-5:]])
            
        # Mixer rolling residual
        if mixer not in mixer_history:
            mixer_history[mixer] = []
        visible_mix = [h for h in mixer_history[mixer] if h['visible_time'] < t_start]
        if len(visible_mix) >= 1:
            df.at[idx, 'roll_resid_mean_10b_mixer'] = np.mean([h['y_resid'] for h in visible_mix[-10:]])
            
        # Append
        family_history[family].append({
            'visible_time': row['visible_time'],
            'y_resid': row['MNY_residual']
        })
        mixer_history[mixer].append({
            'visible_time': row['visible_time'],
            'y_resid': row['MNY_residual']
        })
        
    return df


def run_modeling_pipeline(df_labeled_full, df_all_rows, is_oil_loading_present_val, subset_name, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    brain_subset_dir = os.path.join(BRAIN_DIR, results_dir)
    os.makedirs(brain_subset_dir, exist_ok=True)
    
    PARITY_PNG = os.path.join(brain_subset_dir, "mooney_model_parity_plot.png")
    IMPORTANCE_PNG = os.path.join(brain_subset_dir, "mooney_model_feature_importance.png")
    MODEL_REPORT_MD = os.path.join(brain_subset_dir, "mny_predictive_modeling_report.md")
    
    PARITY_PNG_RESULTS = os.path.join(results_dir, "mooney_model_parity_plot.png")
    IMPORTANCE_PNG_RESULTS = os.path.join(results_dir, "mooney_model_feature_importance.png")
    MODEL_REPORT_MD_RESULTS = os.path.join(results_dir, "mny_predictive_modeling_report.md")
    FEATURE_AUDIT_CSV = os.path.join(results_dir, "interpretable_feature_audit.csv")
    GUIDANCE_CSV = os.path.join(results_dir, "process_adjustment_guidance.csv")
    FEATURE_AUDIT_CSV_BRAIN = os.path.join(brain_subset_dir, "interpretable_feature_audit.csv")
    GUIDANCE_CSV_BRAIN = os.path.join(brain_subset_dir, "process_adjustment_guidance.csv")
    MODEL_BUNDLE = os.path.join(results_dir, "mooney_model_bundle.joblib")
    MODEL_BUNDLE_BRAIN = os.path.join(brain_subset_dir, "mooney_model_bundle.joblib")
    
    df_labeled_full = df_labeled_full.copy()
    if 'MixerLine' in df_labeled_full.columns:
        df_labeled_full['MixerLine'] = df_labeled_full['MixerLine'].astype(str).str.strip().fillna('UNKNOWN')
        dummies = pd.get_dummies(df_labeled_full['MixerLine'], prefix='MixerLine', dtype=float)
        df_labeled_full = pd.concat([df_labeled_full, dummies], axis=1)
        
    df_labeled = add_physics_interactions(df_labeled_full)
    
    print("\n--- Constructing Gated/Conditioned Features for Carbon Black vs Silica Regimes ---")
    df_labeled['is_silica_system'] = ((df_labeled['silica_phr'] >= 25.0) & (df_labeled['weight_pct_silian'] > 0.0)).astype(float)
    
    n_silica = (df_labeled['is_silica_system'] == 1.0).sum()
    n_cb = (df_labeled['is_silica_system'] == 0.0).sum()
    print(f"  Silica system batches: {n_silica} | Carbon Black system batches: {n_cb}")
    
    # Gated features for Silica (Chemical Reaction in PID stage is active)
    df_labeled['silica_PID_Duration'] = df_labeled['Stage5_PID_Duration'].where(df_labeled['is_silica_system'] == 1.0, 0.0)
    df_labeled['silica_PID_Specific_Energy'] = df_labeled['Stage5_PID_Specific_Energy'].where(df_labeled['is_silica_system'] == 1.0, 0.0)
    df_labeled['silica_temp_integral_above_100'] = df_labeled['phys_temp_integral_above_100'].where(df_labeled['is_silica_system'] == 1.0, 0.0)
    df_labeled['silica_PID_temp_Mean'] = df_labeled['Stage5_PID_temp_Mean'].where(df_labeled['is_silica_system'] == 1.0, 0.0)
    
    # Gated features for Carbon Black (Mechanical mastication & dispersion are active, no silanization)
    df_labeled['cb_DryMixing_Duration'] = df_labeled['Stage2_DryMixing_Duration'].where(df_labeled['is_silica_system'] == 0.0, 0.0)
    df_labeled['cb_DryMixing_Specific_Energy'] = df_labeled['Stage2_DryMixing_Specific_Energy'].where(df_labeled['is_silica_system'] == 0.0, 0.0)
    df_labeled['cb_power_decay_slope'] = df_labeled['Stage2_power_decay_slope'].where(df_labeled['is_silica_system'] == 0.0, 0.0)
    df_labeled = add_physics_interactions(df_labeled)
    
    if len(df_labeled) < 20:
        print(f"Error: Too few labeled samples to train {subset_name} model. Need at least 20.")
        return
        
    # Define features and exclude metadata columns
    exclude_cols = [
        'OrderID', 'BatchNumber', 'CompoundName', 'CompoundDescription', 
        'batch_information_fk_final', 'temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam',
        'CurrentValue', 'PrevStepValue', 'MNY target',
        'ratio_nr_rubber', 'ratio_filler_polymer', 'ratio_oil_polymer', 'ratio_oil_filler',
        'is_oil_loading_present', 'silica_phr',
        'time_reach_discharge', 'time_total_mixing'
    ]
    numeric_cols = df_labeled.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols and col != 'MNY' and not col.startswith('idx_')]
    
    # 1. Train-Test Split (80/20) first to prevent data leakage in anomaly detection
    df_train, df_test = train_test_split(df_labeled, test_size=0.2, random_state=42)
    
    # 2. Run Isolation Forest Anomaly Detection (Fit on Train, Predict/Filter on Test)
    df_train_cleaned, df_test_cleaned = fit_filter_anomalies_isolation_forest(
        df_train, df_test, feature_cols, contamination=0.05, 
        results_dir=results_dir, brain_subset_dir=brain_subset_dir
    )
    
    # Check for negative phase durations or idx_t4_PID_Start > idx_t5_Discharge after Isolation Forest
    print("\n--- Checking for negative phase durations after Isolation Forest ---")
    duration_cols = [col for col in df_train_cleaned.columns if col.endswith('_Duration')]
    
    def filter_negative_durations(df_subset, name):
        if len(df_subset) == 0:
            return df_subset
        cond_neg_duration = pd.Series(False, index=df_subset.index)
        for col in duration_cols:
            cond_neg_duration = cond_neg_duration | (df_subset[col] < 0)
            
        cond_pid_discharge = pd.Series(False, index=df_subset.index)
        if 'idx_t4_PID_Start' in df_subset.columns and 'idx_t5_Discharge' in df_subset.columns:
            cond_pid_discharge = df_subset['idx_t4_PID_Start'] > df_subset['idx_t5_Discharge']
            
        cond_discharge_bot = pd.Series(False, index=df_subset.index)
        if 'idx_t5_Discharge' in df_subset.columns and 'idx_t6_BottomMixing_Start' in df_subset.columns:
            cond_discharge_bot = df_subset['idx_t5_Discharge'] > df_subset['idx_t6_BottomMixing_Start']
            
        cond_bot_end = pd.Series(False, index=df_subset.index)
        if 'idx_t6_BottomMixing_Start' in df_subset.columns and 'idx_t7_BottomMixing_End' in df_subset.columns:
            cond_bot_end = df_subset['idx_t6_BottomMixing_Start'] > df_subset['idx_t7_BottomMixing_End']
            
        invalid_mask = cond_neg_duration | cond_pid_discharge | cond_discharge_bot | cond_bot_end
        
        if invalid_mask.any():
            invalid_rows = df_subset[invalid_mask]
            print(f"WARNING: Found {len(invalid_rows)} rows with negative/invalid phase durations in {name} set! Dropping them.")
            for _, r in invalid_rows.head(15).iterrows():
                msg = f"  Dropped Batch: OrderID={r.get('OrderID')}, Batch={r.get('BatchNumber')}, Compound={r.get('CompoundName')}"
                if 'idx_t4_PID_Start' in r and 'idx_t5_Discharge' in r and 'idx_t6_BottomMixing_Start' in r:
                    msg += f" | t4_PID={r['idx_t4_PID_Start']}, t5_Discharge={r['idx_t5_Discharge']}, t6_BotMix={r['idx_t6_BottomMixing_Start']}"
                neg_stages = [f"{col}={r[col]}" for col in duration_cols if r[col] < 0]
                if neg_stages:
                    msg += f" | Negatives: {', '.join(neg_stages)}"
                print(msg)
            df_clean = df_subset[~invalid_mask].copy()
        else:
            print(f"No negative phase durations found in {name} set.")
            df_clean = df_subset
        return df_clean

    df_train_cleaned = filter_negative_durations(df_train_cleaned, "Train")
    df_test_cleaned = filter_negative_durations(df_test_cleaned, "Test")

    # Add flag to distinguish train and test for chronological residual rolling features
    df_train_cleaned['_is_train'] = 1
    df_test_cleaned['_is_train'] = 0
    
    df_combined = pd.concat([df_train_cleaned, df_test_cleaned]).sort_values('OrderStartTime').reset_index(drop=True)
    
    # Calculate family_medians from train portion only
    train_subset = df_combined[df_combined['_is_train'] == 1]
    family_medians = train_subset.groupby('CompoundName')['MNY'].median().to_dict()
    track_median = float(train_subset['MNY'].median())
    
    # Compute rolling residuals chronologically on combined set
    df_combined = compute_rolling_residuals(df_combined, family_medians, track_median)
    
    # Split back
    df_train_cleaned = df_combined[df_combined['_is_train'] == 1].drop(columns=['_is_train']).copy()
    df_test_cleaned = df_combined[df_combined['_is_train'] == 0].drop(columns=['_is_train']).copy()
    
    # Add rolling residual features to feature_cols
    feature_cols = list(feature_cols) + ['roll_resid_mean_5b_family', 'roll_resid_mean_10b_mixer']

    X_train = df_train_cleaned[feature_cols].copy()
    y_train = df_train_cleaned['MNY_residual'].copy()
    X_test = df_test_cleaned[feature_cols].copy()
    y_test = df_test_cleaned['MNY_residual'].copy()
    
    # Drop columns that are completely NaN or have zero variance (constant) *based on the training set*
    nan_cols = X_train.columns[X_train.isna().all()].tolist()
    const_cols = [col for col in X_train.columns if X_train[col].nunique() <= 1]
    cols_to_drop = list(set(nan_cols + const_cols))
    if cols_to_drop:
        print(f"Dropping redundant/empty columns in {subset_name}: {cols_to_drop}")
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)
        feature_cols = [col for col in feature_cols if col not in cols_to_drop]

    print("\n--- Overriding with Formulation-Free Streamlined Feature Set (with Recipe Weights) ---")
    mixer_cols = [col for col in X_train.columns if col.startswith('MixerLine_')]
    recipe_features = [
        'weight_pct_solid_elastomer', 'weight_pct_natural_rubber', 'weight_pct_silica',
        'weight_pct_oil', 'weight_pct_silian', 'weight_pct_carbon_black',
        'supplier_rubber_viscosity_avg',
        'supplier_silica_moisture_avg', 'supplier_silica_surface_area_avg',
        'supplier_carbon_black_structure_avg', 'supplier_carbon_black_surface_area_avg',
        'supplier_carbon_black_moisture_avg',
        'Top_Fill_Factor', 'Bot_Fill_Factor'
    ] + mixer_cols
    
    rolling_features = [
        'roll_mny_mean_3b_same_comp', 'roll_mny_std_10b_same_comp',
        'roll_resid_mean_5b_family', 'roll_resid_mean_10b_mixer'
    ]
    
    if is_oil_loading_present_val == 1.0:
        streamlined_features = [
            'phys_init_temp', 'phys_temp_integral_above_100',
            'Stage2_DryMixing_power_Mean', 'Stage2_DryMixing_power_Integral',
            'Stage3_OilLoading_temp_Mean', 'Stage3_OilLoading_temp_Std',
            'Stage4_WetMixing_power_Mean', 'Stage4_WetMixing_Torque_Mean', 'Stage4_WetMixing_Duration',
            'Stage6_BottomMixing_Torque_Mean', 'Stage6_BottomMixing_power_Mean', 'Stage6_BottomMixing_power_Integral', 'Stage6_BottomMixing_Torque_Integral'
        ] + recipe_features + rolling_features
    else:
        streamlined_features = [
            'phys_init_temp', 'phys_temp_rise_rate', 'phys_temp_integral_above_100',
            'Stage1_Loading_WayofRam_Mean', 'Stage1_Loading_RotorSpeed_Integral',
            'Stage2_DryMixing_temp_Std', 'Stage2_DryMixing_WayofRam_Std', 'Stage2_DryMixing_power_Integral',
            'Stage5_PID_WayofRam_Std', 'Stage5_PID_power_Integral', 'Stage5_PID_RotorSpeed_Integral',
            'Stage6_BottomMixing_Torque_Mean', 'Stage6_BottomMixing_power_Mean'
        ] + recipe_features + rolling_features
    
    compact_features = [f for f in streamlined_features if f in X_train.columns]
    missing_feats = [f for f in streamlined_features if f not in X_train.columns]
    if missing_feats:
        print(f"[WARNING] Some streamlined features were missing from X_train: {missing_feats}")
        
    # We still call build_interpretable_feature_set to write the audit CSVs properly
    _, feature_audit_df = build_interpretable_feature_set(X_train, y_train, compact_features, max_features=len(compact_features))
    safe_to_csv(feature_audit_df, FEATURE_AUDIT_CSV, index=False, encoding='utf-8-sig')
    safe_to_csv(feature_audit_df, FEATURE_AUDIT_CSV_BRAIN, index=False, encoding='utf-8-sig')
    print(f"Selected {len(compact_features)} compact features from {len(feature_cols)} candidates.")
    print(f"Saved feature audit to {FEATURE_AUDIT_CSV}")

    X_train_raw_for_guidance = X_train[compact_features].copy()
    X_test_raw_for_guidance = X_test[compact_features].copy()
    X_train = X_train_raw_for_guidance.copy()
    X_test = X_test_raw_for_guidance.copy()
    feature_cols = compact_features
        
    print(f"Feature matrix shape - Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Target vector shape - Train: {y_train.shape} | Test: {y_test.shape}")
    
    # 3. Define Preprocessing Pipeline
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('variance', VarianceThreshold(threshold=0.01))
    ])
    
    # Fit-transform the training data, transform test data
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    # Get features after variance threshold selection
    selected_indices = preprocessor.named_steps['variance'].get_support(indices=True)
    selected_features = [feature_cols[i] for i in selected_indices]
    print(f"Features after preprocessing selection: {len(selected_features)} / {len(feature_cols)}")

    X_train_model = pd.DataFrame(X_train_prep, columns=selected_features, index=X_train.index)
    X_test_model = pd.DataFrame(X_test_prep, columns=selected_features, index=X_test.index)
    
    # 4. Model Comparison: Base Models using 5-Fold Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    base_models = {
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=12),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
        'XGBoost (Baseline)': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=120, random_state=42, max_depth=5, n_jobs=-1),
        'LightGBM (Huber)': lgb.LGBMRegressor(objective='huber', n_estimators=120, random_state=42, max_depth=5, n_jobs=-1, verbose=-1),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=120, random_state=42),
        'CatBoost (Baseline)': CatBoostRegressor(iterations=120, random_seed=42, verbose=0)
    }
    
    print("\n--- Running Baseline 5-Fold Cross Validation ---")
    cv_results = {}
    for name, model in base_models.items():
        scores = cross_validate(
            model, X_train_model, y_train, cv=cv,
            scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'},
            n_jobs=-1
        )
        
        mean_r2 = np.mean(scores['test_r2'])
        mean_mae = -np.mean(scores['test_mae'])
        mean_rmse = -np.mean(scores['test_rmse'])
        
        cv_results[name] = {
            'R2_CV': mean_r2,
            'MAE_CV': mean_mae,
            'RMSE_CV': mean_rmse
        }
        print(f"{name:<25} | CV R^2: {mean_r2:.4f} | CV MAE: {mean_mae:.4f} | CV RMSE: {mean_rmse:.4f}")
        
    # 5. Hyperparameter Tuning using Optuna
    lgb_params, lgb_cv_score = tune_model_optuna(X_train_model, y_train, 'LightGBM', n_trials=25)
    xgb_params, xgb_cv_score = tune_model_optuna(X_train_model, y_train, 'XGBoost', n_trials=25)
    
    # 6. Final Model Selection -> Replaced by StackingRegressor
    print("\n--- Creating Stacked Ensemble (LGBM + XGB + CatBoost + HistGBM + RF) ---")
    best_model_name = "Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF)"
    
    tuned_lgb = lgb.LGBMRegressor(**lgb_params, objective='huber', random_state=42, n_jobs=-1, verbose=-1)
    tuned_xgb = xgb.XGBRegressor(**xgb_params, objective='reg:squarederror', random_state=42, n_jobs=-1)
    tuned_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=12)
    catboost_base = CatBoostRegressor(iterations=120, random_seed=42, verbose=0)
    histgbm_base = HistGradientBoostingRegressor(max_iter=120, random_state=42)
    
    best_model = StackingRegressor(
        estimators=[
            ('lgb', tuned_lgb),
            ('xgb', tuned_xgb),
            ('catboost', catboost_base),
            ('histgbm', histgbm_base),
            ('rf', tuned_rf)
        ],
        final_estimator=Ridge(alpha=5.0),
        n_jobs=-1
    )
    
    print("Evaluating Tuned LightGBM with 5-Fold Cross Validation...")
    lgb_scores = cross_validate(
        tuned_lgb, X_train_model, y_train, cv=cv,
        scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'},
        n_jobs=-1
    )
    cv_results['LightGBM (Tuned)'] = {
        'R2_CV': np.mean(lgb_scores['test_r2']),
        'MAE_CV': -np.mean(lgb_scores['test_mae']),
        'RMSE_CV': -np.mean(lgb_scores['test_rmse'])
    }
    
    print("Evaluating Tuned XGBoost with 5-Fold Cross Validation...")
    xgb_scores = cross_validate(
        tuned_xgb, X_train_model, y_train, cv=cv,
        scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'},
        n_jobs=-1
    )
    cv_results['XGBoost (Tuned)'] = {
        'R2_CV': np.mean(xgb_scores['test_r2']),
        'MAE_CV': -np.mean(xgb_scores['test_mae']),
        'RMSE_CV': -np.mean(xgb_scores['test_rmse'])
    }
    
    best_cv_score = cv_results['LightGBM (Tuned)']['RMSE_CV']
    best_params = f"LGBM: {lgb_params} | XGB: {xgb_params}"
    
    print(f"Selected Model: {best_model_name}")
    print(f"Optimal Hyperparameters: {best_params}")
    
    # Fit Final Model and Evaluate
    best_model.fit(X_train_model, y_train)
    y_pred_residual = best_model.predict(X_test_model)
    
    # Convert back to absolute Mooney values for evaluation!
    y_pred = y_pred_residual + df_test_cleaned['family_baseline'].values
    y_test_absolute = df_test_cleaned['MNY'].values
    
    test_r2 = r2_score(y_test_absolute, y_pred)
    test_mae = mean_absolute_error(y_test_absolute, y_pred)
    test_rmse = root_mean_squared_error(y_test_absolute, y_pred)
    
    print(f"\nFinal Tuned Model Performance on Test Set ({subset_name}):")
    print(f"Test R^2: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # 7. Generate Parity Plot
    print(f"Generating parity plot at: {PARITY_PNG}")
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test_absolute, y_pred, alpha=0.4, color='#1f77b4', edgecolors='none', s=25, label='Predicted vs Actual')
    
    min_val = min(y_test_absolute.min(), y_pred.min()) - 1
    max_val = max(y_test_absolute.max(), y_pred.max()) + 1
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Parity Line (y = x)')
    
    plt.xlabel('Measured Lab Mooney (MNY)', fontsize=12)
    plt.ylabel('Predicted Mooney (MNY)', fontsize=12)
    plt.title(f'Parity Plot for Mooney Viscosity prediction ({subset_name})\nModel: {best_model_name}', fontsize=13, fontweight='bold')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, alpha=0.3)
    
    textstr = '\n'.join((
        f'Test $R^2$: {test_r2:.4f}',
        f'Test MAE: {test_mae:.4f}',
        f'Test RMSE: {test_rmse:.4f}'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(PARITY_PNG, dpi=120)
    plt.savefig(PARITY_PNG_RESULTS, dpi=120)
    plt.close()
    
    # 8. Extract and Plot Feature Importance
    print("Extracting feature importances...")
    importances = None
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'estimators_'):
        base_importances = []
        for est in best_model.estimators_:
            if hasattr(est, 'feature_importances_'):
                imp = est.feature_importances_
                if imp is not None and len(imp) > 0:
                    norm_imp = imp / np.sum(imp) if np.sum(imp) > 0 else imp
                    base_importances.append(norm_imp)
        if base_importances:
            importances = np.mean(base_importances, axis=0)
        
    if importances is not None:
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print("\nTop 15 most important features:")
        print(importance_df.head(15).to_string(index=False))
        
        print(f"Generating feature importance plot at: {IMPORTANCE_PNG}")
        plt.figure(figsize=(10, 6))
        top_n = importance_df.head(15).sort_values(by='Importance', ascending=True)
        plt.barh(top_n['Feature'], top_n['Importance'], color='#2ca02c', alpha=0.8)
        plt.xlabel('Importance Weight', fontsize=12)
        plt.title(f'Top 15 Predictive Features for Mooney (MNY) ({subset_name})\nModel: {best_model_name}', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(IMPORTANCE_PNG, dpi=120)
        plt.savefig(IMPORTANCE_PNG_RESULTS, dpi=120)
        plt.close()
    else:
        importance_df = pd.DataFrame(columns=['Feature', 'Importance'])
        print("Feature importances not supported by best model.")

    guidance_df = build_process_guidance(importance_df, feature_audit_df, X_train_raw_for_guidance, y_train, top_n=25)
    safe_to_csv(guidance_df, GUIDANCE_CSV, index=False, encoding='utf-8-sig')
    safe_to_csv(guidance_df, GUIDANCE_CSV_BRAIN, index=False, encoding='utf-8-sig')
    print(f"Saved process adjustment guidance to {GUIDANCE_CSV}")

    # Calculate Compound Family metrics on Test Set
    test_df_eval = df_test_cleaned.copy()
    test_df_eval['Predicted_MNY'] = y_pred
    test_df_eval['Error'] = test_df_eval['Predicted_MNY'] - test_df_eval['MNY']
    test_df_eval['Absolute_Error'] = test_df_eval['Error'].abs()
    
    # Save test set predictions to CSV for downstream comparisons (e.g., compare_model_performance_by_compound)
    test_df_save = test_df_eval.copy()
    test_df_save = test_df_save.rename(columns={'MNY': 'Actual_MNY'})
    cols_to_save = ['OrderID', 'BatchNumber', 'CompoundName', 'Actual_MNY', 'Predicted_MNY', 'Error', 'Absolute_Error']
    cols_to_save = [c for c in cols_to_save if c in test_df_save.columns]
    
    TEST_PREDS_CSV = os.path.join(results_dir, "test_set_predictions.csv")
    TEST_PREDS_CSV_BRAIN = os.path.join(brain_subset_dir, "test_set_predictions.csv")
    safe_to_csv(test_df_save[cols_to_save], TEST_PREDS_CSV, index=False, encoding='utf-8-sig')
    safe_to_csv(test_df_save[cols_to_save], TEST_PREDS_CSV_BRAIN, index=False, encoding='utf-8-sig')
    print(f"Saved GBDT test set predictions to {TEST_PREDS_CSV} and {TEST_PREDS_CSV_BRAIN}")
    
    def get_compound_family(name):
        if not isinstance(name, str):
            return 'Unknown'
        prefix = name.split()[0] if name.split() else name
        return prefix.rstrip('-')
        
    test_df_eval['compound_family'] = test_df_eval['CompoundName'].apply(get_compound_family)
    
    family_metrics = []
    for family, group in test_df_eval.groupby('compound_family'):
        n_rows = len(group)
        if n_rows == 0:
            continue
        lab_mean = group['MNY'].mean()
        pred_mean = group['Predicted_MNY'].mean()
        bias = pred_mean - lab_mean
        mae = group['Absolute_Error'].mean()
        rmse = root_mean_squared_error(group['MNY'], group['Predicted_MNY'])
        r2 = r2_score(group['MNY'], group['Predicted_MNY']) if n_rows >= 2 and group['MNY'].nunique() > 1 else np.nan
        family_metrics.append({
            'compound_family': family,
            'test_rows': n_rows,
            'lab_mean': lab_mean,
            'pred_mean': pred_mean,
            'bias_pred_minus_lab': bias,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })
    family_df = pd.DataFrame(family_metrics).sort_values(by='RMSE', ascending=False)
    
    FAMILY_CSV_BRAIN = os.path.join(brain_subset_dir, "compound_family_rmse_report.csv")
    safe_to_csv(family_df, FAMILY_CSV_BRAIN, index=False, encoding='utf-8-sig')
    print(f"Saved compound family report to {FAMILY_CSV_BRAIN}")

    family_table_rows = ""
    for idx, (_, row) in enumerate(family_df.head(15).iterrows(), 1):
        r2_str = f"{row['R2']:.4f}" if not pd.isna(row['R2']) else "N/A"
        family_table_rows += (
            f"| {idx} | `{row['compound_family']}` | {int(row['test_rows'])} | "
            f"{row['lab_mean']:.2f} | {row['pred_mean']:.2f} | "
            f"{row['bias_pred_minus_lab']:.4f} | {row['MAE']:.4f} | "
            f"{row['RMSE']:.4f} | {r2_str} |\n"
        )

    test_metrics = {'r2': test_r2, 'mae': test_mae, 'rmse': test_rmse}
    save_model_bundle(
        MODEL_BUNDLE, best_model, preprocessor, feature_cols, selected_features,
        X_train_raw_for_guidance, X_train_model.reset_index(drop=True), y_train,
        subset_name, best_model_name, best_params, best_cv_score, test_metrics,
        feature_audit_df, guidance_df, importance_df,
        family_medians=family_medians, track_median=track_median
    )
    save_model_bundle(
        MODEL_BUNDLE_BRAIN, best_model, preprocessor, feature_cols, selected_features,
        X_train_raw_for_guidance, X_train_model.reset_index(drop=True), y_train,
        subset_name, best_model_name, best_params, best_cv_score, test_metrics,
        feature_audit_df, guidance_df, importance_df,
        family_medians=family_medians, track_median=track_median
    )
    print(f"Saved deployable model bundle to {MODEL_BUNDLE}")
        
    # 9. Generate Report
    print(f"Writing predictive modeling report to: {MODEL_REPORT_MD}")
    cv_table_rows = ""
    for k, v in cv_results.items():
        r2_val = f"{v['R2_CV']:.4f}" if not pd.isna(v['R2_CV']) else "N/A"
        mae_val = f"{v['MAE_CV']:.4f}" if not pd.isna(v['MAE_CV']) else "N/A"
        rmse_val = f"{v['RMSE_CV']:.4f}" if not pd.isna(v['RMSE_CV']) else "N/A"
        cv_table_rows += f"| {k} | {r2_val} | {mae_val} | {rmse_val} |\n"
        
    feat_importance_rows = ""
    if len(importance_df) > 0:
        for idx, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
            feat = row['Feature']
            explanation = "流变/物理参数描述"
            if "eta_torque" in feat:
                stage = feat.split('_')[0]
                explanation = f"{stage}阶段视在粘度平均值（Torque/RotorSpeed）"
            elif "Torque_Mean" in feat:
                stage = feat.split('_')[0]
                explanation = f"{stage}阶段扭矩平均值"
            elif "Torque_Std" in feat:
                stage = feat.split('_')[0]
                explanation = f"{stage}阶段扭矩波动标准差"
            elif "Torque_Integral" in feat:
                stage = feat.split('_')[0]
                explanation = f"{stage}阶段扭矩累积积分（功历史）"
            elif "RotorSpeed_Mean" in feat:
                stage = feat.split('_')[0]
                explanation = f"{stage}阶段平均转子转速"
            elif "RotorSpeed_Integral" in feat:
                explanation = f"{feat.split('_')[0]}阶段剪切历史积分（转子总圈数）"
            elif "Duration" in feat:
                stage = feat.split('_')[0]
                explanation = f"{stage}阶段持续时间（秒）"
            elif "phys_" in feat:
                if "discharge_temp" in feat: explanation = "排胶温度"
                elif "temp_integral" in feat: explanation = "总混炼温度-时间积分（热历史）"
                elif "power_integral" in feat: explanation = "总混炼消耗电能积分"
                elif "eta_app" in feat: explanation = f"基于功率计算的视在粘度"
                elif "shear_history" in feat: explanation = "总混炼累积剪切圈数"
                else: explanation = "流变学计算特征"
            elif "supplier_rubber_viscosity_avg" in feat:
                explanation = "供应商生胶出厂粘度平均值"
            elif "weight_pct_" in feat:
                explanation = f"配方中该组分的重量百分比"
            feat_importance_rows += f"| {idx} | `{feat}` | {row['Importance']:.4f} | {explanation} |\n"

    guidance_rows = ""
    if len(guidance_df) > 0:
        for idx, (_, row) in enumerate(guidance_df.head(15).iterrows(), 1):
            corr_val = row['spearman_corr_to_MNY']
            corr_text = f"{corr_val:.4f}" if not pd.isna(corr_val) else "N/A"
            guidance_rows += (
                f"| {idx} | `{row['Feature']}` | {row['model_importance']:.4f} | "
                f"{corr_text} | {row['direction_hint']} | {row['process_guidance']} |\n"
            )
            
    report_content = f"""# 胶料门尼粘度 (MNY) 预测模型开发与评估报告 ({subset_name} 胶料模型)
    
本报告详细说明了针对 **{subset_name}** 胶料数据集进行门尼粘度（MNY）回归建模的评估结果。

---

## 1. 建模数据概况

- **数据集类型**：{subset_name} 胶料数据（is_oil_loading_present = {is_oil_loading_present_val}）
- **总车次行数**：{len(df_all_rows)} 车
- **异常值清理**：使用**孤立森林 (Isolation Forest) 算法**对温度、功率、扭矩和持续时间等核心曲线特征进行多维异常诊断。
- **清洗后带有门尼粘度标签的车次**：{len(df_labeled)} 车
- **特征工程预处理**：
  1. 使用中位数填充缺失值。
  2. 进行标准差标准化（StandardScaler）。
    3. 先执行物理可解释特征审计，剔除高缺失、高单一值、高共线和低优先级特征。
    4. 使用方差过滤（VarianceThreshold，阈值=0.01）移除噪声/恒定特征，过滤后剩余 {len(selected_features)} 个特征进行训练。

### 1.1 可解释特征筛选

- 候选特征数：{len(feature_audit_df)}
- 进入紧凑特征集：{len(feature_cols)}
- 最终进入模型训练：{len(selected_features)}
- 特征审计明细：`interpretable_feature_audit.csv`

---

## 2. 模型选择与交叉验证 (Model Comparison)

使用 **5折交叉验证 (5-Fold CV)** 在训练集上评估了多种回归算法。性能比较如下：

| 模型算法 | 平均 CV $R^2$ (决定系数) | 平均 CV MAE (平均绝对误差) | 平均 CV RMSE (均方根误差) |
| :--- | :---: | :---: | :---: |
{cv_table_rows}
*注：$R^2$ 越接近 1.0 预测拟合效果越好。MAE 代表门尼粘度预测值与实测值的绝对平均偏差值。*

---

## 3. Optuna 超参数优化与测试集最终表现

最优模型为 **{best_model_name}**，其最优超参数集为：
- `best_params`: `{best_params}`

### 3.1 测试集（Held-out Test Set, 20% 未参训数据）的最终表现：
- **测试集决定系数 $R^2$**：**{test_r2:.4f}**
- **测试集平均绝对误差 (MAE)**：**{test_mae:.4f}**
- **测试集均方根误差 (RMSE)**：**{test_rmse:.4f}**

### 3.2 预测效果 Parity 对齐图
![Mooney预测Parity图](file:///{PARITY_PNG.replace('\\', '/')})

---

## 4. 特征贡献度分析 (Feature Importance)

以下是经过调优后的 {best_model_name} 模型在预测门尼粘度时，排名前 15 位的核心物理特征：

| 排名 | 特征名称 | 贡献权重 (Importance) | 物理流变学解释 |
| :---: | :--- | :---: | :--- |
{feat_importance_rows}

### 4.2 核心贡献度图解
![Mooney特征重要性图](file:///{IMPORTANCE_PNG.replace('\\', '/')})

---

## 5. 工艺调整解释表

下表将模型重要性、单调相关趋势和密炼工艺含义放在一起，用于预测完成后的工艺诊断。注意：该方向代表训练数据中的统计趋势，实际调整必须结合胶料体系、配方窗口和现场约束确认。

| 排名 | 特征名称 | 模型重要性 | Spearman趋势 | 趋势解释 | 工艺含义/调整方向 |
| :---: | :--- | :---: | :---: | :--- | :--- |
{guidance_rows}

完整明细见：`process_adjustment_guidance.csv`

"""
    report_content += f"""
---

## 6. 各配方系列 (Compound Family) 的预测表现

以下是测试集中前 15 个主要胶料配方系列的表现统计（按 RMSE 降序排列）：

| 排名 | 配方系列 | 测试车次 | 真实门尼均值 | 预测门尼均值 | 偏差 (Bias) | MAE | RMSE | $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{family_table_rows}
*注：对于仅有1个测试样本或方差为0的系列，$R^2$ 显示为 N/A。*
"""
    with open(MODEL_REPORT_MD, 'w', encoding='utf-8') as f:
        f.write(report_content)
    with open(MODEL_REPORT_MD_RESULTS, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"=== MOONEY PREDICTIVE MODELING REPORT SUCCESSFULLY GENERATED AT {MODEL_REPORT_MD} ===")
    
    # Sync from brain_subset_dir to results_dir
    print("\n--- Copying all reports and artifacts from Brain directory to workspace results folder ---")
    import shutil
    for filename in os.listdir(brain_subset_dir):
        if filename.endswith(('.md', '.png', '.csv', '.joblib')) and not filename.startswith('.'):
            src_path = os.path.join(brain_subset_dir, filename)
            dst_path = os.path.join(results_dir, filename)
            try:
                shutil.copy2(src_path, dst_path)
                print(f"  Copied: {filename:<45} -> {results_dir}/")
            except Exception as e:
                print(f"  Failed to copy {filename}: {e}")


def remove_mny_anomalies(df):
    """
    Remove MNY anomalies from the dataset:
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
    
    print(f"\n[MNY Data Cleaning Summary] Cleaned {len(all_drop_indices)} anomalous MNY rows. Labeled rows reduced from {initial_count} to {len(df_clean)}.\n")
    return df_clean


def main():
    print("=== STARTING SEPARATED (WITH-OIL / WITHOUT-OIL) PREDICTIVE MODELING PIPELINE ===")
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} does not exist. Please run the curve segmenter first.")
        return
        
    print(f"Loading feature dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Loaded {len(df)} total rows.")
    
    # Filter only for M1 and R1 compounds (first-stage mixes)
    df = df[
        df['CompoundName'].str.startswith('M1-', na=False) |
        df['CompoundName'].str.startswith('m1-', na=False) |
        df['CompoundName'].str.startswith('R1-', na=False) |
        df['CompoundName'].str.startswith('r1-', na=False)
    ].copy()
    print(f"Filtered to M1/R1 base compounds: {len(df)} rows.")
    
    # Filter for labeled rows (batches with MNY results)
    df_labeled = df.dropna(subset=['MNY']).copy()
    print(f"Labeled batches (with MNY test results) before cleaning: {len(df_labeled)}")
    
    # Apply MNY anomaly cleaning
    df_labeled = remove_mny_anomalies(df_labeled)
    print(f"Labeled batches (with MNY test results) after cleaning: {len(df_labeled)}")
    
    # Compute rolling MNY features chronologically on labeled subset
    df_labeled = compute_rolling_features(df_labeled)
    
    # Add silica system flag: silica_phr >= 25.0 and weight_pct_silian > 0.0
    df_labeled['is_silica_system'] = ((df_labeled['silica_phr'] >= 25.0) & (df_labeled['weight_pct_silian'] > 0.0)).astype(float)
    df['is_silica_system'] = ((df['silica_phr'] >= 25.0) & (df['weight_pct_silian'] > 0.0)).astype(float)
    
    sub_tracks = [
        ("With-Oil High-Silica", 1.0, 1.0, "results_with_oil_high_silica"),
        ("With-Oil Carbon-Black", 1.0, 0.0, "results_with_oil_carbon_black"),
        ("Without-Oil High-Silica", 0.0, 1.0, "results_without_oil_high_silica"),
        ("Without-Oil Carbon-Black", 0.0, 0.0, "results_without_oil_carbon_black")
    ]
    
    for track_name, is_oil, is_silica, folder_name in sub_tracks:
        df_sub_labeled = df_labeled[(df_labeled['is_oil_loading_present'] == is_oil) & (df_labeled['is_silica_system'] == is_silica)].copy()
        df_sub_all = df[(df['is_oil_loading_present'] == is_oil) & (df['is_silica_system'] == is_silica)].copy()
        
        print(f"\n=======================================================")
        print(f"Running modeling pipeline for sub-track: {track_name}...")
        print(f"Labeled: {len(df_sub_labeled)} | Total rows: {len(df_sub_all)}")
        print(f"=======================================================")
        
        # Fallback routing check: if sub-track has less than 120 samples, fallback to general oil/no-oil track
        if len(df_sub_labeled) < 120:
            print(f"  -> [Warning] Too few sub-track samples ({len(df_sub_labeled)} < 120). Falling back to parent track.")
            df_sub_labeled = df_labeled[df_labeled['is_oil_loading_present'] == is_oil].copy()
            df_sub_all = df[df['is_oil_loading_present'] == is_oil].copy()
            track_name = "With-Oil Fallback" if is_oil == 1.0 else "Without-Oil Fallback"
            
        run_modeling_pipeline(
            df_sub_labeled, 
            df_sub_all, 
            is_oil, 
            track_name, 
            os.path.join(PARENT_DIR, "models", folder_name)
        )
        
    print("\n=== 4-TRACK MOONEY PREDICTIVE MODELING PIPELINE RUNS COMPLETED ===")

if __name__ == '__main__':
    main()
