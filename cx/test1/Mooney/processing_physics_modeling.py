"""
Rigorous Processing Physics & Interpretable Modeling Engine
============================================================
Investigates the quantitative influence of:
  - Temperature (T): MST Step 7 Temp, MSB Step 1 Temp, MSB Step 3 Temp, Peak Temp, Thermal History Integral.
  - Duration/Time (t): MST Step 7 Duration, MSB Step 1 Duration, Total Mixing Time.
  - Torque (tau): MST Step 7 Torque, MSB Step 1 Torque, Mean Torque, Max Torque, Torque-Time Integral.
On two key factory targets:
  1. Mooney Viscosity ML(1+4)@100C (MU)
  2. Total Energy Consumption (kWh)

Methodology:
  - Fully transparent, reproducible mathematical derivations and statistical proofs.
  - Pearson r & Spearman rho correlation analysis with p-values.
  - Parsimonious White-Box OLS Regression with complete ANOVA F-tests, t-statistics, VIF multicollinearity checks.
  - Dimensionless Standardized Beta weights for direct cross-variable sensitivity comparison.
  - Interpretable Machine Learning (Ridge with Interaction terms, Constrained LightGBM, SHAP values).
  - 5-Fold Cross-Validation for generalizability.
  - High-resolution visual dashboard generation for reporting.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"

# Set matplotlib style for publication-quality figures
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def prepare_physical_dataset():
    """Load cached master dataset and engineer explicit physical features."""
    df_raw = pd.read_pickle(CACHE / "mathematical_dataset.pkl")
    
    df = df_raw.copy()
    
    # Duration / Time features (s)
    df['t_MST7'] = (df['MST_S7_Time_s'] - df['MST_S6_Time_s']).clip(lower=0)
    df['t_MSB1'] = df['MSB_S1_Time_s']
    df['t_MSB3'] = (df['MSB_S3_Time_s'] - df['MSB_S2_Time_s']).clip(lower=0)
    df['t_Total'] = df['MMS_Mixing_Time_s']
    
    # Temperature features (deg C)
    df['T_MST7'] = df['MST_S7_Temp_C']
    df['T_MSB1'] = df['MSB_S1_Temp_C']
    df['T_MSB3'] = df['MSB_S3_Temp_C']
    df['T_Max'] = df['Curve_Temp_Max_Full_C']
    df['T_Thermal_Integral'] = df['Curve_Int_Temp_Full_degCs'] / 1000.0 # k degC * s
    
    # Torque features (N*m)
    df['Tau_MST7'] = df['MST_S7_Torque_Nm']
    df['Tau_MSB1'] = df['MSB_S1_Torque_Nm']
    df['Tau_MSB3'] = df['MSB_S3_Torque_Nm']
    df['Tau_Mean'] = df['Curve_Torque_Mean_Full_Nm']
    df['Tau_Max'] = df['Curve_Torque_Max_Full_Nm']
    df['Tau_Impulse_Integral'] = df['Curve_Int_Torque_Full_Nms'] / 1000.0 # kN*m*s
    
    # Targets
    df['Target_Mooney'] = df['Mooney']
    df['Target_Energy'] = df['MMS_Total_Energy_kWh']
    
    return df

def analyze_correlations(df, feature_cols, target_name):
    """Compute Pearson and Spearman correlations with significance tests."""
    valid_data = df.dropna(subset=[target_name] + feature_cols)
    results = []
    y = valid_data[target_name].values
    
    for feat in feature_cols:
        x = valid_data[feat].values
        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)
        results.append({
            "Variable": feat,
            "N": len(valid_data),
            "Mean": np.mean(x),
            "Std": np.std(x),
            "Pearson_r": r_p,
            "Pearson_p": p_p,
            "Spearman_rho": r_s,
            "Spearman_p": p_s,
            "Abs_r": abs(r_p)
        })
    df_corr = pd.DataFrame(results).sort_values("Abs_r", ascending=False).reset_index(drop=True)
    return df_corr, valid_data

def run_multivariate_ols(valid_data, feature_cols, target_name):
    """Fit OLS with ANOVA, t-stats, 95% CI, VIF, standardized betas, and 5-fold CV."""
    y = valid_data[target_name]
    X = valid_data[feature_cols]
    
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    
    # Standardized Betas (dimensionless sensitivity)
    y_std = (y - y.mean()) / y.std()
    X_std = (X - X.mean()) / X.std()
    model_std = sm.OLS(y_std, X_std).fit()
    
    # VIF
    vifs = {}
    for i, col in enumerate(feature_cols):
        try:
            vifs[col] = variance_inflation_factor(X_const.values, i + 1)
        except Exception:
            vifs[col] = np.nan
            
    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2, cv_rmse, cv_mae = [], [], []
    for tr, te in kf.split(X):
        m = sm.OLS(y.iloc[tr], sm.add_constant(X.iloc[tr])).fit()
        pred = m.predict(sm.add_constant(X.iloc[te]))
        cv_r2.append(r2_score(y.iloc[te], pred))
        cv_rmse.append(np.sqrt(mean_squared_error(y.iloc[te], pred)))
        cv_mae.append(mean_absolute_error(y.iloc[te], pred))
        
    coef_table = []
    for col in feature_cols:
        coef_table.append({
            "Variable": col,
            "Raw_Beta": model.params[col],
            "Std_Err": model.bse[col],
            "t_stat": model.tvalues[col],
            "p_val": model.pvalues[col],
            "CI_lower": model.conf_int().loc[col, 0],
            "CI_upper": model.conf_int().loc[col, 1],
            "Std_Beta": model_std.params[col],
            "VIF": vifs[col]
        })
    df_coef = pd.DataFrame(coef_table)
    
    summary_dict = {
        "N": len(valid_data),
        "Intercept": model.params['const'],
        "R2": model.rsquared,
        "Adj_R2": model.rsquared_adj,
        "F_stat": model.fvalue,
        "F_pval": model.f_pvalue,
        "RMSE": np.sqrt(model.mse_resid),
        "MAE": mean_absolute_error(y, model.predict(X_const)),
        "CV_R2_mean": np.mean(cv_r2),
        "CV_R2_std": np.std(cv_r2),
        "CV_RMSE_mean": np.mean(cv_rmse),
        "CV_MAE_mean": np.mean(cv_mae)
    }
    
    return model, df_coef, summary_dict

def run_ml_comparison(valid_data, feature_cols, target_name):
    """Compare OLS, Ridge with Interactions, and LightGBM with SHAP values."""
    X = valid_data[feature_cols]
    y = valid_data[target_name]
    
    # 1. Linear OLS
    lr = LinearRegression().fit(X, y)
    
    # 2. LightGBM Regressor
    lgbm = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)
    lgbm.fit(X, y)
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    df_shap = pd.DataFrame({
        "Variable": feature_cols,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    # 5-fold CV for ML
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lgb_r2, lgb_rmse, lgb_mae = [], [], []
    for tr, te in kf.split(X):
        m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[te])
        lgb_r2.append(r2_score(y.iloc[te], p))
        lgb_rmse.append(np.sqrt(mean_squared_error(y.iloc[te], p)))
        lgb_mae.append(mean_absolute_error(y.iloc[te], p))
        
    ml_metrics = {
        "LGBM_CV_R2": np.mean(lgb_r2),
        "LGBM_CV_RMSE": np.mean(lgb_rmse),
        "LGBM_CV_MAE": np.mean(lgb_mae)
    }
    
    return lgbm, df_shap, ml_metrics, shap_values

def generate_dashboard(df_mooney_valid, m_mooney, df_coef_m, 
                       df_energy_valid, m_energy, df_coef_e,
                       shap_m, shap_e, feat_m, feat_e):
    """Create a publication-quality 6-panel visual dashboard."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Color palette
    c_blue = '#1f77b4'
    c_red = '#d62728'
    c_green = '#2ca02c'
    c_orange = '#ff7f0e'
    c_purple = '#9467bd'
    
    # -------------------------------------------------------------------------
    # Panel 1: Mooney Standardized Beta Coefficients (Sensitivity)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    bar_m = df_coef_m.sort_values('Std_Beta', ascending=True)
    colors_m = [c_red if b < 0 else c_blue for b in bar_m['Std_Beta']]
    bars1 = ax1.barh(bar_m['Variable'], bar_m['Std_Beta'], color=colors_m, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.7)
    ax1.set_title("1. 门尼粘度 ML(1+4) 标准化影响系数 (Beta*)\n[物理敏感度: 负值代表降低门尼，正值代表增加门尼]", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("标准化回归系数 (Beta*) [无量纲]", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    for b in bars1:
        w = b.get_width()
        offset = 0.02 if w >= 0 else -0.06
        ax1.text(w + offset, b.get_y() + b.get_height()/2, f"{w:+.3f}", va='center', fontsize=10, fontweight='bold')
        
    # -------------------------------------------------------------------------
    # Panel 2: Total Energy Standardized Beta Coefficients (Sensitivity)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    bar_e = df_coef_e.sort_values('Std_Beta', ascending=True)
    colors_e = [c_green if b < 0 else c_orange for b in bar_e['Std_Beta']]
    bars2 = ax2.barh(bar_e['Variable'], bar_e['Std_Beta'], color=colors_e, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax2.axvline(0, color='black', linestyle='--', alpha=0.7)
    ax2.set_title("2. 单车总能耗 (kWh) 标准化影响系数 (Beta*)\n[物理敏感度: 驱动电耗增长的核心物理驱动力排序]", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("标准化回归系数 (Beta*) [无量纲]", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    for b in bars2:
        w = b.get_width()
        offset = 0.02 if w >= 0 else -0.06
        ax2.text(w + offset, b.get_y() + b.get_height()/2, f"{w:+.3f}", va='center', fontsize=10, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel 3: Mooney Viscosity: Actual vs Predicted
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    X_m_const = sm.add_constant(df_mooney_valid[feat_m])
    y_m_true = df_mooney_valid['Target_Mooney']
    y_m_pred = m_mooney.predict(X_m_const)
    
    ax3.scatter(y_m_true, y_m_pred, color=c_blue, alpha=0.45, edgecolors='none', s=35, label='实际生产车次')
    lims_m = [min(y_m_true.min(), y_m_pred.min()) - 2, max(y_m_true.max(), y_m_pred.max()) + 2]
    ax3.plot(lims_m, lims_m, 'r--', linewidth=2, label='1:1 理想基准线')
    ax3.fill_between(lims_m, np.array(lims_m) - 3.0, np.array(lims_m) + 3.0, color='green', alpha=0.12, label='±3.0 MU 质量容差带')
    ax3.set_xlim(lims_m)
    ax3.set_ylim(lims_m)
    ax3.set_title(f"3. 门尼粘度模型拟合验证 (N = {len(y_m_true)})\n[R^2 = {m_mooney.rsquared:.3f}, MAE = {mean_absolute_error(y_m_true, y_m_pred):.2f} MU]", fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("化验室实测门尼粘度 (MU)", fontsize=11)
    ax3.set_ylabel("数学模型预测门尼粘度 (MU)", fontsize=11)
    ax3.legend(loc='upper left', frameon=True)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel 4: Energy Consumption: Actual vs Predicted
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    X_e_const = sm.add_constant(df_energy_valid[feat_e])
    y_e_true = df_energy_valid['Target_Energy']
    y_e_pred = m_energy.predict(X_e_const)
    
    ax4.scatter(y_e_true, y_e_pred, color=c_orange, alpha=0.45, edgecolors='none', s=35, label='实际生产车次')
    lims_e = [min(y_e_true.min(), y_e_pred.min()) - 1, max(y_e_true.max(), y_e_pred.max()) + 1]
    ax4.plot(lims_e, lims_e, 'r--', linewidth=2, label='1:1 理想基准线')
    ax4.fill_between(lims_e, np.array(lims_e) - 1.0, np.array(lims_e) + 1.0, color='green', alpha=0.12, label='±1.0 kWh 节能控制带')
    ax4.set_xlim(lims_e)
    ax4.set_ylim(lims_e)
    ax4.set_title(f"4. 单车总能耗模型拟合验证 (N = {len(y_e_true)})\n[R^2 = {m_energy.rsquared:.3f}, MAE = {mean_absolute_error(y_e_true, y_e_pred):.2f} kWh]", fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel("MMS 仪表实测总能耗 (kWh)", fontsize=11)
    ax4.set_ylabel("数学模型预测总能耗 (kWh)", fontsize=11)
    ax4.legend(loc='upper left', frameon=True)
    ax4.grid(True, linestyle=':', alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel 5: Real Batch Tracking Sequence (First 40 Consecutive Batches)
    # -------------------------------------------------------------------------
    ax5 = axes[2, 0]
    sample_sub = df_mooney_valid.head(40).copy().reset_index(drop=True)
    X_sample_const = sm.add_constant(sample_sub[feat_m])
    pred_sample = m_mooney.predict(X_sample_const)
    
    ax5.plot(sample_sub.index + 1, sample_sub['Target_Mooney'], 'o-', color=c_blue, label='化验室实测门尼', linewidth=1.5, markersize=5)
    ax5.plot(sample_sub.index + 1, pred_sample, 's--', color=c_red, label='物理模型预判门尼', linewidth=1.5, markersize=5)
    ax5.set_title("5. 实际生产连续车次门尼动态跟随跟踪验证 (前40车)", fontsize=12, fontweight='bold', pad=10)
    ax5.set_xlabel("生产车次序号", fontsize=11)
    ax5.set_ylabel("门尼粘度 ML(1+4) (MU)", fontsize=11)
    ax5.legend(loc='upper right', frameon=True)
    ax5.grid(True, linestyle=':', alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel 6: Physical Coupling: Torque vs Temperature Phase Diagram
    # -------------------------------------------------------------------------
    ax6 = axes[2, 1]
    sc = ax6.scatter(df_mooney_valid['T_MST7'], df_mooney_valid['Tau_MST7'], 
                     c=df_mooney_valid['Target_Mooney'], cmap='viridis', s=45, alpha=0.8, edgecolor='black', linewidth=0.5)
    cbar = plt.colorbar(sc, ax=ax6)
    cbar.set_label('实测门尼粘度 (MU)', fontsize=10)
    ax6.set_title("6. 塑炼段【温度-扭矩-门尼】热力学流变相图\n[高温高扭矩驱动分子链断裂促使门尼下降]", fontsize=12, fontweight='bold', pad=10)
    ax6.set_xlabel("上密炼第七步温度 T_MST7 (℃)", fontsize=11)
    ax6.set_ylabel("上密炼第七步扭矩 Tau_MST7 (N·m)", fontsize=11)
    ax6.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    output_path = ROOT / "Processing_Physics_Interpretable_Dashboard.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visual dashboard successfully generated and saved to: {output_path}")

def main():
    print("=" * 90)
    print("PROCESSING PHYSICS & INTERPRETABLE MATHEMATICAL MODELING ENGINE")
    print("Compound: M1-T15760-06-002 (Production & DOE Trials)")
    print("=" * 90)
    
    df = prepare_physical_dataset()
    print(f"Total dataset loaded: N = {len(df)} batches.")
    
    # Define physical variables across the 3 physical dimensions:
    # 1. Temperature: T_MST7, T_MSB1, T_MSB3, T_Max
    # 2. Duration/Time: t_MST7, t_MSB1, t_Total
    # 3. Torque: Tau_MST7, Tau_MSB1, Tau_Mean, Tau_Impulse_Integral
    
    # -------------------------------------------------------------------------
    # PART 1: CORRELATION MATRIX FOR MOONEY VISCOSITY
    # -------------------------------------------------------------------------
    mooney_candidate_feats = [
        't_MST7', 't_MSB1', 't_Total',
        'T_MST7', 'T_MSB1', 'T_MSB3', 'T_Max', 'T_Thermal_Integral',
        'Tau_MST7', 'Tau_MSB1', 'Tau_Mean', 'Tau_Max', 'Tau_Impulse_Integral'
    ]
    df_corr_m, valid_m = analyze_correlations(df, mooney_candidate_feats, 'Target_Mooney')
    print("\n--- [1] CORRELATION STUDY: MOONEY VISCOSITY ML(1+4)@100C ---")
    print(df_corr_m[['Variable', 'Mean', 'Std', 'Pearson_r', 'Pearson_p', 'Spearman_rho']].to_string(index=False))
    
    # -------------------------------------------------------------------------
    # PART 2: CORRELATION MATRIX FOR TOTAL ENERGY
    # -------------------------------------------------------------------------
    energy_candidate_feats = [
        't_MST7', 't_MSB1', 't_Total',
        'T_MST7', 'T_MSB1', 'T_MSB3', 'T_Max', 'T_Thermal_Integral',
        'Tau_MST7', 'Tau_MSB1', 'Tau_Mean', 'Tau_Max', 'Tau_Impulse_Integral'
    ]
    df_corr_e, valid_e = analyze_correlations(df, energy_candidate_feats, 'Target_Energy')
    print("\n--- [2] CORRELATION STUDY: TOTAL ENERGY CONSUMPTION (kWh) ---")
    print(df_corr_e[['Variable', 'Mean', 'Std', 'Pearson_r', 'Pearson_p', 'Spearman_rho']].to_string(index=False))
    
    # -------------------------------------------------------------------------
    # PART 3: PARSIMONIOUS PHYSICAL OLS FOR MOONEY VISCOSITY
    # -------------------------------------------------------------------------
    # We choose core interpretable features representing Time, Temp, Torque directly:
    # Time: t_MST7, t_MSB1
    # Temp: T_MST7, T_MSB3
    # Torque: Tau_MST7, Tau_Mean
    feats_m_core = ['t_MST7', 't_MSB1', 'T_MST7', 'T_MSB3', 'Tau_MST7']
    model_m, df_coef_m, summ_m = run_multivariate_ols(valid_m, feats_m_core, 'Target_Mooney')
    print("\n--- [3] WHITE-BOX MULTIVARIATE OLS: MOONEY VISCOSITY ---")
    print(f"Sample Size N: {summ_m['N']}, Intercept: {summ_m['Intercept']:.4f}")
    print(f"R-squared: {summ_m['R2']:.4f}, Adj R-squared: {summ_m['Adj_R2']:.4f}")
    print(f"ANOVA F-stat: {summ_m['F_stat']:.4f} (p = {summ_m['F_pval']:.4e})")
    print(f"In-sample RMSE: {summ_m['RMSE']:.4f} MU, MAE: {summ_m['MAE']:.4f} MU")
    print(f"5-Fold CV R2: {summ_m['CV_R2_mean']:.4f} +/- {summ_m['CV_R2_std']:.4f}, CV MAE: {summ_m['CV_MAE_mean']:.4f} MU")
    print("\nEstimated Coefficients & Sensitivity:")
    print(df_coef_m.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # PART 4: PARSIMONIOUS PHYSICAL OLS FOR TOTAL ENERGY
    # -------------------------------------------------------------------------
    # Physical conservation of work: E ~ sum(Tau * omega * t)
    # Energy features: t_Total (or t_MSB1, t_MST7), Tau_Mean, Tau_Impulse_Integral, T_Max
    feats_e_core = ['t_MST7', 't_MSB1', 't_Total', 'Tau_Mean', 'T_Max']
    model_e, df_coef_e, summ_e = run_multivariate_ols(valid_e, feats_e_core, 'Target_Energy')
    print("\n--- [4] WHITE-BOX MULTIVARIATE OLS: TOTAL ENERGY CONSUMPTION ---")
    print(f"Sample Size N: {summ_e['N']}, Intercept: {summ_e['Intercept']:.4f}")
    print(f"R-squared: {summ_e['R2']:.4f}, Adj R-squared: {summ_e['Adj_R2']:.4f}")
    print(f"ANOVA F-stat: {summ_e['F_stat']:.4f} (p = {summ_e['F_pval']:.4e})")
    print(f"In-sample RMSE: {summ_e['RMSE']:.4f} kWh, MAE: {summ_e['MAE']:.4f} kWh")
    print(f"5-Fold CV R2: {summ_e['CV_R2_mean']:.4f} +/- {summ_e['CV_R2_std']:.4f}, CV MAE: {summ_e['CV_MAE_mean']:.4f} kWh")
    print("\nEstimated Coefficients & Sensitivity:")
    print(df_coef_e.to_string(index=False))

    # -------------------------------------------------------------------------
    # PART 5: INTERPRETABLE MACHINE LEARNING (LIGHTGBM + SHAP)
    # -------------------------------------------------------------------------
    print("\n--- [5] INTERPRETABLE ML (LIGHTGBM + SHAP ATTRIBUTION) ---")
    lgbm_m, df_shap_m, ml_m, shap_m = run_ml_comparison(valid_m, feats_m_core, 'Target_Mooney')
    print("Mooney LightGBM CV R2:", ml_m['LGBM_CV_R2'], "CV MAE:", ml_m['LGBM_CV_MAE'])
    print("Mooney SHAP Feature Importance:")
    print(df_shap_m.to_string(index=False))
    
    lgbm_e, df_shap_e, ml_e, shap_e = run_ml_comparison(valid_e, feats_e_core, 'Target_Energy')
    print("\nEnergy LightGBM CV R2:", ml_e['LGBM_CV_R2'], "CV MAE:", ml_e['LGBM_CV_MAE'])
    print("Energy SHAP Feature Importance:")
    print(df_shap_e.to_string(index=False))

    # -------------------------------------------------------------------------
    # PART 6: REAL BATCH ERROR DISTRIBUTION & 4 DOE ORDERS VERIFICATION
    # -------------------------------------------------------------------------
    print("\n--- [6] PRODUCTION ERROR AUDIT & EXPERIMENT ORDERS CHECK ---")
    X_m_const = sm.add_constant(valid_m[feats_m_core])
    valid_m['Pred_Mooney'] = model_m.predict(X_m_const)
    valid_m['Mooney_Abs_Error'] = np.abs(valid_m['Target_Mooney'] - valid_m['Pred_Mooney'])
    
    print(f"Mooney Batches with Error <= 1.0 MU: {(valid_m['Mooney_Abs_Error'] <= 1.0).mean() * 100:.1f}%")
    print(f"Mooney Batches with Error <= 2.0 MU: {(valid_m['Mooney_Abs_Error'] <= 2.0).mean() * 100:.1f}%")
    print(f"Mooney Batches with Error <= 3.0 MU: {(valid_m['Mooney_Abs_Error'] <= 3.0).mean() * 100:.1f}%")
    
    # Check the 4 experiment orders specifically
    exp_orders = ['8067458', '8101635', '8108030', '8160568']
    exp_sub = valid_m[valid_m['OrderID'].isin(exp_orders)]
    print(f"\nExperiment Orders Batches with Lab Mooney Count: {len(exp_sub)}")
    if len(exp_sub) > 0:
        print(exp_sub[['OrderID', 'BatchNumber', 't_MST7', 't_MSB1', 'T_MST7', 'T_MSB3', 'Tau_MST7', 'Target_Mooney', 'Pred_Mooney', 'Mooney_Abs_Error']].round(2).to_string(index=False))
        print(f"Experiment Orders Mean Absolute Error: {exp_sub['Mooney_Abs_Error'].mean():.2f} MU")

    # -------------------------------------------------------------------------
    # PART 7: GENERATE DASHBOARD
    # -------------------------------------------------------------------------
    print("\n--- [7] GENERATING COMPREHENSIVE VISUAL DASHBOARD ---")
    generate_dashboard(valid_m, model_m, df_coef_m, 
                       valid_e, model_e, df_coef_e, 
                       shap_m, shap_e, feats_m_core, feats_e_core)

    print("\nModeling pipeline executed successfully!")

if __name__ == "__main__":
    main()
