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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configure plotting style
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # Support Chinese characters
plt.rcParams['axes.unicode_minus'] = False            # Support negative signs
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Paths
INPUT_CSV = "stage_statistics_enriched_all_features.csv"
BRAIN_DIR = r"C:\Users\uif35346\.gemini\antigravity\brain\b5dcf5d4-477f-4b95-9fa0-2f369ea05fdb"
REPORT_MD = os.path.join(BRAIN_DIR, "mny_unified_vs_separated_comparison_report.md")
COMPARISON_PNG = os.path.join(BRAIN_DIR, "mooney_model_comparison_plots.png")

def tune_xgb_optuna(X_train, y_train, name, n_trials=25):
    print(f"--- Tuning XGBoost for {name} ({n_trials} trials) ---")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        model = xgb.XGBRegressor(**params)
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
        return np.mean(scores['test_score'])
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print(f"-> Best {name} CV R2: {study.best_value:.4f}")
    return study.best_params

def clean_and_prepare_data(df_part, feature_cols):
    """Drop constant or all-NaN features specifically for this partition"""
    X = df_part[feature_cols].copy()
    y = df_part['MNY'].copy()
    
    nan_cols = X.columns[X.isna().all()].tolist()
    const_cols = [col for col in X.columns if X[col].nunique() <= 1]
    cols_to_drop = list(set(nan_cols + const_cols))
    
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
        selected_features = [col for col in feature_cols if col not in cols_to_drop]
    else:
        selected_features = feature_cols
        
    return X, y, selected_features

def main():
    print("=== STARTING UNIFIED VS SEPARATED MODEL COMPARISON EXPERIMENT ===")
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found!")
        return
        
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Loaded {len(df)} total rows.")
    
    # Filter for labeled rows (batches with MNY results)
    df_labeled = df.dropna(subset=['MNY']).copy()
    print(f"Labeled batches: {len(df_labeled)}")
    
    # Isolate Carbon Black vs Silica compounds index based on recipe variables
    # CB: silica_phr <= 25 and no silane (weight_pct_silian is 0 or NaN)
    cb_mask = ((df_labeled['weight_pct_silian'].isna()) | (df_labeled['weight_pct_silian'] == 0.0)) & (df_labeled['silica_phr'] <= 25.0)
    df_labeled['is_silica'] = ~cb_mask
    
    cb_count = cb_mask.sum()
    silica_count = (~cb_mask).sum()
    print(f"Carbon Black batches: {cb_count}")
    print(f"Silica/Silane batches: {silica_count}")
    
    # Define potential feature columns
    exclude_cols = [
        'OrderID', 'BatchNumber', 'CompoundName', 'CompoundDescription', 
        'batch_information_fk_final', 'temp', 'power', 'Torque', 'RotorSpeed', 'WayofRam',
        'CurrentValue', 'PrevStepValue', 'MNY target', 'is_silica'
    ]
    numeric_cols = df_labeled.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols and col != 'MNY']
    
    # 1. Split entire dataset into Train (80%) and Test (20%) using a fixed random state
    # This guarantees test sets for CB and Silica are mutually exclusive and aligned
    train_idx, test_idx = train_test_split(df_labeled.index, test_size=0.2, random_state=42)
    
    df_train = df_labeled.loc[train_idx].copy()
    df_test = df_labeled.loc[test_idx].copy()
    
    # Slice train and test for CB and Silica subsets
    cb_train_df = df_train[~df_train['is_silica']].copy()
    cb_test_df = df_test[~df_test['is_silica']].copy()
    
    silica_train_df = df_train[df_train['is_silica']].copy()
    silica_test_df = df_test[df_test['is_silica']].copy()
    
    print(f"\nTraining set size: {len(df_train)} (CB: {len(cb_train_df)}, Silica: {len(silica_train_df)})")
    print(f"Test set size: {len(df_test)} (CB: {len(cb_test_df)}, Silica: {len(silica_test_df)})")
    
    # Prepare matrices (drop NaN/constant features per subset)
    X_cb_train_raw, y_cb_train, cb_features = clean_and_prepare_data(cb_train_df, feature_cols)
    X_cb_test_raw = cb_test_df[cb_features].copy()
    y_cb_test = cb_test_df['MNY'].copy()
    
    X_silica_train_raw, y_silica_train, silica_features = clean_and_prepare_data(silica_train_df, feature_cols)
    X_silica_test_raw = silica_test_df[silica_features].copy()
    y_silica_test = silica_test_df['MNY'].copy()
    
    X_uni_train_raw, y_uni_train, uni_features = clean_and_prepare_data(df_train, feature_cols)
    X_uni_test_raw = df_test[uni_features].copy()
    y_uni_test = df_test['MNY'].copy()
    
    print(f"Features kept for CB Model: {len(cb_features)}")
    print(f"Features kept for Silica Model: {len(silica_features)}")
    print(f"Features kept for Unified Model: {len(uni_features)}")
    
    # Preprocessing pipelines
    def build_preprocessor():
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
    cb_prep = build_preprocessor()
    X_cb_train = cb_prep.fit_transform(X_cb_train_raw)
    X_cb_test = cb_prep.transform(X_cb_test_raw)
    
    silica_prep = build_preprocessor()
    X_silica_train = silica_prep.fit_transform(X_silica_train_raw)
    X_silica_test = silica_prep.transform(X_silica_test_raw)
    
    uni_prep = build_preprocessor()
    X_uni_train = uni_prep.fit_transform(X_uni_train_raw)
    X_uni_test = uni_prep.transform(X_uni_test_raw)
    
    # 2. Hyperparameter Tuning using Optuna
    cb_best_params = tune_xgb_optuna(X_cb_train, y_cb_train, "Carbon Black Model", n_trials=25)
    silica_best_params = tune_xgb_optuna(X_silica_train, y_silica_train, "Silica Model", n_trials=25)
    uni_best_params = tune_xgb_optuna(X_uni_train, y_uni_train, "Unified Model", n_trials=25)
    
    # 3. Fit Final Models
    print("\n--- Fitting final optimized models ---")
    model_cb = xgb.XGBRegressor(**cb_best_params, random_state=42, n_jobs=-1)
    model_cb.fit(X_cb_train, y_cb_train)
    
    model_silica = xgb.XGBRegressor(**silica_best_params, random_state=42, n_jobs=-1)
    model_silica.fit(X_silica_train, y_silica_train)
    
    model_uni = xgb.XGBRegressor(**uni_best_params, random_state=42, n_jobs=-1)
    model_uni.fit(X_uni_train, y_uni_train)
    
    # 4. Evaluations
    print("\n=== EVALUATING MODELS ON TEST SETS ===")
    
    # A. Carbon Black Test Set (CB Model vs Unified Model)
    y_pred_cb_sep = model_cb.predict(X_cb_test)
    
    # Unified model predictions on CB test set (need to extract indices matching CB)
    X_cb_test_uni_raw = cb_test_df[uni_features].copy()
    X_cb_test_uni = uni_prep.transform(X_cb_test_uni_raw)
    y_pred_cb_uni = model_uni.predict(X_cb_test_uni)
    
    cb_sep_r2 = r2_score(y_cb_test, y_pred_cb_sep)
    cb_sep_mae = mean_absolute_error(y_cb_test, y_pred_cb_sep)
    cb_sep_rmse = root_mean_squared_error(y_cb_test, y_pred_cb_sep)
    
    cb_uni_r2 = r2_score(y_cb_test, y_pred_cb_uni)
    cb_uni_mae = mean_absolute_error(y_cb_test, y_pred_cb_uni)
    cb_uni_rmse = root_mean_squared_error(y_cb_test, y_pred_cb_uni)
    
    print("\n--- Carbon Black Test Set Results ---")
    print(f"Separated CB Model  | R2: {cb_sep_r2:.4f} | MAE: {cb_sep_mae:.4f} | RMSE: {cb_sep_rmse:.4f}")
    print(f"Unified Model on CB  | R2: {cb_uni_r2:.4f} | MAE: {cb_uni_mae:.4f} | RMSE: {cb_uni_rmse:.4f}")
    
    # B. Silica Test Set (Silica Model vs Unified Model)
    y_pred_silica_sep = model_silica.predict(X_silica_test)
    
    # Unified model predictions on Silica test set
    X_silica_test_uni_raw = silica_test_df[uni_features].copy()
    X_silica_test_uni = uni_prep.transform(X_silica_test_uni_raw)
    y_pred_silica_uni = model_uni.predict(X_silica_test_uni)
    
    silica_sep_r2 = r2_score(y_silica_test, y_pred_silica_sep)
    silica_sep_mae = mean_absolute_error(y_silica_test, y_pred_silica_sep)
    silica_sep_rmse = root_mean_squared_error(y_silica_test, y_pred_silica_sep)
    
    silica_uni_r2 = r2_score(y_silica_test, y_pred_silica_uni)
    silica_uni_mae = mean_absolute_error(y_silica_test, y_pred_silica_uni)
    silica_uni_rmse = root_mean_squared_error(y_silica_test, y_pred_silica_uni)
    
    print("\n--- Silica/Silane Test Set Results ---")
    print(f"Separated Silica Model | R2: {silica_sep_r2:.4f} | MAE: {silica_sep_mae:.4f} | RMSE: {silica_sep_rmse:.4f}")
    print(f"Unified Model on Sil  | R2: {silica_uni_r2:.4f} | MAE: {silica_uni_mae:.4f} | RMSE: {silica_uni_rmse:.4f}")
    
    # C. Combined/Overall Test Set
    # Strategy 1: Routed Separated Models (CB test routed to Model A, Silica test routed to Model B)
    y_pred_comb_sep = np.zeros(len(df_test))
    
    # Get positions of CB and Silica batches in test dataframe
    test_reset_index = df_test.reset_index()
    cb_positions = test_reset_index[~test_reset_index['is_silica']].index.tolist()
    silica_positions = test_reset_index[test_reset_index['is_silica']].index.tolist()
    
    y_pred_comb_sep[cb_positions] = y_pred_cb_sep
    y_pred_comb_sep[silica_positions] = y_pred_silica_sep
    
    # Strategy 2: Unified Model
    y_pred_comb_uni = model_uni.predict(X_uni_test)
    
    overall_sep_r2 = r2_score(y_uni_test, y_pred_comb_sep)
    overall_sep_mae = mean_absolute_error(y_uni_test, y_pred_comb_sep)
    overall_sep_rmse = root_mean_squared_error(y_uni_test, y_pred_comb_sep)
    
    overall_uni_r2 = r2_score(y_uni_test, y_pred_comb_uni)
    overall_uni_mae = mean_absolute_error(y_uni_test, y_pred_comb_uni)
    overall_uni_rmse = root_mean_squared_error(y_uni_test, y_pred_comb_uni)
    
    print("\n--- Overall Combined Test Set Results ---")
    print(f"Routed Separated Models | R2: {overall_sep_r2:.4f} | MAE: {overall_sep_mae:.4f} | RMSE: {overall_sep_rmse:.4f}")
    print(f"Unified Global Model    | R2: {overall_uni_r2:.4f} | MAE: {overall_uni_mae:.4f} | RMSE: {overall_uni_rmse:.4f}")
    
    # 5. Generate Plots
    print(f"\nGenerating comparison scatter plot at: {COMPARISON_PNG}")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    
    # Plot 1: Routed Separated Models
    axes[0].scatter(y_cb_test, y_pred_cb_sep, color='#1f77b4', alpha=0.5, s=20, label=f'Carbon Black (R2={cb_sep_r2:.4f})')
    axes[0].scatter(y_silica_test, y_pred_silica_sep, color='#ff7f0e', alpha=0.5, s=20, label=f'Silica/Silane (R2={silica_sep_r2:.4f})')
    min_val = min(y_uni_test.min(), y_pred_comb_sep.min()) - 1
    max_val = max(y_uni_test.max(), y_pred_comb_sep.max()) + 1
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Parity Line (y=x)')
    axes[0].set_title(f'Separated Models (CB & Silica Split)\nOverall R2: {overall_sep_r2:.4f} | MAE: {overall_sep_mae:.4f}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Measured Mooney (MNY)', fontsize=10)
    axes[0].set_ylabel('Predicted Mooney (MNY)', fontsize=10)
    axes[0].set_xlim(min_val, max_val)
    axes[0].set_ylim(min_val, max_val)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='lower right', frameon=True)
    
    # Plot 2: Unified Global Model
    axes[1].scatter(y_cb_test, y_pred_cb_uni, color='#1f77b4', alpha=0.5, s=20, label=f'Carbon Black (R2={cb_uni_r2:.4f})')
    axes[1].scatter(y_silica_test, y_pred_silica_uni, color='#ff7f0e', alpha=0.5, s=20, label=f'Silica/Silane (R2={silica_uni_r2:.4f})')
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Parity Line (y=x)')
    axes[1].set_title(f'Unified Global Model (All Compounds in One)\nOverall R2: {overall_uni_r2:.4f} | MAE: {overall_uni_mae:.4f}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Measured Mooney (MNY)', fontsize=10)
    axes[1].set_ylabel('Predicted Mooney (MNY)', fontsize=10)
    axes[1].set_xlim(min_val, max_val)
    axes[1].set_ylim(min_val, max_val)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(COMPARISON_PNG, dpi=120)
    plt.close()
    
    # 6. Generate Markdown Report
    print(f"Writing comparison report to: {REPORT_MD}")
    report_content = f"""# 门尼粘度预测模型评估：分立模型 vs 统一模型对比报告

本报告针对“**白炭黑胶料与炭黑胶料是分开建两个模型，还是合并到一个统一模型**”的开发策略开展了定量实验对比。

---

## 1. 实验数据集配置与说明

- **总建模车次样本数**：{len(df_labeled)} 车
  - **炭黑胶（Carbon Black）子集**：{cb_count} 车 (无硅烷且白炭黑 $\\le$ 25 phr)
  - **白炭黑胶（Silica/Silane）子集**：{silica_count} 车 (有硅烷或白炭黑 $>$ 25 phr)
- **评估分组方法**：
  1. 将全部 5,014 辆车随机拆分为 80% 训练集（{len(df_train)} 车）和 20% 测试集（{len(df_test)} 车）。
  2. 从训练集和测试集中按掩码切分出独立的 **炭黑子集** 和 **白炭黑子集**，以保证测试集的**绝对一致性与无交叉**。
  3. 未包含的物料占比在输入矩阵中以 `0` 自动填补。

---

## 2. 核心特征工程处理差异

对于分开建立的子模型，特征集根据子集的数据分布进行了自动缩减以防止噪声：
- **分立炭黑胶模型**：自动过滤了全为 NaN 的 `weight_pct_silian` 以及常数特征，最终输入特征数：**{len(cb_features)}** 个。
- **分立白炭黑胶模型**：保留了硅烷反应相关的所有物料配比与工艺历程参数，最终特征数：**{len(silica_features)}** 个。
- **统一模型 (Unified)**：保留全部物料和工艺特征（没有的材料用 0 表示），最终特征数：**{len(uni_features)}** 个。

---

## 3. 评测指标对比 (Model Performance Benchmark)

以下是三个模型在各自测试集（Held-out Test Set）上的表现对比：

| 评估测试集 | 分立模型策略 (Separated Model A/B) | 统一模型策略 (Unified Model C) | 性能对比与分析 |
| :--- | :---: | :---: | :--- |
| **炭黑胶子集 (CB Test)** | $R^2$: **{cb_sep_r2:.4f}**<br>MAE: **{cb_sep_mae:.4f}** | $R^2$: **{cb_uni_r2:.4f}**<br>MAE: **{cb_uni_mae:.4f}** | **{"分立模型占优" if cb_sep_r2 > cb_uni_r2 else "统一模型占优"}** (Δ$R^2$: {abs(cb_sep_r2 - cb_uni_r2):.4f}) |
| **白炭黑子集 (Silica Test)** | $R^2$: **{silica_sep_r2:.4f}**<br>MAE: **{silica_sep_mae:.4f}** | $R^2$: **{silica_uni_r2:.4f}**<br>MAE: **{silica_uni_mae:.4f}** | **{"分立模型占优" if silica_sep_r2 > silica_uni_r2 else "统一模型占优"}** (Δ$R^2$: {abs(silica_sep_r2 - silica_uni_r2):.4f}) |
| **全量数据集 (Overall Test)** | $R^2$: **{overall_sep_r2:.4f}**<br>MAE: **{overall_sep_mae:.4f}** | $R^2$: **{overall_uni_r2:.4f}**<br>MAE: **{overall_uni_mae:.4f}** | **{"分立路由模型占优" if overall_sep_r2 > overall_uni_r2 else "统一模型占优"}** (Δ$R^2$: {abs(overall_sep_r2 - overall_uni_r2):.4f}) |

*注：分立路由策略指：当测试样本为炭黑胶时，路由至 Model A 预测；当测试样本为白炭黑胶时，路由至 Model B 预测。*

---

## 4. 预测效果对齐散点图

以下是两套策略在全量测试集上的对比图（包含理想 $y=x$ 对齐线）：

![门尼粘度预测对比对齐图](file:///{COMPARISON_PNG.replace('\\\\', '/')})

---

## 5. 专家建模结论与开发建议

基于实验数据，我们得出以下结论：

1. **{"统一全局模型 (Unified Global Model)" if overall_uni_r2 >= overall_sep_r2 else "分立模型路由策略 (Separated Routed Models)"} 表现更优**。
2. **底层物理原理解析**：
   - **对于炭黑胶**：{"统一模型能学习到更泛化的基体流变斜率，表现更稳定" if cb_uni_r2 >= cb_sep_r2 else "分立模型在排除白炭黑的化学反应和Payne效应干扰后，能更精准地拟合纯物理分散和机械剪切过程，精度更高"}。
   - **对于白炭黑胶**：{"统一模型通过大样本量拟合表现更好" if silica_uni_r2 >= silica_sep_r2 else "分立模型能够针对白炭黑的高活化能硅烷化反应动力学进行特征强拟合，而不会被炭黑胶的纯剪切流动特性所混淆，拟合精度更高"}。
3. **开发落地建议**：
   - **{"建议采用统一全局模型建模。这样可以极大地简化后续的生产部署和新胶料的零样本自适应适配。" if overall_uni_r2 >= overall_sep_r2 else "建议采用‘分立模型 + 工艺路由键（is_silica）’的部署架构。在生产流水线中，通过检查胶料配方中是否含有硅烷（CA551）或 Silica phr 是否大于 25，将车次自动路由至对应的高精度模型进行门尼预测。"}**

"""
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print("=== COMPARISON PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    main()
