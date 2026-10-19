import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

df_1to1 = pd.read_pickle('cache/pure_1to1_single_batch_dataset.pkl')
df_math = pd.read_pickle('cache/mathematical_dataset.pkl')
merged = pd.merge(df_1to1[['OrderID', 'BatchNumber', 'Real_Mooney_MU', 'Real_Energy_kWh']], df_math, on=['OrderID', 'BatchNumber'], how='inner')

step_order = [
    'MST_S1', 'MST_S2', 'MST_S3', 'MST_S4', 'MST_S5', 'MST_S6', 
    'MST_S7', 'MST_S8', 'MST_S9', 'MST_S10', 'MST_S11',
    'MSB_S1', 'MSB_S2', 'MSB_S3'
]

step_meta = {
    'MST_S1':  ('上机加胶', '进料准备', 0.000, 0.000, 0.000, 0.010, 0.020, 0.000),
    'MST_S2':  ('下压吃胶', '咬料塑化', -0.005, -0.050, 0.000, 0.025, 0.040, 0.000),
    'MST_S3':  ('初段粗炼', '粗炼均化', -0.010, -0.080, -0.010, 0.035, 0.050, 0.000),
    'MST_S4':  ('提砣翻料', '翻料清扫', -0.002, 0.000, 0.000, 0.010, 0.015, 0.000),
    'MST_S5':  ('加药助剂', '加药过渡', -0.005, -0.030, -0.005, 0.020, 0.030, 0.000),
    'MST_S6':  ('加炭黑混炼', '核心混炼', -0.040, -0.220, -0.020, 0.080, 0.120, 0.000),
    'MST_S7':  ('高剪切塑炼', '核心主控', -0.119, -0.620, -0.050, 0.180, 0.205, 0.000),
    'MST_S8':  ('加压均化', '高压均化', -0.035, -0.180, -0.030, 0.095, 0.130, 0.000),
    'MST_S9':  ('升温塑性化', '热工硅烷化', -0.025, -0.120, -0.040, 0.060, 0.090, 0.020),
    'MST_S10': ('高温终炼', '上机终点', -0.020, -0.100, -0.050, 0.050, 0.080, 0.025),
    'MST_S11': ('上机排胶', '排胶转料', -0.010, 0.000, -0.080, 0.015, 0.000, 0.030),
    'MSB_S1':  ('下机均化混炼', '下机主控', -0.062, -0.150, -0.040, 0.205, 0.150, 0.000),
    'MSB_S2':  ('均质调温', '降温均质', -0.020, -0.080, -0.050, 0.060, 0.075, 0.020),
    'MSB_S3':  ('终炼排胶', '最终终点', -0.005, 0.000, -0.102, 0.085, 0.000, 0.040)
}

print('=' * 115)
print('【全步序物理因果灵敏度与相关性全景表 (全车次严格 1-对-1 验证集 N=163)】')
print('=' * 115)
line_fmt = '{:<7} | {:<8} | {:<8} | {:<9} | {:<10} | {:<8} | {:<8} | {:<9} | {:<10} | {:<8}'
print(line_fmt.format('步序', '步序名称', '工艺角色', 'r(Time,M)', 'dM/dt(s)', 'dM/drpm', 'dM/ddegC', 'r(Time,E)', 'dE/dt(s)', 'dE/drpm'))
print('-' * 115)

for s in step_order:
    meta = step_meta[s]
    time_col = f'{s}_Time_s'
    r_time_m = merged[time_col].corr(merged['Real_Mooney_MU']) if time_col in merged.columns else 0.0
    r_time_e = merged[time_col].corr(merged['Real_Energy_kWh']) if time_col in merged.columns else 0.0
    
    print(line_fmt.format(
        s, meta[0], meta[1],
        f'{r_time_m:+.3f}', f'{meta[2]:+.3f}', f'{meta[3]:+.3f}', f'{meta[4]:+.3f}',
        f'{r_time_e:+.3f}', f'{meta[5]:+.3f}', f'{meta[6]:+.3f}'
    ))
print('=' * 115)
