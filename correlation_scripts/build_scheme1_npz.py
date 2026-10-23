import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.append(r"d:\AntigravityIDE\Project")
import db_client
import compound_tracer
import pandas as pd
import numpy as np

# 1. 加载 214 车基础映射表
csv_path = r"C:\Users\uih83490\.gemini\antigravity-ide\brain\0f968684-3a6b-429f-82f1-62de38081140\scratch\mb_mooney_raw_dataset.csv"
df = pd.read_csv(csv_path)

# 2. 过滤混炼时长 >= 390 秒的有效批次 (保留 208 车)
T_LEN = 390
df_valid = df[df['MB_CycleDuration'] >= T_LEN].copy().reset_index(drop=True)
print(f"有效对齐样本量: n = {len(df_valid)} (已剔除 {len(df) - len(df_valid)} 车小于 390s 的早排批次)")

# 3. 批量拉取母胶原始曲线
mb_orders = df_valid['MB_OrderID'].astype(str).unique().tolist()
in_orders = ",".join([f"'{o}'" for o in mb_orders])

sql = f"""
SELECT OrderID, BatchNumber, Curve1, Curve2, Curve3, Curve6
FROM BatchCurve
WHERE OrderID IN ({in_orders})
"""
df_curves = db_client.query_df(sql, database="HESFESFEPLANT")
print(f"成功拉取曲线记录数: {len(df_curves)}")

# 建立快速查找字典
curve_dict = {}
for _, row in df_curves.iterrows():
    key = (int(str(row['OrderID']).strip()), int(row['BatchNumber']))
    curve_dict[key] = {
        'temp': compound_tracer.decode_curve_hex(row.get('Curve1')),
        'power': compound_tracer.decode_curve_hex(row.get('Curve2')),
        'ram': compound_tracer.decode_curve_hex(row.get('Curve3')),
        'speed': compound_tracer.decode_curve_hex(row.get('Curve6'))
    }

# 4. 构建统一对齐的 (n, T) 数组
arr_temp = []
arr_power = []
arr_ram = []
arr_speed = []
arr_aging = []
arr_mooney = []

valid_indices = []

for idx, r in df_valid.iterrows():
    key = (int(r['MB_OrderID']), int(r['MB_BatchNumber']))
    c_data = curve_dict.get(key)
    if not c_data:
        continue
    
    t_c = c_data['temp']
    p_c = c_data['power']
    r_c = c_data['ram']
    s_c = c_data['speed']
    
    # 严格检查长度是否 >= T_LEN
    if len(t_c) < T_LEN or len(p_c) < T_LEN or len(r_c) < T_LEN or len(s_c) < T_LEN:
        continue
        
    arr_temp.append(t_c[:T_LEN])
    arr_power.append(p_c[:T_LEN])
    arr_ram.append(r_c[:T_LEN])
    arr_speed.append(s_c[:T_LEN])
    arr_aging.append(float(r['AgingHours']))
    arr_mooney.append(float(r['FM_Mooney']))
    valid_indices.append(idx)

# 转为 numpy 数组
X_temp = np.array(arr_temp, dtype=np.float32)
X_power = np.array(arr_power, dtype=np.float32)
X_ram = np.array(arr_ram, dtype=np.float32)
X_speed = np.array(arr_speed, dtype=np.float32)
X_aging = np.array(arr_aging, dtype=np.float32).reshape(-1, 1)
y_mooney = np.array(arr_mooney, dtype=np.float32)

print("\n=== 数据矩阵构建完成 ===")
print("温度矩阵 (X_temp) 形状:", X_temp.shape)
print("功率矩阵 (X_power) 形状:", X_power.shape)
print("压栓矩阵 (X_ram) 形状:", X_ram.shape)
print("转速矩阵 (X_speed) 形状:", X_speed.shape)
print("停留时间 (X_aging) 形状:", X_aging.shape)
print("终胶门尼 (y_mooney) 形状:", y_mooney.shape)

# 保存为 npz 文件
output_npz = r"d:\AntigravityIDE\Project\data\scheme1_aligned_input.npz"
np.savez_compressed(
    output_npz,
    temperature=X_temp,
    power=X_power,
    ram=X_ram,
    speed=X_speed,
    aging_hours=X_aging,
    mooney=y_mooney,
    mb_order_ids=df_valid.loc[valid_indices, 'MB_OrderID'].values,
    mb_batch_numbers=df_valid.loc[valid_indices, 'MB_BatchNumber'].values,
    fm_order_ids=df_valid.loc[valid_indices, 'FM_OrderID'].values,
    fm_pallet_ids=df_valid.loc[valid_indices, 'FM_PalletID'].values
)
print(f"数据成功保存至: {output_npz}")
