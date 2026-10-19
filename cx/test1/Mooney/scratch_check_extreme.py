import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

math_df = pd.read_pickle('cache/mathematical_dataset.pkl')
lab = pd.read_pickle('cache/lab.pkl')
strict_lab = lab[lab['LabelInPallet'] == True].rename(columns={'SampleBatch': 'BatchNumber'})
math_df['OrderID'] = math_df['OrderID'].astype(str)
strict_lab['OrderID'] = strict_lab['OrderID'].astype(str)
df = math_df.drop(columns=['Mooney']).merge(strict_lab, on=['OrderID', 'BatchNumber'])

orders = ['2300167', '2139521', '2320613', '2288700', '2351308']
sub = df[df['OrderID'].isin(orders)].drop_duplicates('OrderID')
for _, r in sub.iterrows():
    t7 = r.get('MST_S7_Torque_Nm', None)
    t8 = r.get('MST_S8_Torque_Nm', None)
    mean_t = r.get('Curve_Torque_Mean_MST_Nm', None)
    int_t = r.get('Curve_Int_Torque_Full_Nms', None)
    p_mean = r.get('Curve_Power_Mean_MST_kW', None)
    print(f"Order {r['OrderID']}: Mooney={r['Mooney']:.2f}, S7_T={t7}, S8_T={t8}, Mean_T={mean_t}, Int_T={int_t}, P_mean={p_mean}")
