import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

# Load batch dataset (Sep 18 - Oct 08, 20,985 batches)
df = pd.read_csv('reports/msb_minus10s_october_doe/m1_batches_sep18_oct08.csv')
bom = pd.read_csv('scratch/compound_bom_classification.csv')

# Filter out cold-start batches (BatchNumber <= 3) and invalid weights/energies
df_clean = df[(df['BatchNumber'] > 3) & (df['BatchWeight'] > 150) & (df['MST_Energy_kWh'] > 10)].copy()

# Determine Total Batch Energy & Net MSB Energy properly per mixer line
# On 92-1100, MSB_Energy_kWh is incremental (~7-15 kWh), whereas on 92-1020..1090 MSB_Energy_kWh is cumulative (> MST_Energy_kWh)
df_clean['Total_Energy_kWh'] = np.where(
    df_clean['MSB_Energy_kWh'] > df_clean['MST_Energy_kWh'],
    df_clean['MSB_Energy_kWh'],
    df_clean['MST_Energy_kWh'] + df_clean['MSB_Energy_kWh'].fillna(0)
)
df_clean['MSB_Net_Energy_kWh'] = np.where(
    df_clean['MSB_Energy_kWh'] > df_clean['MST_Energy_kWh'],
    df_clean['MSB_Energy_kWh'] - df_clean['MST_Energy_kWh'],
    df_clean['MSB_Energy_kWh'].fillna(0)
)
df_clean['Spec_Energy_kWh_t'] = df_clean['Total_Energy_kWh'] / (df_clean['BatchWeight'] / 1000.0)
df_clean['MST_Spec_Energy_kWh_t'] = df_clean['MST_Energy_kWh'] / (df_clean['BatchWeight'] / 1000.0)
df_clean['MSB_Spec_Energy_kWh_t'] = df_clean['MSB_Net_Energy_kWh'] / (df_clean['BatchWeight'] / 1000.0)

# Merge BOM filler info
df_clean = df_clean.merge(bom[['CompoundName', 'AvgSilicaSilane_kg', 'AvgCarbonBlack_kg']], on='CompoundName', how='left')

# Classify into Silica vs Carbon Black
# Rule: If AvgSilicaSilane_kg > 40 -> Silica; else if AvgCarbonBlack_kg > 40 -> Carbon Black;
# Fallback if not in BOM: M1-T (except T00011, T05073) with MST_MaxTemp_C >= 158 and MST_StepCount >= 8 -> Silica, else Carbon Black
def classify_filler(row):
    sil = row['AvgSilicaSilane_kg']
    cb = row['AvgCarbonBlack_kg']
    cname = str(row['CompoundName'])
    if pd.notna(sil) and pd.notna(cb) and (sil > 10 or cb > 10):
        return 'Silica (白炭黑)' if sil > cb else 'Carbon Black (炭黑)'
    if cname.startswith('M1-T') and not ('T00011' in cname or 'T05073' in cname):
        return 'Silica (白炭黑)'
    return 'Carbon Black (炭黑)'

df_clean['FillerType'] = df_clean.apply(classify_filler, axis=1)

# Remove extreme 1% outliers per compound (equipment stops / sensor spikes) before computing quantiles
def calc_compound_metrics(g):
    # Trim 1st and 99th percentile of Spec_Energy_kWh_t
    q01 = g['Spec_Energy_kWh_t'].quantile(0.01)
    q99 = g['Spec_Energy_kWh_t'].quantile(0.99)
    sub = g[(g['Spec_Energy_kWh_t'] >= q01) & (g['Spec_Energy_kWh_t'] <= q99)]
    if len(sub) < 30:
        sub = g
    
    p15 = sub['Spec_Energy_kWh_t'].quantile(0.15)
    p50 = sub['Spec_Energy_kWh_t'].quantile(0.50)
    p85 = sub['Spec_Energy_kWh_t'].quantile(0.85)
    mean_e = sub['Spec_Energy_kWh_t'].mean()
    std_e = sub['Spec_Energy_kWh_t'].std()
    
    mst_p15 = sub['MST_Spec_Energy_kWh_t'].quantile(0.15)
    mst_p85 = sub['MST_Spec_Energy_kWh_t'].quantile(0.85)
    msb_p15 = sub['MSB_Spec_Energy_kWh_t'].quantile(0.15)
    msb_p85 = sub['MSB_Spec_Energy_kWh_t'].quantile(0.85)
    
    total_batches = len(g)
    total_tons = g['BatchWeight'].sum() / 1000.0
    # Annualized tons (assuming 20 days window -> * 17.5 for 350 working days/yr)
    annual_tons = total_tons * 17.5
    
    gap_p50_p15 = p50 - p15
    gap_p85_p15 = p85 - p15
    
    # Annualized savings potential (kWh/year) if mean/median shifts to P15 golden benchmark
    annual_kwh_saving = gap_p50_p15 * annual_tons
    
    return pd.Series({
        'FillerType': g['FillerType'].iloc[0],
        'MixerID': g['MixerID'].mode().iloc[0],
        'Batches': total_batches,
        'WindowTons': total_tons,
        'AnnualTons_Est': annual_tons,
        'BatchWeight_kg': sub['BatchWeight'].mean(),
        'MST_Steps': sub['MST_StepCount'].median(),
        'MSB_Steps': sub['MSB_StepCount'].median(),
        'MST_DumpTemp_C': sub['MST_MaxTemp_C'].mean(),
        'MSB_Step1_s': sub['MSB_Step1_Duration_s'].mean(),
        'TotalBatchEnergy_kWh': sub['Total_Energy_kWh'].mean(),
        'MST_Energy_kWh': sub['MST_Energy_kWh'].mean(),
        'MSB_NetEnergy_kWh': sub['MSB_Net_Energy_kWh'].mean(),
        'SpecEnergy_Mean': mean_e,
        'SpecEnergy_Std': std_e,
        'SpecEnergy_CV_pct': (std_e / mean_e) * 100.0,
        'SpecEnergy_P15': p15,
        'SpecEnergy_P50': p50,
        'SpecEnergy_P85': p85,
        'Gap_P85_P15_kWh_t': gap_p85_p15,
        'Gap_P50_P15_kWh_t': gap_p50_p15,
        'MST_Spread_P85_P15': mst_p85 - mst_p15,
        'MSB_Spread_P85_P15': msb_p85 - msb_p15,
        'AnnualSavingPotential_kWh': annual_kwh_saving,
        'Silica_kg': g['AvgSilicaSilane_kg'].iloc[0],
        'CarbonBlack_kg': g['AvgCarbonBlack_kg'].iloc[0]
    })

ranking_df = df_clean.groupby('CompoundName', group_keys=False).apply(calc_compound_metrics).reset_index()
# Filter compounds with meaningful production volume (>= 180 batches in the 20-day window)
ranking_valid = ranking_df[ranking_df['Batches'] >= 180].copy()

# Sort by AnnualSavingPotential_kWh (which combines Gap_P50_P15_kWh_t * Production Volume)
cb_top = ranking_valid[ranking_valid['FillerType'] == 'Carbon Black (炭黑)'].sort_values('AnnualSavingPotential_kWh', ascending=False)
sil_top = ranking_valid[ranking_valid['FillerType'] == 'Silica (白炭黑)'].sort_values('AnnualSavingPotential_kWh', ascending=False)

print("=== CARBON BLACK (炭黑胶料) TOP 10 RANKING ===")
cols_show = ['CompoundName', 'MixerID', 'Batches', 'WindowTons', 'SpecEnergy_Mean', 'SpecEnergy_Std', 'SpecEnergy_P15', 'SpecEnergy_P50', 'SpecEnergy_P85', 'Gap_P85_P15_kWh_t', 'Gap_P50_P15_kWh_t', 'AnnualSavingPotential_kWh']
print(cb_top[cols_show].head(10).to_string(index=False))

print("\n=== SILICA (白炭黑胶料) TOP 10 RANKING ===")
print(sil_top[cols_show].head(10).to_string(index=False))

ranking_valid.to_csv('scratch/all_compounds_energy_ranking.csv', index=False)
