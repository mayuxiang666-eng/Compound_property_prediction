# ============================================================================
# V3.2 Regularized Compound Bias Shrinkage (Stage 1b)
# ============================================================================
# Formula:
#   w(c) = n_eff(c) / (n_eff(c) + k)
#   Shrunk_Bias(c) = w(c) * Raw_Bias(c) + (1 - w(c)) * Cluster_Bias(Cluster(c))
# ============================================================================

import numpy as np
import pandas as pd


class RegularizedBiasEstimator:
    """
    Computes shrinkage-regularized nominal bias per compound.
    Shrinks small-sample compound biases toward cluster-level average.
    """
    def __init__(self, shrinkage_k=5.0, compound_col="CompoundName", cluster_col="cluster_id"):
        self.shrinkage_k = shrinkage_k
        self.compound_col = compound_col
        self.cluster_col = cluster_col
        
        self.global_bias_ = 0.0
        self.cluster_biases_ = {}
        self.compound_biases_ = {}
        self.bias_report_ = pd.DataFrame()
        
    def fit(self, y_true, y_stage1_pred, df_meta):
        """
        Fits shrinkage bias lookup maps from training set residuals.
        """
        residuals = y_true - y_stage1_pred
        if self.compound_col not in df_meta.columns:
            raise ValueError(f"Missing required compound column: {self.compound_col}")

        work = df_meta[[self.compound_col]].copy()
        work['residual'] = residuals
        
        if '_w_label_raw' in df_meta.columns:
            work['w'] = df_meta['_w_label_raw']
        else:
            work['w'] = 1.0
            
        if self.cluster_col in df_meta.columns:
            work['cluster_id'] = df_meta[self.cluster_col].fillna('GLOBAL').astype(str)
        else:
            work['cluster_id'] = 'GLOBAL'

        work['w'] = pd.to_numeric(work['w'], errors='coerce').fillna(0.0)
        if work['w'].sum() <= 0:
            raise ValueError("Bias shrinkage requires positive label-group support weights.")
            
        # Global bias
        self.global_bias_ = np.average(work['residual'], weights=work['w'])
        
        # Cluster-level bias
        cluster_grp = work.groupby('cluster_id').apply(
            lambda g: np.average(g['residual'], weights=g['w']) if g['w'].sum() > 0 else self.global_bias_,
            include_groups=False,
        )
        self.cluster_biases_ = cluster_grp.to_dict()
        
        # Compound-level raw bias and n_eff
        comp_stats = work.groupby(self.compound_col).agg(
            n_eff=('w', 'sum'),
            cluster_id=('cluster_id', 'first')
        ).reset_index()
        
        # Compute weighted raw compound bias
        comp_raw_biases = {}
        for comp, g in work.groupby(self.compound_col):
            comp_raw_biases[comp] = np.average(g['residual'], weights=g['w'])
            
        # Compute shrunk bias per compound
        self.compound_biases_ = {}
        report_rows = []
        for idx, row in comp_stats.iterrows():
            comp = row[self.compound_col]
            n_eff = row['n_eff']
            c_id = row['cluster_id']
            
            raw_b = comp_raw_biases.get(comp, self.global_bias_)
            cluster_b = self.cluster_biases_.get(c_id, self.global_bias_)
            
            w_shrink = n_eff / (n_eff + self.shrinkage_k)
            shrunk_b = w_shrink * raw_b + (1.0 - w_shrink) * cluster_b
            self.compound_biases_[comp] = shrunk_b
            report_rows.append({
                'compound_name': comp,
                'cluster_id': c_id,
                'raw_compound_bias': raw_b,
                'cluster_bias': cluster_b,
                'n_rows': int((work[self.compound_col] == comp).sum()),
                'n_label_groups': int(df_meta.loc[df_meta[self.compound_col] == comp, '_label_group_id'].nunique())
                if '_label_group_id' in df_meta.columns else int((work[self.compound_col] == comp).sum()),
                'n_eff': n_eff,
                'shrinkage_weight': w_shrink,
                'shrunk_bias': shrunk_b,
                'bias_source': 'compound_shrunk_to_cluster',
            })

        self.bias_report_ = pd.DataFrame(report_rows).sort_values(
            ['cluster_id', 'n_eff', 'compound_name'],
            ascending=[True, False, True],
        ).reset_index(drop=True)
            
        return self

    def get_bias_report(self):
        """Return the fitted Stage 1b audit table without exposing mutable state."""
        return self.bias_report_.copy()
        
    def predict_bias(self, df_meta):
        """
        Predicts nominal Stage 1b bias for new batches.
        Unseen compounds fall back to cluster bias or global bias.
        """
        biases = np.zeros(len(df_meta))
        
        for i, (_, row) in enumerate(df_meta.iterrows()):
            comp = row.get(self.compound_col, None)
            c_id = row.get(self.cluster_col, 0)
            
            if comp in self.compound_biases_:
                biases[i] = self.compound_biases_[comp]
            elif c_id in self.cluster_biases_:
                biases[i] = self.cluster_biases_[c_id]
            else:
                biases[i] = self.global_bias_
                
        return biases


if __name__ == '__main__':
    print("Regularized Bias Estimator Module ready.")
