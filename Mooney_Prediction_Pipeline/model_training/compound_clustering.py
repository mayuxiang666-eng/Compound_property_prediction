"""
Compound Clustering Module V1.0
================================
Dual-dimension clustering system combining:
  - Recipe PHR Fingerprint (14D)
  - Mixing Curve Shape Fingerprint (20+D)
to identify and group compounds by physical behavior rather than name strings.

Uses sklearn HDBSCAN for adaptive density clustering.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestCentroid, NearestNeighbors
import warnings

# ============================================================================
# Feature Definitions
# ============================================================================

# 14D Recipe Fingerprint - captures "what is in the compound"
RECIPE_FINGERPRINT_COLS = [
    'weight_pct_solid_elastomer',
    'weight_pct_natural_rubber',
    'weight_pct_silica',
    'weight_pct_oil',
    'weight_pct_silian',
    'weight_pct_carbon_black',
    'silica_phr',
    'Top_Fill_Factor',
    'Bot_Fill_Factor',
    'Target_Temperature',
    'is_oil_loading_present',
    'ratio_nr_rubber',
    'ratio_filler_polymer',
    'ratio_oil_filler',
]

# 20+D Curve Shape Fingerprint - captures "how the compound behaves during mixing"
CURVE_SHAPE_FINGERPRINT_COLS = [
    # Energy distribution across stages (how energy is consumed)
    'Stage2_DryMixing_Specific_Energy',
    'Stage4_WetMixing_Specific_Energy',
    'Stage5_PID_Specific_Energy',
    'Stage6_BottomMixing_Specific_Energy',
    # Temperature curve shape
    'phys_temp_rise_rate',
    'phys_temp_integral_above_100',
    'phys_discharge_temp',
    'phys_temp_change_rate_std',
    # Torque response pattern
    'Stage2_DryMixing_Torque_Mean',
    'Stage6_BottomMixing_Torque_Mean',
    'Stage1_Torque_Max',
    # Apparent viscosity evolution
    'phys_eta_app_overall_mean',
    'phys_eta_app_pid_mean',
    'Stage2_eta_torque_End',
    # Power decay dynamics
    'Stage2_power_decay_slope',
    'phys_power_stability_pid',
    'phys_peak_power',
    # Time allocation pattern (reflects mixing program differences)
    'time_pct_DryMixing',
    'time_pct_WetMixing',
    'time_pct_PID',
    # Shear history
    'phys_shear_history_total',
    # Total mixing work
    'phys_power_integral',
]


class CompoundClusterer:
    """
    Dual-dimension compound clustering engine.
    
    Clusters compounds using a fused fingerprint of recipe PHR features and
    mixing curve shape features, via HDBSCAN adaptive density clustering.
    
    For small-sample compounds (< min_samples_for_independent_bias), cluster-level
    bias is used. For large-sample compounds, independent compound-level bias is
    preserved for maximum accuracy.
    """
    
    def __init__(self,
                 recipe_weight=0.5,
                 curve_weight=0.5,
                 min_samples_for_independent_bias=5,
                 min_cluster_size=3,
                 large_sample_threshold=30):
        """
        Args:
            recipe_weight: Weight for recipe fingerprint in fusion (alpha)
            curve_weight: Weight for curve shape fingerprint in fusion (beta)
            min_samples_for_independent_bias: Compounds with fewer samples use cluster bias
            min_cluster_size: HDBSCAN min_cluster_size parameter
            large_sample_threshold: Compounds with >= this many samples always keep independent bias
        """
        self.recipe_weight = recipe_weight
        self.curve_weight = curve_weight
        self.min_samples_for_independent_bias = min_samples_for_independent_bias
        self.min_cluster_size = min_cluster_size
        self.large_sample_threshold = large_sample_threshold
        
        # Fitted components
        self.recipe_scaler = None
        self.curve_scaler = None
        self.recipe_imputer = None
        self.curve_imputer = None
        self.hdbscan_model = None
        self.compound_to_cluster = {}
        self.compound_fingerprints = {}  # compound_name -> fused fingerprint vector
        self.cluster_centroids = {}      # cluster_id -> centroid vector
        self.compound_sample_counts = {} # compound_name -> number of training batches
        self.n_clusters_ = 0
        self.silhouette_score_ = None
        self.cluster_stats_ = None       # DataFrame with per-cluster statistics
        
    def _build_compound_fingerprints(self, df):
        """
        Build per-compound fingerprint vectors by taking the MEDIAN of all batches
        belonging to each CompoundName. Median is more robust to outliers than mean.
        
        Returns:
            compound_names: list of compound names
            recipe_matrix: (n_compounds, 14) recipe fingerprints
            curve_matrix: (n_compounds, 22) curve shape fingerprints
        """
        # Filter to columns that exist in the dataframe
        available_recipe_cols = [c for c in RECIPE_FINGERPRINT_COLS if c in df.columns]
        available_curve_cols = [c for c in CURVE_SHAPE_FINGERPRINT_COLS if c in df.columns]
        
        # Group by CompoundName and compute median for each feature
        grouped = df.groupby('CompoundName')
        
        recipe_medians = grouped[available_recipe_cols].median()
        curve_medians = grouped[available_curve_cols].median()
        sample_counts = grouped.size()
        
        compound_names = recipe_medians.index.tolist()
        
        return compound_names, recipe_medians.values, curve_medians.values, sample_counts.to_dict(), \
               available_recipe_cols, available_curve_cols
    
    def fit(self, df):
        """
        Fit the clusterer on training data.
        
        1. Build per-compound recipe + curve fingerprints (median aggregation)
        2. Scale and fuse the two fingerprint dimensions
        3. Run HDBSCAN clustering
        4. Build compound_to_cluster mapping
        
        Args:
            df: Training DataFrame with CompoundName, recipe features, and curve features
            
        Returns:
            self
        """
        print("\n" + "="*70)
        print("COMPOUND CLUSTERING: Dual-Dimension Fingerprint Analysis")
        print("="*70)
        
        # Build fingerprints
        compound_names, recipe_matrix, curve_matrix, sample_counts, \
            recipe_cols_used, curve_cols_used = self._build_compound_fingerprints(df)
        
        self.compound_sample_counts = sample_counts
        n_compounds = len(compound_names)
        
        print(f"\nTotal compounds: {n_compounds}")
        print(f"Recipe fingerprint dimensions: {recipe_matrix.shape[1]} ({len(recipe_cols_used)} features)")
        print(f"Curve shape fingerprint dimensions: {curve_matrix.shape[1]} ({len(curve_cols_used)} features)")
        print(f"Compounds with < {self.min_samples_for_independent_bias} samples: "
              f"{sum(1 for v in sample_counts.values() if v < self.min_samples_for_independent_bias)}")
        
        # Impute missing values
        self.recipe_imputer = SimpleImputer(strategy='median')
        self.curve_imputer = SimpleImputer(strategy='median')
        
        recipe_imputed = self.recipe_imputer.fit_transform(recipe_matrix)
        curve_imputed = self.curve_imputer.fit_transform(curve_matrix)
        
        # Scale independently
        self.recipe_scaler = StandardScaler()
        self.curve_scaler = StandardScaler()
        
        recipe_scaled = self.recipe_scaler.fit_transform(recipe_imputed)
        curve_scaled = self.curve_scaler.fit_transform(curve_imputed)
        
        # Weighted fusion
        fused = np.hstack([
            recipe_scaled * self.recipe_weight,
            curve_scaled * self.curve_weight
        ])
        
        # Store fingerprints for cold-start lookup
        for i, name in enumerate(compound_names):
            self.compound_fingerprints[name] = fused[i]
        
        # Run HDBSCAN
        print(f"\nRunning HDBSCAN (min_cluster_size={self.min_cluster_size})...")
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=2,
            metric='euclidean',
            store_centers='centroid'
        )
        labels = self.hdbscan_model.fit_predict(fused)
        
        # Build mapping
        n_noise = sum(1 for l in labels if l == -1)
        unique_labels = set(labels) - {-1}
        self.n_clusters_ = len(unique_labels)
        
        for i, name in enumerate(compound_names):
            self.compound_to_cluster[name] = labels[i]
        
        # Build cluster centroids for cold-start nearest-neighbor matching
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            self.cluster_centroids[cluster_id] = fused[mask].mean(axis=0)
        
        # Compute silhouette score (exclude noise points)
        valid_mask = labels != -1
        if valid_mask.sum() > 1 and self.n_clusters_ > 1:
            self.silhouette_score_ = silhouette_score(fused[valid_mask], labels[valid_mask])
        else:
            self.silhouette_score_ = None
        
        # Build cluster statistics table
        cluster_stats = []
        for cluster_id in sorted(unique_labels):
            members = [compound_names[i] for i in range(n_compounds) if labels[i] == cluster_id]
            total_batches = sum(sample_counts.get(m, 0) for m in members)
            cluster_stats.append({
                'cluster_id': cluster_id,
                'n_compounds': len(members),
                'total_batches': total_batches,
                'members_preview': ', '.join(members[:5]) + ('...' if len(members) > 5 else '')
            })
        
        noise_members = [compound_names[i] for i in range(n_compounds) if labels[i] == -1]
        if noise_members:
            cluster_stats.append({
                'cluster_id': -1,
                'n_compounds': len(noise_members),
                'total_batches': sum(sample_counts.get(m, 0) for m in noise_members),
                'members_preview': ', '.join(noise_members[:5]) + ('...' if len(noise_members) > 5 else '')
            })
        
        self.cluster_stats_ = pd.DataFrame(cluster_stats)
        
        # Print report
        print(f"\n--- Clustering Results ---")
        print(f"Clusters found: {self.n_clusters_}")
        print(f"Noise compounds (unclustered): {n_noise}")
        if self.silhouette_score_ is not None:
            print(f"Silhouette Score: {self.silhouette_score_:.4f}")
        print(f"\nCluster size distribution:")
        for _, row in self.cluster_stats_.iterrows():
            cid = int(row['cluster_id'])
            label = f"Cluster {cid}" if cid >= 0 else "Noise (-1)"
            print(f"  {label:15s}: {int(row['n_compounds']):3d} compounds, "
                  f"{int(row['total_batches']):5d} batches | {row['members_preview']}")
        
        print("="*70 + "\n")
        
        return self
    
    def get_cluster_for_compound(self, compound_name, row_features=None):
        """
        Get cluster ID for a compound. If compound is unknown (cold start),
        use nearest centroid matching on the fused fingerprint vector.
        
        Args:
            compound_name: The CompoundName string
            row_features: Optional single-row DataFrame with features for cold-start fingerprinting
            
        Returns:
            cluster_id (int), is_cold_start (bool)
        """
        if compound_name in self.compound_to_cluster:
            return self.compound_to_cluster[compound_name], False
        
        # Cold start: compute fingerprint and find nearest cluster centroid
        if row_features is not None and len(self.cluster_centroids) > 0:
            available_recipe_cols = [c for c in RECIPE_FINGERPRINT_COLS if c in row_features.columns]
            available_curve_cols = [c for c in CURVE_SHAPE_FINGERPRINT_COLS if c in row_features.columns]
            
            recipe_vec = row_features[available_recipe_cols].values.reshape(1, -1)
            curve_vec = row_features[available_curve_cols].values.reshape(1, -1)
            
            recipe_imp = self.recipe_imputer.transform(recipe_vec)
            curve_imp = self.curve_imputer.transform(curve_vec)
            
            recipe_sc = self.recipe_scaler.transform(recipe_imp)
            curve_sc = self.curve_scaler.transform(curve_imp)
            
            fused_vec = np.hstack([recipe_sc * self.recipe_weight, curve_sc * self.curve_weight])
            
            # Find nearest centroid
            best_cluster = -1
            best_dist = float('inf')
            for cid, centroid in self.cluster_centroids.items():
                dist = np.linalg.norm(fused_vec[0] - centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = cid
            
            return best_cluster, True
        
        return -1, True
    
    def should_use_independent_bias(self, compound_name):
        """
        Determine if a compound should use its own independent bias
        or inherit from cluster.
        
        Compounds with >= min_samples_for_independent_bias keep their own bias.
        """
        count = self.compound_sample_counts.get(compound_name, 0)
        return count >= self.min_samples_for_independent_bias
    
    def get_fused_fingerprints_for_viz(self):
        """
        Return the fused fingerprint matrix and labels for visualization.
        
        Returns:
            names: list of compound names
            fused_matrix: (n_compounds, D) fused fingerprint matrix
            labels: list of cluster labels
            sample_counts: dict of compound -> sample count
        """
        names = list(self.compound_fingerprints.keys())
        fused_matrix = np.array([self.compound_fingerprints[n] for n in names])
        labels = [self.compound_to_cluster.get(n, -1) for n in names]
        return names, fused_matrix, labels, self.compound_sample_counts
