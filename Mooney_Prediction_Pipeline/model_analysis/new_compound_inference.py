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

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(__file__))
from train_mooney_models import add_physics_interactions


def add_gated_features(df):
    df = add_physics_interactions(df)
    if 'silica_phr' not in df.columns:
        df['silica_phr'] = 0.0
    if 'weight_pct_silian' not in df.columns:
        df['weight_pct_silian'] = 0.0

    df['is_silica_system'] = ((df['silica_phr'] >= 25.0) & (df['weight_pct_silian'] > 0.0)).astype(float)

    gated_specs = [
        ('silica_PID_Duration', 'Stage5_PID_Duration', 1.0),
        ('silica_PID_Specific_Energy', 'Stage5_PID_Specific_Energy', 1.0),
        ('silica_temp_integral_above_100', 'phys_temp_integral_above_100', 1.0),
        ('silica_PID_temp_Mean', 'Stage5_PID_temp_Mean', 1.0),
        ('cb_DryMixing_Duration', 'Stage2_DryMixing_Duration', 0.0),
        ('cb_DryMixing_Specific_Energy', 'Stage2_DryMixing_Specific_Energy', 0.0),
        ('cb_power_decay_slope', 'Stage2_power_decay_slope', 0.0),
    ]
    for new_col, source_col, active_value in gated_specs:
        if source_col in df.columns:
            df[new_col] = df[source_col].where(df['is_silica_system'] == active_value, 0.0)
        else:
            df[new_col] = 0.0

    return add_physics_interactions(df)


def prepare_feature_matrix(df, compact_features, preprocessor, selected_features):
    df = df.copy()
    if 'MixerLine' in df.columns:
        df['MixerLine'] = df['MixerLine'].astype(str).str.strip().fillna('UNKNOWN')
        dummies = pd.get_dummies(df['MixerLine'], prefix='MixerLine', dtype=float)
        df = pd.concat([df, dummies], axis=1)
        
    df = add_gated_features(df)
    
    for feature in compact_features:
        if feature not in df.columns:
            if feature.startswith('MixerLine_'):
                df[feature] = 0.0
            else:
                df[feature] = np.nan
    raw_matrix = df[compact_features].apply(pd.to_numeric, errors='coerce')
    transformed = preprocessor.transform(raw_matrix)
    model_matrix = pd.DataFrame(transformed, columns=selected_features, index=df.index)
    return df, raw_matrix, model_matrix


def compute_applicability(model_matrix, train_model_matrix):
    train_matrix = pd.DataFrame(train_model_matrix).reset_index(drop=True)
    neighbor_count = min(6, len(train_matrix))
    if neighbor_count < 2:
        mean_distance = np.zeros(len(model_matrix))
        thresholds = {'p50': 0.0, 'p90': 0.0, 'p95': 0.0, 'p99': 0.0}
    else:
        reference_nn = NearestNeighbors(n_neighbors=neighbor_count, metric='euclidean')
        reference_nn.fit(train_matrix)
        reference_distances, _ = reference_nn.kneighbors(train_matrix)
        train_mean_distance = reference_distances[:, 1:].mean(axis=1)
        thresholds = {
            'p50': float(np.quantile(train_mean_distance, 0.50)),
            'p90': float(np.quantile(train_mean_distance, 0.90)),
            'p95': float(np.quantile(train_mean_distance, 0.95)),
            'p99': float(np.quantile(train_mean_distance, 0.99)),
        }

        new_neighbor_count = min(5, len(train_matrix))
        new_nn = NearestNeighbors(n_neighbors=new_neighbor_count, metric='euclidean')
        new_nn.fit(train_matrix)
        distances, _ = new_nn.kneighbors(model_matrix)
        mean_distance = distances.mean(axis=1)

    reliability = []
    for distance in mean_distance:
        if distance <= thresholds['p90']:
            reliability.append('OK_in_domain')
        elif distance <= thresholds['p95']:
            reliability.append('Caution_near_edge')
        elif distance <= thresholds['p99']:
            reliability.append('High_risk_far_edge')
        else:
            reliability.append('Out_of_domain')
    return mean_distance, reliability, thresholds


def summarize_out_of_range_features(raw_matrix, reference_profile, max_features=8):
    profile = reference_profile.set_index('Feature') if len(reference_profile) else pd.DataFrame()
    summaries = []
    counts = []
    for _, row in raw_matrix.iterrows():
        flags = []
        for feature, value in row.items():
            if feature not in profile.index or pd.isna(value):
                continue
            low = profile.at[feature, 'train_p01']
            high = profile.at[feature, 'train_p99']
            median = profile.at[feature, 'train_p50']
            if pd.notna(low) and value < low:
                flags.append((feature, abs(value - median), 'below_train_p01'))
            elif pd.notna(high) and value > high:
                flags.append((feature, abs(value - median), 'above_train_p99'))
        flags = sorted(flags, key=lambda item: item[1], reverse=True)
        summaries.append('; '.join([f'{feature}:{direction}' for feature, _, direction in flags[:max_features]]))
        counts.append(len(flags))
    return counts, summaries


def estimate_bias_from_calibration(bundle, calibration_csv, label_col):
    if not calibration_csv:
        return 0.0, {}
    calibration_df = pd.read_csv(calibration_csv, low_memory=False)
    if label_col not in calibration_df.columns:
        raise ValueError(f'Calibration file must contain label column: {label_col}')

    _, _, calibration_matrix = prepare_feature_matrix(
        calibration_df,
        bundle['compact_features'],
        bundle['preprocessor'],
        bundle['selected_features'],
    )
    raw_pred = bundle['model'].predict(calibration_matrix)
    
    # Check if this is a residual model
    if 'family_medians' in bundle:
        family_medians = bundle.get('family_medians', {})
        track_median = bundle.get('track_median', 0.0)
        
        calibration_pred_absolute = []
        for idx, row in calibration_df.iterrows():
            comp_name = None
            for col in ['CompoundName', 'Compound', 'compound_name', 'CompoundDescription']:
                if col in row and pd.notna(row[col]):
                    comp_name = str(row[col]).strip()
                    break
            baseline = family_medians.get(comp_name, track_median)
            pred_res = raw_pred[idx]
            calibration_pred_absolute.append(float(baseline + pred_res))
        calibration_pred = np.array(calibration_pred_absolute)
    else:
        calibration_pred = raw_pred

    actual = pd.to_numeric(calibration_df[label_col], errors='coerce')
    valid_mask = actual.notna()
    residual = actual[valid_mask].values - calibration_pred[valid_mask.values]
    if len(residual) == 0:
        return 0.0, {'calibration_rows': 0}

    bias = float(np.mean(residual))
    summary = {
        'calibration_rows': int(len(residual)),
        'bias_correction': bias,
        'calibration_mae_before': float(mean_absolute_error(actual[valid_mask], calibration_pred[valid_mask.values])),
        'calibration_rmse_before': float(root_mean_squared_error(actual[valid_mask], calibration_pred[valid_mask.values])),
        'calibration_mae_after': float(mean_absolute_error(actual[valid_mask], calibration_pred[valid_mask.values] + bias)),
        'calibration_rmse_after': float(root_mean_squared_error(actual[valid_mask], calibration_pred[valid_mask.values] + bias)),
    }
    return bias, summary


def load_calibration_biases():
    # Look in the parent of the data processing directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bias_path = os.path.join(PARENT_DIR, 'models', 'results_m2_analysis', 'calibration_biases.json')
    if os.path.exists(bias_path):
        try:
            with open(bias_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load calibration biases from {bias_path}: {e}")
    return None


def get_compound_family(name):
    if not isinstance(name, str):
        return 'Unknown'
    prefix = name.split()[0] if name.split() else name
    return prefix.rstrip('-')


def get_similarity_weighted_residual_prediction(bundle, target_row, feature_row_processed, selected_features):
    try:
        train_raw = bundle.get('train_raw_df')
        if train_raw is None or len(train_raw) == 0:
            return None
            
        comp_name = None
        for col in ['CompoundName', 'Compound', 'compound_name', 'CompoundDescription']:
            if col in target_row.index and pd.notna(target_row[col]):
                comp_name = str(target_row[col]).strip()
                break
        if not comp_name:
            return None
            
        comp_samples = train_raw[train_raw['CompoundName'] == comp_name]
        n_samples = len(comp_samples)
        
        # Apply similarity weighted local training if n_samples < 50
        if n_samples >= 50 or n_samples == 0:
            return None
            
        # Recipe fingerprint columns
        fingerprint_cols = [
            'weight_pct_solid_elastomer',
            'weight_pct_natural_rubber',
            'weight_pct_silica',
            'weight_pct_oil',
            'weight_pct_silian',
            'weight_pct_carbon_black',
            'silica_phr',
            'MNY target'
        ]
        
        # Check if all fingerprint columns are present
        for col in fingerprint_cols:
            if col not in train_raw.columns or col not in target_row.index:
                return None
                
        # Extract target fingerprint
        target_fp = target_row[fingerprint_cols].values.astype(float)
        
        # Group training set by compound to get unique recipes
        grouped = train_raw.groupby('CompoundName')[fingerprint_cols].first().dropna()
        if len(grouped) < 2:
            return None
            
        # Weighted Euclidean distance
        # Weights: silica and oil are 2.5, others are 1.0
        weights = np.array([1.0, 1.0, 2.5, 2.5, 1.0, 1.0, 1.0, 1.0])
        
        distances = {}
        for other_comp, other_fp in grouped.iterrows():
            if other_comp == comp_name:
                continue
            other_fp_val = other_fp.values.astype(float)
            # Weighted Euclidean
            dist = np.sqrt(np.sum(weights * (target_fp - other_fp_val)**2))
            distances[other_comp] = dist
            
        if not distances:
            return None
            
        # Sort and select top K = 5 nearest compounds
        sorted_neighbors = sorted(distances.items(), key=lambda x: x[1])[:5]
        
        # Build local training set starting with target compound samples
        local_df = comp_samples.copy()
        local_weights = [1.0] * len(local_df)
        
        # Include neighbor samples with distance-decayed weights
        gamma = 1.5
        for neighbor_comp, dist in sorted_neighbors:
            neighbor_samples = train_raw[train_raw['CompoundName'] == neighbor_comp].copy()
            local_df = pd.concat([local_df, neighbor_samples], ignore_index=True)
            decay_weight = np.exp(-gamma * dist)
            local_weights.extend([decay_weight] * len(neighbor_samples))
            
        if len(local_df) < 8:
            return None
            
        # Preprocess features
        preprocessor = bundle['preprocessor']
        compact_features = bundle['compact_features']
        
        # Clean and preprocess local training features
        local_raw_matrix = local_df[compact_features].apply(pd.to_numeric, errors='coerce')
        local_transformed = preprocessor.transform(local_raw_matrix)
        local_model_matrix = pd.DataFrame(local_transformed, columns=selected_features)
        
        # Calculate target residuals (y_resid) for local training
        family_medians = bundle.get('family_medians', {})
        track_median = bundle.get('track_median', 0.0)
        
        local_baselines = []
        for idx, row in local_df.iterrows():
            c_name = row.get('CompoundName')
            baseline = family_medians.get(c_name, track_median)
            local_baselines.append(baseline)
            
        local_y = local_df['MNY'].values - np.array(local_baselines)
        
        # Fit localized weighted Ridge model
        from sklearn.linear_model import Ridge
        local_model = Ridge(alpha=10.0)
        local_model.fit(local_model_matrix, local_y, sample_weight=local_weights)
        
        # Predict target row residual
        pred_res = local_model.predict(feature_row_processed)
        return float(pred_res[0])
    except Exception as e:
        print(f"Warning in on-the-fly similarity weighted local training: {e}")
        return None


def compute_confidence_score(model_matrix, bundle, ensemble_preds=None):
    # 1. AD Score (Mahalanobis Distance)
    inv_cov = bundle.get('inv_covariance_matrix')
    mean_vec = bundle.get('mean_vector')
    
    ad_scores = []
    distances_maha = []
    
    if inv_cov is not None and mean_vec is not None:
        for idx, row in pd.DataFrame(model_matrix).iterrows():
            diff = row.values - mean_vec
            try:
                d_m = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
                # Convert Mahalanobis distance to score between 0 and 1
                score = np.exp(-0.3 * d_m)
            except Exception:
                d_m = np.nan
                score = 0.5
            distances_maha.append(float(d_m))
            ad_scores.append(float(score))
    else:
        ad_scores = [1.0] * len(model_matrix)
        distances_maha = [0.0] * len(model_matrix)
        
    # 2. Density Score (based on KNN Euclidean distance to train_model_matrix)
    density_scores = []
    train_matrix = pd.DataFrame(bundle['train_model_matrix']).reset_index(drop=True)
    if len(train_matrix) >= 5:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(5, len(train_matrix)), metric='euclidean')
        nn.fit(train_matrix)
        distances, _ = nn.kneighbors(model_matrix)
        mean_dists = distances.mean(axis=1)
        
        # Calculate training set 99th percentile for normalization
        train_distances, _ = nn.kneighbors(train_matrix)
        train_mean_dists = train_distances.mean(axis=1)
        p99_dist = np.quantile(train_mean_dists, 0.99)
        if p99_dist == 0:
            p99_dist = 1.0
            
        for dist in mean_dists:
            score = np.exp(-1.5 * (dist / p99_dist))
            density_scores.append(float(score))
    else:
        density_scores = [1.0] * len(model_matrix)
        
    # 3. Ensemble Uncertainty Score (based on predictions spread if available)
    uncertainty_scores = []
    if ensemble_preds is not None and len(ensemble_preds) > 0:
        preds_array = np.array(ensemble_preds)  # Shape: (num_models, num_samples)
        stds = np.std(preds_array, axis=0)      # Shape: (num_samples,)
        for std in stds:
            score = np.exp(-0.35 * std)
            uncertainty_scores.append(float(score))
    else:
        uncertainty_scores = [1.0] * len(model_matrix)
        
    # Combine scores with weights: w1=0.4, w2=0.3, w3=0.3
    confidence_scores = []
    confidence_labels = []
    
    for i in range(len(model_matrix)):
        cs = 0.4 * ad_scores[i] + 0.3 * density_scores[i] + 0.3 * uncertainty_scores[i]
        confidence_scores.append(cs)
        
        if cs >= 0.70:
            confidence_labels.append("Green")
        elif cs >= 0.40:
            confidence_labels.append("Yellow")
        else:
            confidence_labels.append("Red")
            
    return confidence_scores, confidence_labels, distances_maha


def predict_new_compound(bundle_path, input_csv, output_csv, calibration_csv=None, label_col='MNY'):
    bundle = joblib.load(bundle_path)
    df = pd.read_csv(input_csv, low_memory=False)
    prepared_df, raw_matrix, model_matrix = prepare_feature_matrix(
        df,
        bundle['compact_features'],
        bundle['preprocessor'],
        bundle['selected_features'],
    )

    raw_prediction = bundle['model'].predict(model_matrix)
    
    # Extract ensemble predictions from GBDT stack if available for confidence scoring
    ensemble_preds = []
    stack_model = bundle['model']
    if hasattr(stack_model, 'named_estimators_'):
        for name, est in stack_model.named_estimators_.items():
            try:
                pred_est = est.predict(model_matrix)
                ensemble_preds.append(pred_est)
            except Exception:
                pass
                
    # Compute multi-factor confidence score
    confidence_scores, confidence_labels, mahalanobis_dists = compute_confidence_score(
        model_matrix, bundle, ensemble_preds
    )
    
    # Check if this is a residual model and convert back to absolute
    if 'family_medians' in bundle:
        family_medians = bundle.get('family_medians', {})
        track_median = bundle.get('track_median', 0.0)
        
        prediction_absolute = []
        is_similarity_applied = []
        is_ood_fallback_applied = []
        
        for idx, row in prepared_df.iterrows():
            comp_name = None
            for col in ['CompoundName', 'Compound', 'compound_name', 'CompoundDescription']:
                if col in row and pd.notna(row[col]):
                    comp_name = str(row[col]).strip()
                    break
            baseline = family_medians.get(comp_name, track_median)
            
            # If OOD (Red label), reject residual prediction and fallback to Baseline
            if confidence_labels[idx] == "Red":
                pred_res = 0.0
                is_ood_fallback_applied.append(True)
                is_similarity_applied.append(False)
            else:
                is_ood_fallback_applied.append(False)
                # Check for similarity weighted local prediction
                local_pred_res = get_similarity_weighted_residual_prediction(
                    bundle, row, model_matrix.iloc[[idx]], bundle['selected_features']
                )
                if local_pred_res is not None:
                    pred_res = local_pred_res
                    is_similarity_applied.append(True)
                else:
                    pred_res = raw_prediction[idx]
                    is_similarity_applied.append(False)
                
            prediction_absolute.append(float(baseline + pred_res))
        prediction = np.array(prediction_absolute)
    else:
        prediction = raw_prediction
        is_similarity_applied = [False] * len(prepared_df)
        is_ood_fallback_applied = [False] * len(prepared_df)

    distance, reliability, thresholds = compute_applicability(model_matrix, bundle['train_model_matrix'])
    out_of_range_count, out_of_range_summary = summarize_out_of_range_features(raw_matrix, bundle['reference_profile'])
    bias, calibration_summary = estimate_bias_from_calibration(bundle, calibration_csv, label_col)

    # Load dynamic biases from results_m2_analysis/calibration_biases.json
    dynamic_biases = load_calibration_biases()
    
    bias_corrections = []
    bias_sources = []
    
    for idx, row in prepared_df.iterrows():
        comp_name = None
        for col in ['CompoundName', 'Compound', 'compound_name', 'CompoundDescription']:
            if col in row and pd.notna(row[col]):
                comp_name = str(row[col]).strip()
                break
        
        is_oil = float(row.get('is_oil_loading_present', 0.0)) == 1.0
        track_name = "With-Oil" if is_oil else "Without-Oil"
        
        row_bias = 0.0
        row_source = "None"
        
        if dynamic_biases:
            family = get_compound_family(comp_name) if comp_name else 'Unknown'
            if family in dynamic_biases.get('families', {}):
                row_bias = dynamic_biases['families'][family]
                row_source = f"Family:{family}"
            elif track_name in dynamic_biases.get('tracks', {}):
                row_bias = dynamic_biases['tracks'][track_name]
                row_source = f"Track:{track_name}"
        
        if row_bias == 0.0 and bias != 0.0:
            row_bias = bias
            row_source = "calibration_csv"
            
        bias_corrections.append(row_bias)
        bias_sources.append(row_source)

    result = prepared_df.copy()
    result['predicted_MNY_base'] = prediction
    result['few_shot_bias_correction'] = bias_corrections
    result['bias_correction_source'] = bias_sources
    result['predicted_MNY_calibrated'] = prediction + bias_corrections
    result['mahalanobis_distance'] = mahalanobis_dists
    result['confidence_score'] = confidence_scores
    result['confidence_label'] = confidence_labels
    result['is_similarity_applied'] = is_similarity_applied
    result['is_ood_fallback_applied'] = is_ood_fallback_applied
    result['applicability_distance'] = distance
    result['applicability_reliability'] = reliability
    result['out_of_reference_range_feature_count'] = out_of_range_count
    result['out_of_reference_range_features'] = out_of_range_summary

    if label_col in result.columns:
        actual = pd.to_numeric(result[label_col], errors='coerce')
        valid_mask = actual.notna()
        result['prediction_error_base'] = actual - result['predicted_MNY_base']
        result['prediction_error_calibrated'] = actual - result['predicted_MNY_calibrated']
        if valid_mask.any():
            calibration_summary['input_rows_with_labels'] = int(valid_mask.sum())
            calibration_summary['input_r2_base'] = float(r2_score(actual[valid_mask], result.loc[valid_mask, 'predicted_MNY_base'])) if valid_mask.sum() >= 2 else np.nan
            calibration_summary['input_mae_base'] = float(mean_absolute_error(actual[valid_mask], result.loc[valid_mask, 'predicted_MNY_base']))
            calibration_summary['input_rmse_base'] = float(root_mean_squared_error(actual[valid_mask], result.loc[valid_mask, 'predicted_MNY_base']))
            calibration_summary['input_mae_calibrated'] = float(mean_absolute_error(actual[valid_mask], result.loc[valid_mask, 'predicted_MNY_calibrated']))
            calibration_summary['input_rmse_calibrated'] = float(root_mean_squared_error(actual[valid_mask], result.loc[valid_mask, 'predicted_MNY_calibrated']))

    result.to_csv(output_csv, index=False, encoding='utf-8-sig')
    summary_path = os.path.splitext(output_csv)[0] + '_summary.json'
    summary = {
        'bundle_path': bundle_path,
        'input_csv': input_csv,
        'output_csv': output_csv,
        'model_subset': bundle.get('subset_name'),
        'model_name': bundle.get('model_name'),
        'distance_thresholds': thresholds,
        'few_shot_calibration': calibration_summary,
        'reliability_counts': result['applicability_reliability'].value_counts().to_dict(),
        'confidence_counts': result['confidence_label'].value_counts().to_dict(),
    }
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f'Saved predictions to {output_csv}')
    print(f'Saved summary to {summary_path}')


def main():
    parser = argparse.ArgumentParser(description='Predict MNY for new compounds with applicability-domain checks.')
    parser.add_argument('--bundle', required=True, help='Path to mooney_model_bundle.joblib')
    parser.add_argument('--input-csv', required=True, help='Feature CSV for new batches')
    parser.add_argument('--output-csv', required=True, help='Prediction output CSV')
    parser.add_argument('--calibration-csv', default=None, help='Optional few-shot labeled CSV used to estimate bias correction')
    parser.add_argument('--label-col', default='MNY', help='Lab label column name for calibration/evaluation')
    args = parser.parse_args()
    predict_new_compound(args.bundle, args.input_csv, args.output_csv, args.calibration_csv, args.label_col)


if __name__ == '__main__':
    main()