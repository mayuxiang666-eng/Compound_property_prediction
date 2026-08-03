# ============================================================================
# Material System Clustering Engine (Silica vs Carbon Black)
# ============================================================================
# Classifies compounds into Silica vs Carbon Black according to plant naming
# convention and formulation features:
#
# Rules:
# - T-prefix / S-prefix compounds (e.g. M1-T..., M1-S...) -> Silica
# - A-prefix / B-prefix compounds (e.g. M1-A..., M1-B...) -> CarbonBlack
# - Formulation PHR override if explicit Silica/CB columns exist.
# ============================================================================

import numpy as np
import pandas as pd


def cluster_silica_carbon_black(df: pd.DataFrame) -> pd.DataFrame:
    """Classify compounds into Silica vs Carbon Black according to Continental plant conventions.

    Naming Conventions:
    - Compounds starting with T- (or M1-T...) are Silica-dominated.
    - Compounds starting with A- or B- (e.g. M1-A..., M1-B...) are Carbon Black-dominated.
    - Adds canonical `material_system` ('Silica' or 'CarbonBlack').
    """
    df = df.copy()

    silica_cols = [c for c in df.columns if 'silica' in c.lower() or 'sio2' in c.lower()]
    cb_cols = [c for c in df.columns if ('carbon' in c.lower() and 'black' in c.lower()) or 'cb' in c.lower()]

    def classify_row(row):
        comp = str(row.get('CompoundName', '')).strip().upper()

        # Extract prefix code (e.g. from M1-T15760... -> T, M1-A00268... -> A)
        parts = comp.split('-')
        code_part = ''
        if len(parts) > 1:
            code_part = parts[1]
        else:
            code_part = comp

        # Rule 1: Naming Convention
        if code_part.startswith('T') or comp.startswith('T'):
            return 'Silica'
        if code_part.startswith('A') or code_part.startswith('B') or comp.startswith('A') or comp.startswith('B'):
            return 'CarbonBlack'
        if code_part.startswith('S') or comp.startswith('S'):
            return 'Silica'

        # Rule 2: Formulation Check
        s_val = sum(row[c] for c in silica_cols if pd.notnull(row[c])) if silica_cols else 0.0
        c_val = sum(row[c] for c in cb_cols if pd.notnull(row[c])) if cb_cols else 0.0
        if s_val > 0 or c_val > 0:
            return 'Silica' if s_val >= c_val else 'CarbonBlack'

        # Rule 3: Keyword Search Fallback
        if any(k in comp for k in ['SILICA', 'SIL', 'SFE', 'WET']):
            return 'Silica'

        return 'CarbonBlack'

    df['material_system'] = df.apply(classify_row, axis=1)
    df['compound_cluster'] = df['material_system']

    return df
