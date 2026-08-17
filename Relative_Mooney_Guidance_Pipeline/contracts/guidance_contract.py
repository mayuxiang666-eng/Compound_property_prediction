"""
Guidance Contract & Data Schemas for Target-Driven Relative Optimization
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class TargetDeltaRequest:
    order_id: str
    batch_number: int
    compound_name: str
    recipe_code: str
    material_system: str  # 'Silica' or 'CarbonBlack'
    mixer_id: str
    target_delta_mu: float  # e.g., -2.0 MU (user requests lowering by 2 MU)
    spec_low: float = 45.0
    spec_high: float = 55.0
    spec_center: float = 50.0

@dataclass
class BestConditionProfile:
    compound_name: str
    recipe_code: str
    material_system: str
    mixer_id: str
    phase: str
    group_name: str
    step_no: int
    param_name: str  # 'time_SP', 'rotor_SP', 'temp_SP', 'ram_SP'
    best_p10: float
    best_p50: float
    best_p90: float
    best_range_low: float
    best_range_high: float
    n_best_batches: int
    best_source: str  # 'same_recipe', 'same_cluster', 'fallback'
    quality_filter: str

@dataclass
class StepRecommendation:
    group_name: str
    step_no: int
    param_name: str
    current_sp: float
    recommended_sp: float
    delta_sp: float
    expected_direction: str  # 'PULL_DOWN', 'PULL_UP', 'STABLE'
    estimated_delta_band: str  # e.g. '-1.2 ~ -2.5 MU'
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    reason: str
    gate_status: str  # 'PASS', 'WARNING', 'BLOCKED'
