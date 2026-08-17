"""Validated, versioned configuration for expert-rule shadow recommendations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


REQUIRED_KEYS = {
    "policy_version",
    "mode",
    "locked_step_numbers",
    "locked_parameters",
    "dry_temperature_cap_c",
    "wet_temperature_cap_source",
    "mst_rotor_range_rpm",
    "tracking_gates",
    "process_response_gates",
    "minimum_mooney_response",
    "energy_status",
}


def load_expert_policy(path: Optional[str] = None) -> Dict[str, Any]:
    source = Path(path) if path else Path(__file__).with_name("expert_policy_v1.json")
    with source.open("r", encoding="utf-8") as handle:
        policy = json.load(handle)

    missing = sorted(REQUIRED_KEYS.difference(policy))
    if missing:
        raise ValueError(f"Expert policy missing required keys: {missing}")
    if policy["mode"] != "SHADOW_ONLY":
        raise ValueError("Expert policy must remain SHADOW_ONLY")
    if float(policy["dry_temperature_cap_c"]) != 120.0:
        raise ValueError("Dry temperature cap must be 120°C")
    rotor_low, rotor_high = map(float, policy["mst_rotor_range_rpm"])
    if rotor_low != 5.0 or rotor_high != 50.0:
        raise ValueError("MST rotor hard range must be 5–50 rpm")
    if policy["energy_status"] != "NOT_RANKED":
        raise ValueError("Energy model must remain NOT_RANKED before completion")
    return policy
