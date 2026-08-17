"""Recipe-defined SCP-LOG termination contracts for Step recommendations.

The recipe is authoritative: a recommender may not reinterpret duration and
temperature as interchangeable controls when the CBS3 recipe declares them as
AND/OR termination conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import pandas as pd


_STEP_PATTERN = re.compile(r"MI\.1\.(MST|MSB)\.(\d+)\.", re.IGNORECASE)
_AND_PATTERN = re.compile(r"SCPAND#_SCP-Log-AND-LBit(\d+)", re.IGNORECASE)
_OR_PATTERN = re.compile(r"SCPOR#_SCP-Log-OR-LBit(\d+)", re.IGNORECASE)
_TIME_PATTERN = re.compile(r"SCP-1-Step-Time", re.IGNORECASE)
_TEMP_PATTERN = re.compile(r"SCP-3-Temp", re.IGNORECASE)

# CBS3 uses the termination-condition bit positions below for the controls
# covered by this recommender. Unknown enabled bits must block a recommendation.
_KNOWN_BITS = {0: "duration", 2: "temperature"}


@dataclass(frozen=True)
class StepTerminationLogic:
    recipe_id: str
    group_name: str
    step_no: int
    relation: str
    active_conditions: tuple[str, ...]
    unknown_bits: tuple[int, ...]

    @property
    def status(self) -> str:
        return "KNOWN" if not self.unknown_bits and self.relation in {"AND", "OR", "SINGLE", "NONE"} else "UNKNOWN"


def _active(value: object) -> bool:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(pd.notna(numeric) and float(numeric) > 0.0)


def parse_recipe_scp_logic(parameters: pd.DataFrame, *, recipe_id: str | None = None) -> tuple[StepTerminationLogic, ...]:
    """Parse RecipeCBS3Parameters rows into auditable Step termination contracts."""
    required = {"RecipeID", "StepNumber", "ParameterID", "ParameterValue"}
    missing = required.difference(parameters.columns)
    if missing:
        raise ValueError(f"Recipe parameter frame missing columns: {sorted(missing)}")

    frame = parameters.copy()
    if recipe_id is not None:
        frame = frame[frame["RecipeID"].astype(str).eq(str(recipe_id))]
    contracts: list[StepTerminationLogic] = []
    for (current_recipe, group_name, step_no), rows in _group_recipe_rows(frame):
        and_bits: set[int] = set()
        or_bits: set[int] = set()
        controls_present: set[str] = set()
        for _, row in rows.iterrows():
            parameter_id = str(row["ParameterID"])
            if _active(row["ParameterValue"]):
                if _TIME_PATTERN.search(parameter_id):
                    controls_present.add("duration")
                if _TEMP_PATTERN.search(parameter_id):
                    controls_present.add("temperature")
                and_match = _AND_PATTERN.search(parameter_id)
                or_match = _OR_PATTERN.search(parameter_id)
                if and_match:
                    and_bits.add(int(and_match.group(1)))
                if or_match:
                    or_bits.add(int(or_match.group(1)))

        enabled_bits = and_bits | or_bits
        unknown_bits = tuple(sorted(bit for bit in enabled_bits if bit not in _KNOWN_BITS))
        logical_controls = {name for bit, name in _KNOWN_BITS.items() if bit in enabled_bits}
        active_conditions = tuple(sorted(logical_controls or controls_present))
        if and_bits and or_bits:
            relation = "UNKNOWN"
        elif and_bits:
            relation = "AND" if len(active_conditions) > 1 else "SINGLE"
        elif or_bits:
            relation = "OR" if len(active_conditions) > 1 else "SINGLE"
        elif len(active_conditions) <= 1:
            relation = "SINGLE" if active_conditions else "NONE"
        else:
            relation = "UNKNOWN"
        contracts.append(
            StepTerminationLogic(
                recipe_id=str(current_recipe),
                group_name=group_name,
                step_no=step_no,
                relation=relation,
                active_conditions=active_conditions,
                unknown_bits=unknown_bits,
            )
        )
    return tuple(sorted(contracts, key=lambda item: (item.group_name, item.step_no)))


def _group_recipe_rows(parameters: pd.DataFrame) -> Iterable[tuple[tuple[str, str, int], pd.DataFrame]]:
    expanded = parameters.copy()
    parsed = expanded["ParameterID"].astype(str).str.extract(_STEP_PATTERN)
    expanded["_group"] = parsed[0].str.upper().map({"MST": "AVR_MST", "MSB": "AVR_MSB"})
    expanded["_step"] = pd.to_numeric(parsed[1], errors="coerce")
    expanded = expanded.dropna(subset=["_group", "_step"])
    for (recipe, group, step), rows in expanded.groupby(["RecipeID", "_group", "_step"], sort=True):
        yield (str(recipe), str(group), int(step)), rows


def validate_step_vector_scp_logic(
    vector: pd.DataFrame,
    contracts: tuple[StepTerminationLogic, ...],
) -> tuple[str, ...]:
    """Fail closed when a recommended vector violates recipe termination logic."""
    required = {"GroupName", "StepNo", "recommended_dur_SP", "recommended_temp_SP"}
    missing = required.difference(vector.columns)
    if missing:
        raise ValueError(f"Step vector missing columns: {sorted(missing)}")
    reasons: list[str] = []
    indexed = vector.set_index(["GroupName", "StepNo"], drop=False)
    for contract in contracts:
        key = (contract.group_name, contract.step_no)
        if key not in indexed.index:
            reasons.append(f"BLOCKED_MISSING_RECIPE_STEP_{contract.group_name}_{contract.step_no}")
            continue
        if contract.status != "KNOWN":
            reasons.append(f"BLOCKED_UNKNOWN_SCP_LOGIC_{contract.group_name}_{contract.step_no}")
            continue
        row = indexed.loc[key]
        if isinstance(row, pd.DataFrame):
            reasons.append(f"BLOCKED_DUPLICATE_RECIPE_STEP_{contract.group_name}_{contract.step_no}")
            continue
        checks = {
            "duration": _active(row["recommended_dur_SP"]),
            "temperature": _active(row["recommended_temp_SP"]),
        }
        selected = [checks[name] for name in contract.active_conditions if name in checks]
        if contract.relation == "AND" and selected and not all(selected):
            reasons.append(f"REJECTED_SCP_LOGIC_AND_{contract.group_name}_{contract.step_no}")
        elif contract.relation == "OR" and selected and not any(selected):
            reasons.append(f"REJECTED_SCP_LOGIC_OR_{contract.group_name}_{contract.step_no}")
        elif contract.relation == "SINGLE" and selected and not selected[0]:
            reasons.append(f"REJECTED_SCP_LOGIC_SINGLE_{contract.group_name}_{contract.step_no}")
    return tuple(sorted(set(reasons)))


def contracts_to_frame(contracts: tuple[StepTerminationLogic, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "recipe_id": item.recipe_id,
                "GroupName": item.group_name,
                "StepNo": item.step_no,
                "scp_relation": item.relation,
                "scp_active_conditions": "|".join(item.active_conditions),
                "scp_unknown_bits": "|".join(map(str, item.unknown_bits)),
                "scp_logic_status": item.status,
            }
            for item in contracts
        ]
    )
