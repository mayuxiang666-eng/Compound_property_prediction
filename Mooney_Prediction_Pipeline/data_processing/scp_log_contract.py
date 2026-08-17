"""Parse recipe-defined SCP-LOG duration/temperature control semantics."""

from __future__ import annotations

import re

import pandas as pd


DURATION_BIT = 0
TEMPERATURE_BIT = 2


def _truthy(value: object) -> bool:
    text = str(value).strip().upper()
    return text not in {"", "0", "0.0", "FALSE", "NONE", "NAN", "B00000000"}


def _active_bits(value: object, parameter_id: object) -> set[int]:
    identifier = str(parameter_id)
    single = re.search(r"LBit(\d+)$", identifier, flags=re.IGNORECASE)
    if single:
        return {int(single.group(1))} if _truthy(value) else set()
    text = str(value).strip().upper()
    if text.startswith("B") and re.fullmatch(r"B[01]+", text):
        mask = int(text[1:], 2)
    else:
        try:
            mask = int(float(text))
        except (TypeError, ValueError):
            return set()
    return {bit for bit in range(32) if mask & (1 << bit)}


def _operator(parameter_id: object) -> str | None:
    text = str(parameter_id).upper()
    if "SCPANDOR#" in text:
        return "ANDOR"
    if "SCPAND#" in text:
        return "AND"
    if "SCPOR#" in text:
        return "OR"
    return None


def _mixer(parameter_id: object) -> str | None:
    match = re.search(r"\.((?:MST|MSB))\.", str(parameter_id), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def parse_scp_log_contract(parameters: pd.DataFrame) -> pd.DataFrame:
    """Return one control-logic row per RecipeID, mixer, and StepNumber.

    Bit0 is the recipe duration condition and Bit2 is the recipe temperature
    condition.  Both legacy aggregate masks and current individual bit rows are
    supported.  Ambiguous mixed or AND/OR masks fail closed instead of being
    guessed as AND or OR.
    """
    required = {"RecipeID", "StepNumber", "ParameterID", "ParameterValue"}
    missing = required - set(parameters.columns)
    if missing:
        raise ValueError(f"SCP-LOG columns missing: {sorted(missing)}")
    work = parameters.loc[
        parameters["ParameterID"].astype(str).str.contains("SCP-LOG", case=False, na=False)
    ].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "RecipeID", "MixerGroup", "StepNumber", "duration_enabled",
                "temperature_enabled", "control_relation", "logic_status",
                "and_bits", "or_bits", "andor_bits",
            ]
        )
    work["MixerGroup"] = work["ParameterID"].map(_mixer)
    work["Operator"] = work["ParameterID"].map(_operator)
    work["ActiveBits"] = [
        _active_bits(value, identifier)
        for value, identifier in zip(work["ParameterValue"], work["ParameterID"])
    ]
    work = work.dropna(subset=["MixerGroup", "Operator"])

    rows = []
    for (recipe, mixer, step), group in work.groupby(
        ["RecipeID", "MixerGroup", "StepNumber"], dropna=False, sort=True
    ):
        by_operator = {
            operator: set().union(*group.loc[group["Operator"].eq(operator), "ActiveBits"].tolist())
            if group["Operator"].eq(operator).any() else set()
            for operator in ("AND", "OR", "ANDOR")
        }
        and_bits, or_bits, andor_bits = (
            by_operator["AND"], by_operator["OR"], by_operator["ANDOR"]
        )
        relevant = (and_bits | or_bits | andor_bits) & {DURATION_BIT, TEMPERATURE_BIT}
        duration_enabled = DURATION_BIT in relevant
        temperature_enabled = TEMPERATURE_BIT in relevant
        if andor_bits & {DURATION_BIT, TEMPERATURE_BIT}:
            relation, status = "AMBIGUOUS_ANDOR", "BLOCKED_AMBIGUOUS_SCP_LOG"
        elif {DURATION_BIT, TEMPERATURE_BIT}.issubset(and_bits):
            relation, status = "AND", "CONTROL_LOGIC_OK"
        elif {DURATION_BIT, TEMPERATURE_BIT}.issubset(or_bits):
            relation, status = "OR", "CONTROL_LOGIC_OK"
        elif duration_enabled and temperature_enabled:
            relation, status = "MIXED", "BLOCKED_AMBIGUOUS_SCP_LOG"
        elif duration_enabled:
            relation, status = "DURATION_ONLY", "CONTROL_LOGIC_OK"
        elif temperature_enabled:
            relation, status = "TEMPERATURE_ONLY", "CONTROL_LOGIC_OK"
        else:
            relation, status = "NONE", "NO_DURATION_TEMPERATURE_CONTROL"
        rows.append(
            {
                "RecipeID": str(recipe).strip(),
                "MixerGroup": mixer,
                "StepNumber": step,
                "duration_enabled": duration_enabled,
                "temperature_enabled": temperature_enabled,
                "control_relation": relation,
                "logic_status": status,
                "and_bits": "|".join(map(str, sorted(and_bits))),
                "or_bits": "|".join(map(str, sorted(or_bits))),
                "andor_bits": "|".join(map(str, sorted(andor_bits))),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["DURATION_BIT", "TEMPERATURE_BIT", "parse_scp_log_contract"]
