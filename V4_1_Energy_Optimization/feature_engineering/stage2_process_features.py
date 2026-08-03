import pandas as pd


PROCESS_FEATURE_PREFIXES = (
    "Stage2_DryMixing_",
    "Stage3_OilLoading_",
    "Stage4_WetMixing_",
    "Stage5_PID_",
    "Stage6_BottomMixing_",
    "phys_",
    "env_",
)


def extract_stage2_process_features(df: pd.DataFrame) -> list[str]:
    """Return observed mixing-process features available in ``df``.

    Feature families are explicit so the residual model cannot silently absorb
    labels, identifiers, or recipe values when source columns change.
    """
    return [
        column
        for column in df.columns
        if column.startswith(PROCESS_FEATURE_PREFIXES)
    ]
