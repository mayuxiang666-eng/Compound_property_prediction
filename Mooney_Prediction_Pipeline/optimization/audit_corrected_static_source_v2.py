from __future__ import annotations

import json
from pathlib import Path

from Mooney_Prediction_Pipeline.data_processing.canonical_mms_source_v2 import build_corrected_static_features


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow" / "mms_source_contract_v2"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = build_corrected_static_features("2016836")
    frame.to_csv(OUT / "order_2016836_corrected_static_features.csv", index=False)
    summary = {
        "order_id": "2016836",
        "batch_count": int(len(frame)),
        "missing_by_column": {column: int(frame[column].isna().sum()) for column in frame.columns},
        "first_batch": frame.head(1).to_dict("records"),
        "material_contract_version": "MATERIAL-TAXONOMY-HF-V2",
        "source": "MMS BatchMaterials + OrderMaterials + RecipeMaterials + supplier QM",
        "read_only_source_audit": True,
    }
    with (OUT / "order_2016836_corrected_static_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
