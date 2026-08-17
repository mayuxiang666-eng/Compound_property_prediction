"""Read-only active/archive curve identity audit for the five parity batches."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.data_processing.canonical_mms_source import query_mms


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "curve_source_parity"


def main() -> None:
    sql = """
    SELECT N'SFEPLANT' SourceDatabase, CAST(OrderID AS VARCHAR(64)) OrderID,
           BatchNumber, EquipmentID, [Timestamp],
           DATALENGTH(Curve1) Curve1Bytes, DATALENGTH(Curve2) Curve2Bytes,
           DATALENGTH(Curve5) Curve5Bytes, DATALENGTH(Curve6) Curve6Bytes,
           DATALENGTH(Curve7) Curve7Bytes,
           CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CONVERT(VARBINARY(MAX), Curve1)), 2) Curve1Hash,
           CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CONVERT(VARBINARY(MAX), Curve2)), 2) Curve2Hash,
           CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CONVERT(VARBINARY(MAX), Curve5)), 2) Curve5Hash,
           CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CONVERT(VARBINARY(MAX), Curve6)), 2) Curve6Hash,
           CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CONVERT(VARBINARY(MAX), Curve7)), 2) Curve7Hash
      FROM SFEPLANT.dbo.BatchCurve WITH (NOLOCK)
     WHERE CAST(OrderID AS VARCHAR(64))='2016836' AND BatchNumber IN (1,2,3,4,21)
    UNION ALL
    SELECT N'SFEPLANT_ARCHIVE', CAST(OrderID AS VARCHAR(64)), BatchNumber, EquipmentID, [Timestamp],
           DATALENGTH(Curve1), DATALENGTH(Curve2), DATALENGTH(Curve5), DATALENGTH(Curve6), DATALENGTH(Curve7),
           CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CONVERT(VARBINARY(MAX), Curve1)), 2),
           CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CONVERT(VARBINARY(MAX), Curve2)), 2),
           CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CONVERT(VARBINARY(MAX), Curve5)), 2),
           CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CONVERT(VARBINARY(MAX), Curve6)), 2),
           CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CONVERT(VARBINARY(MAX), Curve7)), 2)
      FROM SFEPLANT_ARCHIVE.dbo.BatchCurve WITH (NOLOCK)
     WHERE CAST(OrderID AS VARCHAR(64))='2016836' AND BatchNumber IN (1,2,3,4,21)
    """
    frame = query_mms(sql)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "order_2016836_active_archive_curve_hashes.csv", index=False)
    hash_columns = [f"Curve{number}Hash" for number in (1, 2, 5, 6, 7)]
    rows = []
    for batch, group in frame.groupby("BatchNumber"):
        rows.append({
            "BatchNumber": int(batch),
            "source_rows": int(len(group)),
            **{
                f"Curve{number}Same": bool(group[f"Curve{number}Hash"].nunique(dropna=False) == 1)
                for number in (1, 2, 5, 6, 7)
            },
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "order_2016836_active_archive_curve_comparison.csv", index=False)
    summary = {
        "order_id": "2016836",
        "batches": [1, 2, 3, 4, 21],
        "all_curve_hashes_equal": bool(comparison[[column for column in comparison if column.endswith("Same")]].all(axis=None)),
        "comparison": comparison.to_dict("records"),
        "read_only": True,
    }
    with (OUT / "order_2016836_active_archive_curve_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
