from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phase_incremental_builder_v2 import build_incremental_phase_order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order-id", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    features, audit = build_incremental_phase_order(args.order_id, metadata_path=args.metadata)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        features.to_parquet(output, index=False)
    else:
        features.to_csv(output, index=False)
    audit_path = output.with_suffix(".audit.json")
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(json.dumps({**audit, "output": str(output), "audit": str(audit_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
