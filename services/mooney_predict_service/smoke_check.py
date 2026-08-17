from __future__ import annotations

import json

from .db import connect, fetch_active_model


def main() -> None:
    model = fetch_active_model()
    with connect() as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sys.tables WHERE name LIKE 'Mooney%' ORDER BY name"
            ).fetchall()
        ]
        equation = conn.execute(
            """
            SELECT COUNT(*)
            FROM sys.check_constraints
            WHERE parent_object_id = OBJECT_ID(N'dbo.MooneyResult')
              AND name = N'CK_MooneyResult_FinalEquation'
            """
        ).fetchone()[0]
    print(json.dumps({
        "active_model": model["ModelVersion"],
        "release_type": model["ReleaseType"],
        "tables": tables,
        "final_equation_constraint": bool(equation),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

