"""
Convert the MySQL dump data/medcare_mysql.sql into the source CSVs the
pipeline reads: data/demand.csv, sales.csv, inventory.csv, batches.csv.

    python database/import_mysql_dump.py            # then: python main.py init-db

Values are copied exactly as they appear in the INSERT statements; nothing is
calculated. The dump contains only source data (no forecasts or alerts).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

DUMP = config.DATA_DIR / "medcare_mysql.sql"
TARGETS = {"demand": config.SOURCE_DEMAND_CSV, "sales": config.SOURCE_SALES_CSV,
           "inventory": config.INVENTORY_CSV, "batches": config.BATCHES_CSV}
INSERT_RE = re.compile(r"INSERT INTO (\w+) \(([^)]*)\) VALUES\s*(.*?);\s*$", re.S | re.M)
VALUE_RE = re.compile(r"'(?:[^'\\]|\\.)*'|[^,]+")


def parse_dump(path: Path) -> dict[str, pd.DataFrame]:
    text = path.read_text(encoding="utf-8")
    parts: dict[str, list[pd.DataFrame]] = {}
    for m in INSERT_RE.finditer(text):
        table, cols = m.group(1), [c.strip() for c in m.group(2).split(",")]
        rows = []
        for tup in re.findall(r"\(([^()]*)\)", m.group(3)):
            vals = [v.strip() for v in VALUE_RE.findall(tup)]
            rows.append([v[1:-1] if v.startswith("'") else v for v in vals])
        parts.setdefault(table, []).append(pd.DataFrame(rows, columns=cols))
    return {t: pd.concat(p, ignore_index=True) for t, p in parts.items()}


def main() -> int:
    if not DUMP.exists():
        print(f"{DUMP} not found")
        return 1
    tables = parse_dump(DUMP)
    missing = [t for t in TARGETS if t not in tables]
    if missing:
        print(f"Dump is missing tables: {missing}")
        return 1
    for name, target in TARGETS.items():
        tables[name].to_csv(target, index=False)
        print(f"  {name:<10} {len(tables[name]):>6} rows -> {target.relative_to(ROOT)}")
    print("Done. Now run: python main.py init-db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
