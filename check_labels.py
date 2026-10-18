import csv
from pathlib import Path

for name in ["train_labels.csv", "test_labels.csv"]:
    path = Path(name)

    if not path.exists():
        print(f"{name}: NOT FOUND")
        continue

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    queries = {
        row["query_id"]
        for row in rows
        if row.get("query_id")
    }

    print(f"\nFile: {name}")
    print(f"Labelled pairs: {len(rows)}")
    print(f"Unique queries: {len(queries)}")