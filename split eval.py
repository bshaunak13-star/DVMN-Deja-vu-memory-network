import csv
import random
from collections import defaultdict
from pathlib import Path

INPUT_FILE = Path("relevance_labels.csv")
TRAIN_FILE = Path("train_labels.csv")
TEST_FILE = Path("test_labels.csv")

TEST_FRACTION = 0.20
SEED = 42

if not INPUT_FILE.exists():
    raise SystemExit(
        f"ERROR: {INPUT_FILE} not found. "
        "Run this script from your DVMN project folder."
    )

with INPUT_FILE.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        raise SystemExit("ERROR: CSV has no header.")

    required = {"query_id", "query", "episode_id", "relevant"}
    missing = required - set(reader.fieldnames)
    if missing:
        raise SystemExit(f"ERROR: Missing columns: {sorted(missing)}")

    rows = list(reader)
    fieldnames = reader.fieldnames

if not rows:
    raise SystemExit("ERROR: No labelled rows found.")

# Keep all episode judgements for a query in the same split.
groups = defaultdict(list)

for row in rows:
    label = row["relevant"].strip()
    if label not in {"0", "1"}:
        continue
    groups[row["query_id"].strip()].append(row)

query_ids = list(groups)
if len(query_ids) < 5:
    raise SystemExit(
        f"ERROR: Only {len(query_ids)} labelled queries. "
        "At least 5 are needed for this split."
    )

random.Random(SEED).shuffle(query_ids)

test_count = max(1, round(len(query_ids) * TEST_FRACTION))
test_ids = set(query_ids[:test_count])
train_ids = set(query_ids[test_count:])

train_rows = [r for q in train_ids for r in groups[q]]
test_rows = [r for q in test_ids for r in groups[q]]

def save_csv(path, data):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

save_csv(TRAIN_FILE, train_rows)
save_csv(TEST_FILE, test_rows)

def count_labels(data):
    positives = sum(r["relevant"].strip() == "1" for r in data)
    negatives = sum(r["relevant"].strip() == "0" for r in data)
    return positives, negatives

train_pos, train_neg = count_labels(train_rows)
test_pos, test_neg = count_labels(test_rows)

print("\nQUERY-LEVEL SPLIT COMPLETE")
print("--------------------------")
print(f"Total labelled queries: {len(query_ids)}")
print(f"Training queries:       {len(train_ids)}")
print(f"Testing queries:        {len(test_ids)}")
print(f"Training pairs:         {len(train_rows)}")
print(f"Testing pairs:          {len(test_rows)}")
print(f"Training relevant:      {train_pos}")
print(f"Training irrelevant:    {train_neg}")
print(f"Testing relevant:       {test_pos}")
print(f"Testing irrelevant:     {test_neg}")
print("\nCreated:")
print(f"  {TRAIN_FILE}")
print(f"  {TEST_FILE}")
print("\nOriginal relevance_labels.csv was not modified.")