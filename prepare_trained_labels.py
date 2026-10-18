import csv
from pathlib import Path

TRAINED = Path("final_dvmn_trained_rankings.csv")
KEYWORD = Path("final_keyword_rankings.csv")
OLD_LABELS = Path("final_relevance_labels.csv")
OUTPUT = Path("trained_comparison_labels.csv")

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

for path in (TRAINED, KEYWORD, OLD_LABELS):
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

trained = read_csv(TRAINED)
keyword = read_csv(KEYWORD)
old_labels = read_csv(OLD_LABELS)

# Keep existing human labels wherever the same pair appears.
labels = {}
for row in old_labels:
    key = (row["query_id"].strip(), row["episode_id"].strip())
    value = row.get("relevant", "").strip()
    if value in ("0", "1"):
        labels[key] = value

# Include every pair returned by either trained DVMN or keyword.
pairs = {}
for row in trained + keyword:
    key = (row["query_id"].strip(), row["episode_id"].strip())
    if key not in pairs:
        pairs[key] = {
            "query_id": key[0],
            "query": row["query"],
            "category": row.get("category", ""),
            "episode_id": key[1],
            "episode_title": row.get("episode_title", ""),
        }

fields = [
    "query_id", "query", "category",
    "episode_id", "episode_title", "relevant"
]

missing = 0
with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for key, pair in sorted(pairs.items()):
        value = labels.get(key, "")
        if value == "":
            missing += 1
        writer.writerow({**pair, "relevant": value})

print(f"Trained DVMN ranking rows: {len(trained)}")
print(f"Keyword ranking rows: {len(keyword)}")
print(f"Unique query-episode pairs: {len(pairs)}")
print(f"Pairs still needing human labels: {missing}")
print(f"Saved: {OUTPUT}")