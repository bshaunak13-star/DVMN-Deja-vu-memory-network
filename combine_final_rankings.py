import csv
from pathlib import Path

INPUTS = {
    "DVMN": Path("final_dvmn_rankings.csv"),
    "Keyword": Path("final_keyword_rankings.csv"),
}
OUTPUT = Path("final_relevance_labels.csv")

for path in INPUTS.values():
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

if OUTPUT.exists():
    raise SystemExit(
        f"{OUTPUT} already exists. Rename or back it up before rerunning."
    )

pairs = {}

for system, path in INPUTS.items():
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["query_id"], row["episode_id"])
            if key not in pairs:
                pairs[key] = {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "category": row.get("category", ""),
                    "episode_id": row["episode_id"],
                    "episode_title": row["episode_title"],
                    "dvmn_rank": "",
                    "keyword_rank": "",
                    "relevant": "",
                }

            rank_field = "dvmn_rank" if system == "DVMN" else "keyword_rank"
            pairs[key][rank_field] = row["rank"]

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    fields = [
        "query_id", "query", "category", "episode_id",
        "episode_title", "dvmn_rank", "keyword_rank", "relevant"
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(
        sorted(pairs.values(),
               key=lambda r: (r["query_id"], r["episode_id"]))
    )

print(f"Combined unique query-episode pairs: {len(pairs)}")
print(f"Saved to {OUTPUT}")
print("Fill relevant with 1 or 0; leave it blank until judged.")