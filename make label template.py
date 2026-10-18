import csv
from pathlib import Path

RANKINGS = Path("keyword_rankings.csv")
LABELS = Path("relevance_labels.csv")

REQUIRED = {
    "query_id", "query", "episode_id", "episode_title", "rank"
}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    if not RANKINGS.exists():
        raise FileNotFoundError(
            "keyword_rankings.csv not found. Run "
            'python "compare baseline.py" first.'
        )

    rankings = read_csv(RANKINGS)
    if not rankings:
        raise ValueError("keyword_rankings.csv is empty.")

    with RANKINGS.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        columns = set(csv.DictReader(f).fieldnames or [])

    missing = REQUIRED - columns
    if missing:
        raise ValueError(
            "Missing columns in keyword_rankings.csv: "
            + ", ".join(sorted(missing))
        )

    # Preserve existing human labels when regenerating the template.
    old_labels = {}
    if LABELS.exists():
        for row in read_csv(LABELS):
            qid = row.get("query_id", "").strip()
            eid = row.get("episode_id", "").strip()
            label = row.get("relevant", "").strip()
            if qid and eid and label in {"0", "1"}:
                old_labels[(qid, eid)] = label

    output = {}
    for row in rankings:
        qid = row["query_id"].strip()
        eid = row["episode_id"].strip()
        query = row["query"].strip()
        title = row["episode_title"].strip()
        rank = row["rank"].strip()

        if not qid or not eid or not query:
            raise ValueError(f"Invalid ranking row: {row}")

        key = (qid, eid)
        output[key] = {
            "query_id": qid,
            "query": query,
            "episode_id": eid,
            "episode_title": title,
            "rank": rank,
            "relevant": old_labels.get(key, ""),
        }

    fields = [
        "query_id", "query", "episode_id",
        "episode_title", "rank", "relevant"
    ]

    with LABELS.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output.values())

    query_count = len({r["query_id"] for r in output.values()})
    labelled = sum(
        r["relevant"] in {"0", "1"} for r in output.values()
    )

    print("Created/updated relevance_labels.csv")
    print(f"Queries: {query_count}")
    print(f"Unique query-episode pairs: {len(output)}")
    print(f"Existing labels preserved: {labelled}")
    print("Blank relevant cells still need human judgement.")


if __name__ == "__main__":
    main()