import csv
from pathlib import Path
from collections import defaultdict

LABELS = Path("final_relevance_labels.csv")
DVMN_FILE = Path("final_dvmn_rankings.csv")
KEYWORD_FILE = Path("final_keyword_rankings.csv")

for path in (LABELS, DVMN_FILE, KEYWORD_FILE):
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

labels = read_csv(LABELS)
dvmn_rows = read_csv(DVMN_FILE)
keyword_rows = read_csv(KEYWORD_FILE)

# Require every candidate pair to have a valid human label.
judgments = {}
for row in labels:
    key = (row["query_id"].strip(), row["episode_id"].strip())
    value = row.get("relevant", "").strip()

    if value not in ("0", "1"):
        raise SystemExit(
            f"Missing/invalid label for query {key[0]}, "
            f"episode {key[1]}. Fill column H with 0 or 1."
        )

    if key in judgments:
        raise SystemExit(f"Duplicate query-episode pair: {key}")

    judgments[key] = int(value)

def evaluate(rows, system_name):
    by_query = defaultdict(list)

    for row in rows:
        qid = row["query_id"].strip()
        eid = row["episode_id"].strip()
        key = (qid, eid)

        if key not in judgments:
            raise SystemExit(
                f"No human label for {system_name} pair: {key}"
            )

        by_query[qid].append((int(row["rank"]), judgments[key]))

    tp = fp = fn = 0
    recall_at_5 = []
    reciprocal_ranks = []

    for qid, ranked in by_query.items():
        ranked.sort(key=lambda item: item[0])
        relevant_total = sum(label for _, label in ranked)
        relevant_retrieved = sum(
            label for rank, label in ranked if rank <= 5
        )

        tp += relevant_retrieved
        fp += sum(
            1 for rank, label in ranked
            if rank <= 5 and label == 0
        )
        fn += max(0, relevant_total - relevant_retrieved)

        recall_at_5.append(
            relevant_retrieved / relevant_total
            if relevant_total else None
        )

        first_relevant_rank = next(
            (rank for rank, label in ranked if label == 1),
            None
        )
        reciprocal_ranks.append(
            1 / first_relevant_rank if first_relevant_rank else 0
        )

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    valid_recalls = [x for x in recall_at_5 if x is not None]

    print(f"\n{system_name}")
    print(f"Queries evaluated: {len(by_query)}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision@5: {precision:.3f}")
    print(f"Recall@5: {recall:.3f}")
    print(f"F1@5: {f1:.3f}")
    print(
        "Mean Recall@5:",
        f"{sum(valid_recalls) / len(valid_recalls):.3f}"
        if valid_recalls else "N/A"
    )
    print(
        "MRR:",
        f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}"
        if reciprocal_ranks else "N/A"
    )

print(f"Human-labelled pairs: {len(judgments)}")
print(f"Relevant labels: {sum(judgments.values())}")
print(f"Irrelevant labels: {len(judgments) - sum(judgments.values())}")

evaluate(dvmn_rows, "DVMN 0.5")
evaluate(keyword_rows, "Keyword baseline")

print("\nNote: results are preliminary and depend on the human labels.")