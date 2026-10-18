import csv
from pathlib import Path
from collections import defaultdict

LABELS = Path("trained_comparison_labels.csv")
DVMN = Path("final_dvmn_trained_rankings.csv")
KEYWORD = Path("final_keyword_rankings.csv")
TOP_K = 5

for path in (LABELS, DVMN, KEYWORD):
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

label_rows = read_csv(LABELS)
dvmn_rows = read_csv(DVMN)
keyword_rows = read_csv(KEYWORD)

judgments = {}
for row in label_rows:
    qid = row["query_id"].strip()
    eid = row["episode_id"].strip()
    value = row.get("relevant", "").strip()

    if value not in ("0", "1"):
        raise SystemExit(
            f"Missing/invalid relevance label for query {qid}, "
            f"episode {eid}. Fill it with 0 or 1."
        )

    key = (qid, eid)
    if key in judgments:
        raise SystemExit(f"Duplicate label for pair: {key}")
    judgments[key] = int(value)

def evaluate(rows, system_name):
    by_query = defaultdict(list)
    seen = set()

    for row in rows:
        qid = row["query_id"].strip()
        eid = row["episode_id"].strip()
        key = (qid, eid)

        if key in seen:
            raise SystemExit(f"Duplicate ranking for {system_name}: {key}")
        seen.add(key)

        if key not in judgments:
            raise SystemExit(
                f"No human label for {system_name}, query {qid}, "
                f"episode {eid}."
            )

        by_query[qid].append((int(row["rank"]), judgments[key]))

    tp = fp = fn = 0
    recalls = []
    reciprocal_ranks = []

    for qid, ranked in by_query.items():
        ranked.sort(key=lambda x: x[0])
        top = [(rank, label) for rank, label in ranked if rank <= TOP_K]

        relevant_retrieved = sum(label for _, label in top)
        irrelevant_retrieved = sum(1 for _, label in top if label == 0)

        # All labelled candidates in this evaluation union.
        relevant_total = sum(label for _, label in ranked)

        tp += relevant_retrieved
        fp += irrelevant_retrieved
        fn += max(0, relevant_total - relevant_retrieved)

        if relevant_total:
            recalls.append(relevant_retrieved / relevant_total)

        first_relevant = next(
            (rank for rank, label in top if label == 1), None
        )
        reciprocal_ranks.append(
            1 / first_relevant if first_relevant is not None else 0
        )

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print(f"\n{system_name}")
    print(f"Queries evaluated: {len(by_query)}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision@{TOP_K}: {precision:.3f}")
    print(f"Recall@{TOP_K}: {recall:.3f}")
    print(f"F1@{TOP_K}: {f1:.3f}")
    print(
        "Mean Recall@5:",
        f"{sum(recalls) / len(recalls):.3f}" if recalls else "N/A"
    )
    print(
        "MRR:",
        f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}"
        if reciprocal_ranks else "N/A"
    )

print(f"Human-labelled pairs: {len(judgments)}")
print(f"Relevant pairs: {sum(judgments.values())}")
print(f"Irrelevant pairs: {len(judgments) - sum(judgments.values())}")

evaluate(dvmn_rows, "Trained DVMN 0.5")
evaluate(keyword_rows, "Keyword baseline")

print("\nNote: results depend on the human labels and evaluation set.")