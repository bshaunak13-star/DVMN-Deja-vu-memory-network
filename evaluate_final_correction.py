import csv
from pathlib import Path

LABELS = "trained_comparison_labels.csv"
SYSTEMS = {
    "Trained DVMN": "final_dvmn_trained_rankings.csv",
    "Keyword baseline": "final_keyword_rankings.csv",
}

def load_csv(filename):
    with open(filename, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def key(row):
    return (str(row["query_id"]).strip(),
            str(row["episode_id"]).strip())

def main():
    for filename in [LABELS, *SYSTEMS.values()]:
        if not Path(filename).exists():
            raise SystemExit(f"Missing file: {filename}")

    labels = load_csv(LABELS)
    relevant = {}

    for row in labels:
        value = row.get("relevant", "").strip()
        if value not in ("0", "1"):
            raise SystemExit(
                f"Blank or invalid relevance label in {LABELS}: {row}"
            )
        relevant[key(row)] = int(value)

    if not relevant:
        raise SystemExit("No labelled pairs found.")

    all_queries = sorted({q for q, _ in relevant})
    print(f"Labelled pairs: {len(relevant)}")
    print(f"Queries: {len(all_queries)}")
    print(f"Relevant labelled pairs: {sum(relevant.values())}")
    print("\nCorrected evaluation (top 5)")
    print("Recall denominator: all labelled candidates per query.")
    print("")

    for name, filename in SYSTEMS.items():
        rows = load_csv(filename)
        ranked = {}

        for row in rows:
            qid, eid = key(row)
            if (qid, eid) not in relevant:
                raise SystemExit(
                    f"Missing human label for {name}: query {qid}, "
                    f"episode {eid}. Label every ranked candidate first."
                )
            ranked.setdefault(qid, []).append(row)

        tp = fp = fn = 0
        recalls = []
        reciprocal_ranks = []
        evaluated = 0

        for qid in all_queries:
            candidates = [
                (eid, label)
                for (q, eid), label in relevant.items()
                if q == qid
            ]
            total_relevant = sum(label for _, label in candidates)

            if qid not in ranked:
                retrieved = []
            else:
                ordered = sorted(
                    ranked[qid],
                    key=lambda r: int(r.get("rank", "999"))
                )
                retrieved = ordered[:5]

            hits = [
                row for row in retrieved
                if relevant[key(row)] == 1
            ]
            false_positives = sum(
                relevant[key(row)] == 0 for row in retrieved
            )

            tp += len(hits)
            fp += false_positives
            fn += total_relevant - len(hits)

            if total_relevant:
                recalls.append(len(hits) / total_relevant)
                evaluated += 1

            first_hit = next(
                (int(row.get("rank", i + 1))
                 for i, row in enumerate(retrieved)
                 if relevant[key(row)] == 1),
                None
            )
            reciprocal_ranks.append(
                1 / first_hit if first_hit else 0
            )

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        print(name)
        print(f"  Queries evaluated: {len(all_queries)}")
        print(f"  TP: {tp}  FP: {fp}  FN: {fn}")
        print(f"  Precision@5: {precision:.3f}")
        print(f"  Recall@5:    {recall:.3f}")
        print(f"  F1@5:        {f1:.3f}")
        print(
            f"  Mean Recall@5: "
            f"{sum(recalls) / len(recalls):.3f}"
            if recalls else "  Mean Recall@5: N/A"
        )
        print(
            f"  MRR: {sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}"
            if reciprocal_ranks else "  MRR: N/A"
        )
        print()

    print(
        "Note: recall measures the fraction of relevant human-labelled "
        "candidates retrieved from the judged candidate pool, not from "
        "every episode in the database."
    )

if __name__ == "__main__":
    main()