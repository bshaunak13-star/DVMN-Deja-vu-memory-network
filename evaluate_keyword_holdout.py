import csv
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", default="keyword_rankings.csv")
    parser.add_argument("--labels", default="test_labels.csv")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    rankings_path = Path(args.rankings)
    labels_path = Path(args.labels)

    if not rankings_path.exists():
        raise SystemExit(f"Rankings file not found: {rankings_path}")
    if not labels_path.exists():
        raise SystemExit(f"Labels file not found: {labels_path}")

    # Read human relevance labels.
    labels = {}
    with labels_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"query_id", "episode_id", "relevant"}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(
                "test_labels.csv must contain: query_id, episode_id, relevant"
            )

        for row in reader:
            query_id = row["query_id"].strip()
            episode_id = row["episode_id"].strip()
            relevant = row["relevant"].strip()

            if relevant not in {"0", "1"}:
                continue

            labels[(query_id, episode_id)] = int(relevant)

    # Read keyword baseline rankings.
    rankings = []
    with rankings_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                query_id = row[0].strip()
                episode_id = row[3].strip()
                rank = int(row[2])
                score = float(row[5])
            except (ValueError, IndexError):
                continue

            rankings.append((query_id, episode_id, rank, score))

    # Evaluate the labelled query-episode pairs.
    # A pair is predicted relevant if it appears in the top K.
    predicted = set()
    for query_id, episode_id, rank, score in rankings:
        if rank <= args.k:
            predicted.add((query_id, episode_id))

    tp = fp = tn = fn = 0

    for pair, relevant in labels.items():
        is_predicted = pair in predicted

        if is_predicted and relevant == 1:
            tp += 1
        elif is_predicted and relevant == 0:
            fp += 1
        elif not is_predicted and relevant == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    print("\nKEYWORD BASELINE — HELD-OUT EVALUATION")
    print("-------------------------------------")
    print(f"Human-labelled pairs: {len(labels)}")
    print(f"Queries with rankings: {len(set(q for q, _, _, _ in rankings))}")
    print(f"Top-K cutoff:         {args.k}")
    print(f"TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")

    print("\nNote: this evaluates top-K retrieval against the labelled pairs.")
    print("Pairs absent from the rankings are treated as not retrieved.")
    print("Interpret results cautiously if the test set is small.")

if __name__ == "__main__":
    main()