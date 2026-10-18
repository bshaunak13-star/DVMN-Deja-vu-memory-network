import argparse
import csv
import sqlite3
from collections import Counter
from pathlib import Path

import main4 as dvmn


def metrics(labels, predictions):
    tp = fp = tn = fn = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DVMN on held-out human-labelled pairs."
    )
    parser.add_argument("--db", default="dvmn_train.db")
    parser.add_argument("--model", default="dvmn05_train_model.json")
    parser.add_argument("--labels", default="test_labels.csv")
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args()

    for filename in (args.db, args.model, args.labels):
        if not Path(filename).is_file():
            raise SystemExit(f"ERROR: File not found: {filename}")

    with open(args.labels, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("ERROR: The test-label CSV is empty.")

    required = {"query", "episode_id", "relevant"}
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(
            f"ERROR: CSV is missing columns: {', '.join(sorted(missing))}"
        )

    # Validate the labels before scoring.
    for row in rows:
        try:
            row["episode_id"] = int(row["episode_id"])
            row["relevant"] = int(row["relevant"])
        except (ValueError, TypeError):
            raise SystemExit(f"ERROR: Invalid episode ID or label: {row}")

        if row["relevant"] not in (0, 1):
            raise SystemExit(f"ERROR: relevant must be 0 or 1: {row}")

    # Read-only connection: this evaluator never writes to the database.
    db_path = str(Path(args.db).resolve())
    conn = sqlite3.connect(
        f"{Path(db_path).as_uri()}?mode=ro",
        uri=True
    )
    conn.row_factory = sqlite3.Row

    try:
        episodes = dvmn.all_episodes(conn)
    finally:
        conn.close()

    by_id = {e["id"]: e for e in episodes}
    model = dvmn.model_load(args.model)

    if not model.get("trained"):
        raise SystemExit(
            "ERROR: The selected model is not marked as trained. "
            "Check the model filename."
        )

    # Match the actual DVMN scoring implementation.
    graph = dvmn.make_graph(episodes)
    actual = []
    predicted = []
    missing_ids = []
    query_ids = set()

    for row in rows:
        episode = by_id.get(row["episode_id"])
        if episode is None:
            missing_ids.append(row["episode_id"])
            continue

        query = row["query"].strip()
        if not query:
            continue

        features = dvmn.feature_vector(query, episode, graph)
        score = dvmn.score_features(features, model)

        actual.append(row["relevant"])
        predicted.append(int(score >= args.threshold))

        if "query_id" in row and row["query_id"].strip():
            query_ids.add(row["query_id"].strip())
        else:
            query_ids.add(query)

    if missing_ids:
        raise SystemExit(
            "ERROR: Some test episode IDs are absent from the database: "
            + ", ".join(map(str, sorted(set(missing_ids))))
        )

    if not actual:
        raise SystemExit("ERROR: No valid test pairs were evaluated.")

    result = metrics(actual, predicted)

    print("\nDVMN 0.5 — HELD-OUT EVALUATION")
    print("=" * 42)
    print(f"Model:              {args.model}")
    print(f"Database:           {args.db}")
    print(f"Test file:          {args.labels}")
    print(f"Distinct test queries: {len(query_ids)}")
    print(f"Evaluated pairs:    {len(actual)}")
    print(f"Threshold:          {args.threshold:.3f}")
    print(f"Positive labels:    {sum(actual)}")
    print(f"Negative labels:    {len(actual) - sum(actual)}")
    print()
    print(f"TP:                 {result['TP']}")
    print(f"FP:                 {result['FP']}")
    print(f"TN:                 {result['TN']}")
    print(f"FN:                 {result['FN']}")
    print(f"Precision:          {result['precision']:.3f}")
    print(f"Recall:             {result['recall']:.3f}")
    print(f"F1 score:           {result['f1']:.3f}")

    print("\nInterpretation:")
    print("- Precision measures how many predicted relevant pairs were relevant.")
    print("- Recall measures how many labelled relevant pairs were found.")
    print("- F1 balances precision and recall.")
    print(
        "\nThese results apply to the labelled test pairs, "
        "not necessarily every episode in memory."
    )
    print(
        "This script does not import test labels or modify the database."
    )


if __name__ == "__main__":
    main()