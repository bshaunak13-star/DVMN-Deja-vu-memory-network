import argparse
import csv
import sqlite3
from pathlib import Path
import main4 as dvmn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="dvmn_train.db")
    parser.add_argument("--model", default="dvmn05_train_model.json")
    parser.add_argument("--labels", default="test_labels.csv")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    for filename in (args.db, args.model, args.labels):
        if not Path(filename).exists():
            raise SystemExit(f"File not found: {filename}")

    # Load held-out human labels.
    labels = {}
    with open(args.labels, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"query_id", "query", "episode_id", "relevant"}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(f"Labels need columns: {sorted(required)}")

        for row in reader:
            relevant = row["relevant"].strip()
            if relevant in ("0", "1"):
                key = (row["query_id"].strip(),
                       row["episode_id"].strip())
                labels[key] = (row["query"].strip(), int(relevant))

    # Open database read-only.
    db_uri = Path(args.db).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
       episodes = dvmn.all_episodes(conn)
    finally:
        conn.close()

    model = dvmn.model_load(args.model)
    graph = dvmn.make_graph(episodes)

    # Rank every episode for each query appearing in the test labels.
    query_map = {}
    for (query_id, episode_id), (query, relevant) in labels.items():
        query_map[query_id] = query

    predicted = set()
    for query_id, query in query_map.items():
        scored = []
        for episode in episodes:
            features = dvmn.feature_vector(query, episode, graph)
            score = dvmn.score_features(features, model)
            scored.append((score, str(episode["id"])))

        scored.sort(key=lambda item: item[0], reverse=True)
        for score, episode_id in scored[:args.k]:
            predicted.add((query_id, episode_id))

    tp = fp = tn = fn = 0
    for pair, (_, relevant) in labels.items():
        hit = pair in predicted
        if hit and relevant:
            tp += 1
        elif hit and not relevant:
            fp += 1
        elif not hit and not relevant:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print("\nDVMN 0.5 — TOP-K HOLDOUT EVALUATION")
    print("-----------------------------------")
    print(f"Human-labelled pairs: {len(labels)}")
    print(f"Test queries:         {len(query_map)}")
    print(f"Episodes ranked:      {len(episodes)}")
    print(f"Top-K cutoff:         {args.k}")
    print(f"TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")
    print("\nDatabase and labels were not modified.")


if __name__ == "__main__":
    main()