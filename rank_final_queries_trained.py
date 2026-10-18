import csv
import sqlite3
from pathlib import Path
import main4 as dvmn

DB = Path("dvmn_train.db")
MODEL = Path("dvmn05_train_model.json")
QUERIES = Path("final_eval_queries.csv")
OUTPUT = Path("final_dvmn_trained_rankings.csv")
TOP_K = 5

for path in (DB, MODEL, QUERIES):
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

if OUTPUT.exists():
    raise SystemExit(
        f"{OUTPUT} already exists. Back it up or rename it first."
    )

with QUERIES.open("r", encoding="utf-8-sig", newline="") as f:
    queries = list(csv.DictReader(f))

db_uri = DB.resolve().as_uri() + "?mode=ro"
conn = sqlite3.connect(db_uri, uri=True)
conn.row_factory = sqlite3.Row

try:
    episodes = dvmn.all_episodes(conn)
finally:
    conn.close()

model = dvmn.model_load(str(MODEL))
graph = dvmn.make_graph(episodes)

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "query_id", "query", "category", "rank",
        "episode_id", "episode_title", "dvmn_score"
    ])

    for row in queries:
        query_id = row["query_id"]
        query = row["query"]
        category = row["category"]
        scored = []

        for episode in episodes:
            features = dvmn.feature_vector(query, episode, graph)
            score = dvmn.score_features(features, model)
            scored.append((score, episode))

        scored.sort(key=lambda item: item[0], reverse=True)

        for rank, (score, episode) in enumerate(
            scored[:TOP_K], start=1
        ):
            writer.writerow([
                query_id,
                query,
                category,
                rank,
                episode["id"],
                episode.get("title", ""),
                round(float(score), 6),
            ])

print(f"Queries processed: {len(queries)}")
print(f"Episodes in database: {len(episodes)}")
print(f"Saved trained-model rankings to {OUTPUT}")