import csv
import re
import sqlite3
from pathlib import Path

DB = Path("dvmn_memory.db")
QUERIES = Path("final_eval_queries.csv")
OUTPUT = Path("final_keyword_rankings.csv")
TOP_K = 5

for path in (DB, QUERIES):
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

if OUTPUT.exists():
    raise SystemExit(
        f"{OUTPUT} already exists. Rename or back it up before rerunning."
    )

def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))

with QUERIES.open("r", encoding="utf-8-sig", newline="") as f:
    queries = list(csv.DictReader(f))

conn = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
conn.row_factory = sqlite3.Row

try:
    episodes = [
        dict(row)
        for row in conn.execute("SELECT * FROM episodes ORDER BY id")
    ]
finally:
    conn.close()

def keyword_score(query, episode):
    query_words = tokenize(query)
    if not query_words:
        return 0.0

    title_words = tokenize(episode.get("title", ""))
    other_text = " ".join(
        str(episode.get(field, "") or "")
        for field in ("content", "text", "summary", "concepts")
    )
    content_words = tokenize(other_text)

    title_overlap = len(query_words & title_words) / len(query_words)
    content_overlap = len(query_words & content_words) / len(query_words)

    return title_overlap + 0.5 * content_overlap

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "query_id", "query", "category", "rank",
        "episode_id", "episode_title", "keyword_score"
    ])

    for item in queries:
        ranked = [
            (keyword_score(item["query"], ep), ep)
            for ep in episodes
        ]
        ranked.sort(
            key=lambda pair: (
                pair[0],
                str(pair[1].get("id", ""))
            ),
            reverse=True
        )

        for rank, (score, ep) in enumerate(ranked[:TOP_K], start=1):
            writer.writerow([
                item["query_id"],
                item["query"],
                item.get("category", ""),
                rank,
                ep.get("id", ""),
                ep.get("title", ""),
                round(score, 6)
            ])

print(f"Queries processed: {len(queries)}")
print(f"Episodes in database: {len(episodes)}")
print(f"Saved top-{TOP_K} rankings to {OUTPUT}")