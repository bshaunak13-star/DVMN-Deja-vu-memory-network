import csv
import re
import sqlite3

DB = r"C:\Users\DELL\Downloads\DVMN 0.1\dvmn_memory.db"
INPUT_CSV = "eval_queries.csv"
OUTPUT_CSV = "keyword_rankings.csv"
K = 5


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def keyword_score(query, episode):
    query_words = tokens(query)
    episode_text = (
        str(episode["title"]) + " " + str(episode["content"])
    )
    episode_words = tokens(episode_text)

    if not query_words:
        return 0.0

    return len(query_words & episode_words) / len(query_words)


def main():
    with open(INPUT_CSV, encoding="utf-8-sig", newline="") as f:
        queries = list(csv.DictReader(f))

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row

    try:
        episodes = [
            dict(row) for row in con.execute(
                "SELECT id, title, content FROM episodes"
            ).fetchall()
        ]
    finally:
        con.close()

    if not episodes:
        print("No episodes found in database.")
        return

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig",
              newline="") as f:
        fields = [
            "query_id", "query", "rank",
            "episode_id", "episode_title", "keyword_score"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for item in queries:
            ranked = sorted(
                episodes,
                key=lambda ep: keyword_score(item["query"], ep),
                reverse=True,
            )

            for rank, ep in enumerate(ranked[:K], start=1):
                writer.writerow({
                    "query_id": item["query_id"],
                    "query": item["query"],
                    "rank": rank,
                    "episode_id": ep["id"],
                    "episode_title": ep["title"],
                    "keyword_score": round(
                        keyword_score(item["query"], ep), 4
                    ),
                })

    print(f"Queries processed: {len(queries)}")
    print(f"Episodes in database: {len(episodes)}")
    print(f"Saved top-{K} rankings to {OUTPUT_CSV}")
    print("Next: inspect rankings and attach relevance labels.")


if __name__ == "__main__":
    main()