import csv
import os
import sqlite3

BASE_DIR = r"C:\Users\DELL\Downloads\DVMN 0.1"

DB_PATH = os.path.join(BASE_DIR, "dvmn_memory.db")
RANKINGS_PATH = os.path.join(BASE_DIR, "keyword_rankings.csv")
LABELS_PATH = os.path.join(BASE_DIR, "relevance_labels.csv")

LABEL_FIELDS = [
    "query_id",
    "query",
    "episode_id",
    "episode_title",
    "rank",
    "relevant",
]


def load_csv(path):
    if not os.path.exists(path):
        return []

    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def save_labels(labels):
    with open(LABELS_PATH, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=LABEL_FIELDS)
        writer.writeheader()

        for key in sorted(labels, key=lambda item: (
            int(item[0]), int(item[1])
        )):
            writer.writerow(labels[key])


def main():
    if not os.path.exists(DB_PATH):
        print("ERROR: Database not found:")
        print(DB_PATH)
        return

    if not os.path.exists(RANKINGS_PATH):
        print("ERROR: keyword_rankings.csv not found.")
        print("Run compare baseline.py first.")
        return

    rankings = load_csv(RANKINGS_PATH)
    if not rankings:
        print("No ranking rows found.")
        return

    # Load episode titles and full content from the existing database.
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        episodes = {
            str(row["id"]): {
                "title": row["title"],
                "content": row["content"],
            }
            for row in connection.execute(
                "SELECT id, title, content FROM episodes"
            )
        }
    except sqlite3.Error as error:
        print("Could not read episodes:", error)
        connection.close()
        return

    connection.close()

    # Keep one row per query/episode pair.
    candidates = {}
    for row in rankings:
        query_id = row.get("query_id", "").strip()
        episode_id = row.get("episode_id", "").strip()

        if not query_id or not episode_id:
            continue

        key = (query_id, episode_id)
        if key not in candidates:
            candidates[key] = row

    # Preserve labels already entered in an earlier session.
    labels = {}
    for row in load_csv(LABELS_PATH):
        query_id = row.get("query_id", "").strip()
        episode_id = row.get("episode_id", "").strip()
        relevant = row.get("relevant", "").strip()

        if (
            query_id
            and episode_id
            and relevant in {"0", "1"}
            and (query_id, episode_id) in candidates
        ):
            labels[(query_id, episode_id)] = {
                "query_id": query_id,
                "query": row.get("query", ""),
                "episode_id": episode_id,
                "episode_title": row.get("episode_title", ""),
                "rank": row.get("rank", ""),
                "relevant": relevant,
            }

    todo = [
        (key, row)
        for key, row in candidates.items()
        if key not in labels
    ]

    print("\nDVMN 0.5 — MANUAL RELEVANCE LABELLER")
    print("=" * 48)
    print(f"Unique candidate pairs: {len(candidates)}")
    print(f"Already labelled:       {len(labels)}")
    print(f"Remaining:              {len(todo)}")
    print("\nFor each pair, decide whether the episode is genuinely")
    print("relevant to answering or recalling the query.")
    print("\n  1 = Relevant")
    print("  0 = Not relevant")
    print("  s = Skip for now")
    print("  q = Save and quit")
    print("=" * 48)

    for index, (key, row) in enumerate(todo, start=1):
        query_id, episode_id = key

        episode = episodes.get(episode_id)
        if episode is None:
            print(f"\nWARNING: Episode ID {episode_id} is not in the database.")
            continue

        query = row.get("query", "")
        title = episode["title"]
        content = episode["content"] or ""

        print("\n" + "-" * 65)
        print(f"Unlabelled pair {index} of {len(todo)}")
        print(f"Query ID:   {query_id}")
        print(f"Query:      {query}")
        print(f"Episode ID: {episode_id}")
        print(f"Title:      {title}")
        print(f"Keyword rank: {row.get('rank', '')}")
        print(f"Keyword score: {row.get('keyword_score', '')}")
        print("\nEPISODE CONTENT:")
        print(content[:2000])

        if len(content) > 2000:
            print("\n[Content shortened for display.]")

        while True:
            choice = input(
                "\nRelevant? [1=yes, 0=no, s=skip, q=quit]: "
            ).strip().lower()

            if choice in {"1", "0"}:
                labels[key] = {
                    "query_id": query_id,
                    "query": query,
                    "episode_id": episode_id,
                    "episode_title": title,
                    "rank": row.get("rank", ""),
                    "relevant": choice,
                }
                save_labels(labels)
                print(f"Saved label: {choice}")
                print(f"Progress: {len(labels)}/{len(candidates)} pairs labelled")
                break

            elif choice == "s":
                print("Skipped. You can label this pair next time.")
                break

            elif choice == "q":
                save_labels(labels)
                print("\nProgress saved.")
                print(f"Labelled {len(labels)} of {len(candidates)} pairs.")
                print("Run this script again to continue.")
                return

            else:
                print("Enter 1, 0, s, or q.")

    save_labels(labels)

    print("\n" + "=" * 48)
    print("LABELLING SESSION FINISHED")
    print(f"Labels saved to: {LABELS_PATH}")
    print(f"Labelled pairs: {len(labels)} of {len(candidates)}")
    print("=" * 48)


if __name__ == "__main__":
    main()