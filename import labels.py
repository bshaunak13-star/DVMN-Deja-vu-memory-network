import csv
import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(r"C:\Users\DELL\Downloads\DVMN 0.1")
DB_PATH = BASE_DIR / "dvmn_memory.db"
CSV_PATH = BASE_DIR / "relevance_labels.csv"


def main():
    if not DB_PATH.exists():
        raise SystemExit(f"Database not found: {DB_PATH}")

    if not CSV_PATH.exists():
        raise SystemExit(f"Labels file not found: {CSV_PATH}")

    with CSV_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    required = {"query", "episode_id", "relevant"}
    if not rows or not required.issubset(rows[0].keys()):
        raise SystemExit(
            "CSV must contain query, episode_id, and relevant columns."
        )

    connection = sqlite3.connect(DB_PATH)

    added = 0
    duplicates = 0
    conflicts = 0
    invalid = 0

    try:
        with connection:
            for row in rows:
                query = (row.get("query") or "").strip()
                label_text = (row.get("relevant") or "").strip()

                try:
                    episode_id = int(row["episode_id"])
                except (ValueError, TypeError, KeyError):
                    invalid += 1
                    continue

                if not query or label_text not in {"0", "1"}:
                    invalid += 1
                    continue

                label = int(label_text)

                episode = connection.execute(
                    "SELECT id FROM episodes WHERE id = ?",
                    (episode_id,)
                ).fetchone()

                if episode is None:
                    print(f"Skipped missing episode ID: {episode_id}")
                    invalid += 1
                    continue

                existing = connection.execute(
                    """
                    SELECT label FROM feedback
                    WHERE query = ? AND episode_id = ?
                    """,
                    (query, episode_id)
                ).fetchall()

                if existing:
                    existing_labels = {
                        int(item[0]) for item in existing
                    }

                    if existing_labels == {label}:
                        duplicates += 1
                    else:
                        print(
                            "Conflict; preserved existing feedback:",
                            query, episode_id,
                            "existing labels =", existing_labels,
                            "CSV label =", label
                        )
                        conflicts += 1
                    continue

                connection.execute(
                    """
                    INSERT INTO feedback
                        (query, episode_id, label, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        query,
                        episode_id,
                        label,
                        datetime.now().isoformat(timespec="seconds"),
                    )
                )
                added += 1

        print("\nIMPORT COMPLETE")
        print("New feedback rows:", added)
        print("Duplicates skipped:", duplicates)
        print("Conflicts preserved:", conflicts)
        print("Invalid rows skipped:", invalid)

        print("\nFeedback table totals:")
        print(
            connection.execute(
                "SELECT COUNT(*) FROM feedback"
            ).fetchone()[0]
        )

        print(
            "Label distribution:",
            connection.execute(
                """
                SELECT label, COUNT(*)
                FROM feedback
                GROUP BY label
                ORDER BY label
                """
            ).fetchall()
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()