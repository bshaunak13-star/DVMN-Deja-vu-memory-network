import csv
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

SOURCE_DB = Path("dvmn_memory.db")
TRAIN_CSV = Path("train_labels.csv")
TRAIN_DB = Path("dvmn_train.db")

if not SOURCE_DB.exists():
    raise SystemExit(f"Missing database: {SOURCE_DB}")

if not TRAIN_CSV.exists():
    raise SystemExit(f"Missing training CSV: {TRAIN_CSV}")

# Never overwrite an existing training database accidentally.
if TRAIN_DB.exists():
    raise SystemExit(
        f"{TRAIN_DB} already exists. Rename or back it up first."
    )

with TRAIN_CSV.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    required = {"query", "episode_id", "relevant"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise SystemExit(
            f"CSV must contain columns: {sorted(required)}"
        )
    rows = list(reader)

valid_rows = []
for line_number, row in enumerate(rows, start=2):
    try:
        query = row["query"].strip()
        episode_id = int(row["episode_id"])
        label = int(row["relevant"])
        if not query or label not in (0, 1):
            raise ValueError
        valid_rows.append((query, episode_id, label))
    except (ValueError, TypeError, AttributeError):
        raise SystemExit(f"Invalid training CSV row: {line_number}")

if not valid_rows:
    raise SystemExit("No valid training labels found.")

# Copy the original database; all episodes remain intact.
shutil.copy2(SOURCE_DB, TRAIN_DB)

conn = sqlite3.connect(TRAIN_DB)
try:
    conn.execute("PRAGMA foreign_keys = ON")

    # Check the episode IDs before changing the copied database.
    known_ids = {
        row[0] for row in conn.execute("SELECT id FROM episodes")
    }
    missing_ids = sorted({
        episode_id for _, episode_id, _ in valid_rows
        if episode_id not in known_ids
    })
    if missing_ids:
        raise SystemExit(
            f"Episode IDs not found in database: {missing_ids}"
        )

    # Clear feedback only in the COPY, never in the original DB.
    conn.execute("DELETE FROM feedback")

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    conn.executemany(
        """
        INSERT INTO feedback (query, episode_id, label, created_at)
        VALUES (?, ?, ?, ?)
        """,
        [
            (query, episode_id, label, timestamp)
            for query, episode_id, label in valid_rows
        ],
    )
    conn.commit()

    total = conn.execute(
        "SELECT COUNT(*) FROM feedback"
    ).fetchone()[0]
    positives, negatives = conn.execute(
        "SELECT "
        "SUM(CASE WHEN label=1 THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN label=0 THEN 1 ELSE 0 END) "
        "FROM feedback"
    ).fetchone()

    print("TRAINING DATABASE CREATED")
    print(f"Database: {TRAIN_DB.resolve()}")
    print(f"Imported training labels: {total}")
    print(f"Relevant: {positives}")
    print(f"Irrelevant: {negatives}")
    print("Original database was not modified.")
finally:
    conn.close()