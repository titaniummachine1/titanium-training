import sqlite3
from pathlib import Path

db = Path("training/data/canonical/position_teacher_store.db")
c = sqlite3.connect(db)
for row in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"):
    print(row[0])
for q in (
    "SELECT COUNT(*) FROM positions",
    "SELECT COUNT(*) FROM edges",
    "SELECT COUNT(*) FROM games",
    "SELECT COUNT(*) FROM labels WHERE label_type='teacher_value'",
):
    try:
        print(q, c.execute(q).fetchone()[0])
    except Exception as e:
        print(q, e)
