import sqlite3
from pathlib import Path

for name, p in [
    ("game_store", Path("training/data/canonical/game_store.db")),
    ("teacher", Path("training/data/canonical/position_teacher_store.db")),
]:
    if not p.is_file():
        print(name, "missing")
        continue
    c = sqlite3.connect(p)
    print("===", name, "===")
    for row in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"):
        print(" ", row[0])
    for q in ("SELECT COUNT(*) FROM games", "SELECT COUNT(*) FROM positions", "SELECT COUNT(*) FROM edges"):
        try:
            print(q, c.execute(q).fetchone()[0])
        except Exception as e:
            print(q, e)
