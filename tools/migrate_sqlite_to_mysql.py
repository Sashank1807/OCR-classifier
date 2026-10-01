"""
One-off copy of the SQLite database into the configured MySQL schema.

Re-runnable: it skips documents whose id already exists, so an interrupted run
can simply be started again.

    python tools/migrate_sqlite_to_mysql.py --source ocr_database.db [--dry-run]

The source file is never modified - keep it until you are satisfied the MySQL
copy is complete.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings                      # noqa: E402
from app.core.database import Base, SessionLocal, engine  # noqa: E402
from app.models.document import (                          # noqa: E402
    DocumentPage,
    DocumentRecord,
    DocumentResult,
)

TABLES = (
    ("document_records", DocumentRecord),
    ("document_pages", DocumentPage),
    ("document_results", DocumentResult),
)


def _rows(conn: sqlite3.Connection, table: str):
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
    except sqlite3.OperationalError:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ocr_database.db")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"Source not found: {src}")
        return 1

    target = settings.DATABASE_URL.split("@")[-1]
    print(f"Source : {src}")
    print(f"Target : {target}")
    if "sqlite" in settings.DATABASE_URL:
        print("DATABASE_URL still points at SQLite - nothing to migrate into.")
        return 1

    Base.metadata.create_all(bind=engine)
    conn = sqlite3.connect(str(src))
    db = SessionLocal()
    summary = {}

    try:
        for table, model in TABLES:
            rows = _rows(conn, table)
            existing = {r[0] for r in db.query(model.id).all()}
            fields = {c.name for c in model.__table__.columns}
            new = [r for r in rows if r.get("id") not in existing]

            if args.dry_run:
                summary[table] = f"{len(new)} to copy ({len(rows)} in source, {len(existing)} already present)"
                continue

            # Insert in batches: one flush per row is slow, and one flush for
            # 1000 rows makes a single bad row fail the whole migration.
            copied = 0
            for i in range(0, len(new), 200):
                chunk = new[i:i + 200]
                for raw in chunk:
                    db.add(model(**{k: v for k, v in raw.items() if k in fields}))
                try:
                    db.commit()
                    copied += len(chunk)
                except Exception as e:
                    db.rollback()
                    print(f"  ! batch at offset {i} failed ({e}); retrying row by row")
                    for raw in chunk:
                        try:
                            db.add(model(**{k: v for k, v in raw.items() if k in fields}))
                            db.commit()
                            copied += 1
                        except Exception as row_err:
                            db.rollback()
                            print(f"    skipped {table} id={raw.get('id')}: {row_err}")
            summary[table] = f"copied {copied}/{len(new)} (source had {len(rows)})"

        print("\nResult")
        for k, v in summary.items():
            print(f"  {k:<20} {v}")

        if not args.dry_run:
            print("\nVerification")
            for table, model in TABLES:
                print(f"  {table:<20} source={len(_rows(conn, table)):<6} target={db.query(model).count()}")
            print("\nSQLite file left untouched. Keep it until you have verified the copy.")
        return 0
    finally:
        db.close()
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
