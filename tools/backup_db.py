"""
Database backup.

    python tools/backup_db.py [--out backups] [--keep 14]

Works against whatever DATABASE_URL points at:

  * MySQL  - streams a logical dump through the driver. No mysqldump on PATH
             required, which matters on this Windows host.
  * SQLite - uses the online backup API. NEVER file-copy a live SQLite
             database: a copy taken mid-write is silently corrupt.

Schedule it with Task Scheduler (Windows) or cron. Backups are written with
timestamped names and pruned to --keep most recent.
"""

import argparse
import gzip
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402


def _prune(out_dir: Path, pattern: str, keep: int) -> None:
    files = sorted(out_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        old.unlink()
        print(f"  pruned {old.name}")


def _quote(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (bytes, bytearray)):
        return "0x" + v.hex()
    s = str(v).replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")
    return f"'{s}'"


def backup_mysql(url: str, out_dir: Path, keep: int) -> Path:
    import pymysql

    u = urlparse(url)
    db_name = u.path.lstrip("/").split("?")[0]
    conn = pymysql.connect(
        host=u.hostname or "127.0.0.1",
        port=u.port or 3306,
        user=unquote(u.username or ""),
        password=unquote(u.password or ""),
        database=db_name,
        charset="utf8mb4",
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = out_dir / f"{db_name}_{stamp}.sql.gz"

    with conn.cursor() as cur, gzip.open(target, "wt", encoding="utf-8") as fh:
        fh.write(f"-- {db_name} backup {datetime.now().isoformat()}\n")
        fh.write("SET NAMES utf8mb4;\nSET FOREIGN_KEY_CHECKS=0;\n\n")
        cur.execute("SHOW TABLES")
        tables = [r[0] for r in cur.fetchall()]
        for table in tables:
            cur.execute(f"SHOW CREATE TABLE `{table}`")
            fh.write(f"DROP TABLE IF EXISTS `{table}`;\n{cur.fetchone()[1]};\n\n")
            cur.execute(f"SELECT * FROM `{table}`")
            cols = [d[0] for d in cur.description]
            collist = ", ".join(f"`{c}`" for c in cols)
            batch = []
            for row in cur:
                batch.append("(" + ", ".join(_quote(v) for v in row) + ")")
                if len(batch) >= 200:
                    fh.write(f"INSERT INTO `{table}` ({collist}) VALUES\n" + ",\n".join(batch) + ";\n")
                    batch = []
            if batch:
                fh.write(f"INSERT INTO `{table}` ({collist}) VALUES\n" + ",\n".join(batch) + ";\n")
            fh.write("\n")
        fh.write("SET FOREIGN_KEY_CHECKS=1;\n")
    conn.close()
    _prune(out_dir, "*.sql.gz", keep)
    return target


def backup_sqlite(url: str, out_dir: Path, keep: int) -> Path:
    src = Path(url.split("sqlite:///")[-1]).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = out_dir / f"{src.stem}_{stamp}.db"
    source = sqlite3.connect(str(src))
    dest = sqlite3.connect(str(target))
    with dest:
        source.backup(dest)     # consistent snapshot even with writers active
    dest.close()
    source.close()
    _prune(out_dir, "*.db", keep)
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="backups")
    ap.add_argument("--keep", type=int, default=14)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = settings.DATABASE_URL

    if url.startswith("mysql"):
        path = backup_mysql(url, out_dir, args.keep)
    elif url.startswith("sqlite"):
        path = backup_sqlite(url, out_dir, args.keep)
    else:
        print(f"Unsupported DATABASE_URL scheme: {url.split('://')[0]}")
        return 1

    print(f"  wrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")
    print(f"  keeping the {args.keep} most recent in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
