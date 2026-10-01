"""
Startup recovery and retention.

Two problems that only show up once the thing has been running a while:

1. Async jobs live in an in-process ThreadPoolExecutor. A restart - a crash, a
   deploy, or (until now) autoreload firing on a touched file - kills every
   in-flight job silently. The row stays PROCESSING forever, so a caller
   polling /status waits on a document nobody is working on. Nothing ever
   reconciled that.

2. Nothing deleted anything, ever. outputs/ reached 1.8GB and uploads/ 172MB.
   For ID documents that is not only a disk problem: keeping an Aadhaar scan
   indefinitely because no one wrote a cleanup is a privacy decision made by
   accident.
"""

import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.logger import logger


MAX_PROCESSING_ATTEMPTS = 3


def recover_stranded_jobs() -> int:
    """
    Re-queue work a previous process died holding, rather than losing it.

    Called once at startup, before serving. A document whose source file still
    exists is resubmitted; one that has already failed MAX_PROCESSING_ATTEMPTS
    times, or whose upload is gone, is marked FAILED so a caller polling
    /status gets a terminal answer instead of waiting forever.

    The attempt counter is the important half: without it a document that
    crashes the pipeline would be retried on every boot, and every restart
    would crash again on the same file.
    """
    from app.models.document import DocumentRecord
    from app.services.task_service import task_service

    db = SessionLocal()
    requeued = failed = 0
    try:
        stranded = (
            db.query(DocumentRecord)
            .filter(DocumentRecord.status.like("PROCESSING%"))
            .all()
        )
        if not stranded:
            return 0

        for rec in stranded:
            attempts = (rec.processing_attempts or 0) + 1
            rec.processing_attempts = attempts
            source = Path(rec.file_path) if rec.file_path else None

            if attempts > MAX_PROCESSING_ATTEMPTS or not source or not source.exists():
                reason = ("exceeded %d processing attempts" % MAX_PROCESSING_ATTEMPTS
                          if attempts > MAX_PROCESSING_ATTEMPTS
                          else "the uploaded file is no longer on disk")
                rec.status = "FAILED"
                rec.error_message = (
                    f"Processing was interrupted by a service restart and could not be "
                    f"resumed ({reason}). Re-submit the document."
                )
                rec.completed_at = datetime.utcnow()
                failed += 1
            else:
                requeued += 1
        db.commit()

        # Resubmit only after the commit, so a worker cannot pick a row up
        # before its attempt count is persisted.
        for rec in stranded:
            if rec.status != "FAILED":
                try:
                    task_service.resume_job(
                        doc_db_id=rec.id,
                        file_path=Path(rec.file_path),
                        original_filename=rec.original_filename,
                        requested_doc_type=rec.document_type or "AUTO",
                    )
                except Exception as e:
                    logger.error(f"Could not re-queue document {rec.id}: {e}")

        logger.warning(
            f"Startup recovery: {requeued} job(s) re-queued, {failed} marked FAILED "
            f"(interrupted by a previous process)"
        )
        return requeued + failed
    except Exception as e:
        db.rollback()
        logger.error(f"Stranded-job recovery failed: {e}", exc_info=True)
        return 0
    finally:
        db.close()


def _sweep_directory(directory: Path, cutoff: float) -> tuple:
    removed = freed = 0
    if not directory.exists():
        return 0, 0
    for entry in directory.iterdir():
        try:
            if not entry.is_file() or entry.stat().st_mtime >= cutoff:
                continue
            size = entry.stat().st_size
            entry.unlink()
            removed += 1
            freed += size
        except OSError as e:
            # A file being read by a request, or already gone. Not fatal.
            logger.debug(f"Retention: could not remove {entry.name}: {e}")
    return removed, freed


def run_retention_sweep() -> dict:
    """
    Delete uploads, renders and DB rows past the retention window.

    Database rows go first so the UI never points at a file that has been
    deleted underneath it.
    """
    from app.models.document import DocumentRecord

    days = settings.RETENTION_DAYS
    if days <= 0:
        return {"skipped": "RETENTION_DAYS <= 0"}

    cutoff_dt = datetime.utcnow() - timedelta(days=days)
    cutoff_ts = time.time() - days * 86400
    summary = {"days": days, "records": 0, "files": 0, "bytes": 0}

    db = SessionLocal()
    try:
        old = db.query(DocumentRecord).filter(DocumentRecord.created_at < cutoff_dt).all()
        for rec in old:
            db.delete(rec)          # cascades to pages and results
        if old:
            db.commit()
        summary["records"] = len(old)
    except Exception as e:
        db.rollback()
        logger.error(f"Retention: DB sweep failed: {e}", exc_info=True)
    finally:
        db.close()

    for d in (settings.UPLOAD_DIR, settings.OUTPUT_DIR):
        n, b = _sweep_directory(Path(d), cutoff_ts)
        summary["files"] += n
        summary["bytes"] += b

    if summary["records"] or summary["files"]:
        logger.info(
            f"Retention sweep: removed {summary['records']} record(s), "
            f"{summary['files']} file(s), {summary['bytes'] / 1e6:.1f} MB "
            f"older than {days} days"
        )
    return summary


def start_retention_thread() -> None:
    """Run the sweep on an interval for the life of the process."""
    if not settings.ENABLE_RETENTION_SWEEP:
        logger.info("Retention sweep disabled (ENABLE_RETENTION_SWEEP=false)")
        return

    interval = max(1, settings.RETENTION_SWEEP_HOURS) * 3600

    def _loop():
        # Let startup finish before touching the disk.
        time.sleep(60)
        while True:
            try:
                run_retention_sweep()
            except Exception as e:
                logger.error(f"Retention sweep raised: {e}", exc_info=True)
            time.sleep(interval)

    threading.Thread(target=_loop, daemon=True, name="retention-sweep").start()
    logger.info(
        f"Retention sweep scheduled every {settings.RETENTION_SWEEP_HOURS}h "
        f"(keeping {settings.RETENTION_DAYS} days)"
    )
