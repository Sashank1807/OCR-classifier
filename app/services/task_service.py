import time
import uuid
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.logger import logger
from app.models.document import DocumentRecord, DocumentResult
from app.services.ocr_pipeline import pipeline

# Multi-worker thread pool for parallel multi-user background document processing
executor = ThreadPoolExecutor(max_workers=4)


class TaskService:
    """Asynchronous background job execution service for OCR pipeline."""

    def submit_async_job(
        self,
        file_path: Path,
        original_filename: str,
        project_name: str = "DefaultProject",
        user_id: str = "system",
        requested_doc_type: str = "AUTO"
    ) -> str:
        """
        Creates an initial PROCESSING database record with a unique request_id
        and submits the file to the background worker thread pool.
        Returns request_id immediately.
        """
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        db = SessionLocal()
        try:
            stored_filename = file_path.name
            existing_rec = db.query(DocumentRecord).filter(DocumentRecord.stored_filename == stored_filename).first()
            if existing_rec:
                stored_filename = f"{uuid.uuid4().hex[:8]}_{file_path.name}"

            file_type = file_path.suffix.lstrip(".").lower()
            file_size = file_path.stat().st_size if file_path.exists() else 0

            # Create initial PROCESSING record
            record = DocumentRecord(
                request_id=request_id,
                project_name=project_name,
                department=project_name,
                user_id=user_id,
                original_filename=original_filename,
                stored_filename=stored_filename,
                file_path=str(file_path),
                file_type=file_type,
                file_size=file_size,
                document_type=requested_doc_type.upper(),
                status="PROCESSING",
                created_at=datetime.utcnow()
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            doc_db_id = record.id
            db.close()

            # Dispatch background worker task
            executor.submit(
                self._run_pipeline_background,
                doc_db_id,
                file_path,
                original_filename,
                requested_doc_type
            )
            logger.info(f"Submitted async OCR job. Request ID: {request_id}, Project: {project_name}, User: {user_id}")
            return request_id

        except Exception as e:
            db.close()
            logger.error(f"Failed to submit async OCR job: {str(e)}", exc_info=True)
            raise e

    def resume_job(
        self,
        doc_db_id: int,
        file_path: Path,
        original_filename: str,
        requested_doc_type: str = "AUTO",
    ) -> None:
        """
        Put an existing record back on the worker pool.

        Used by startup recovery for work a previous process died holding. It
        reuses the document id rather than creating a new one, so the caller's
        original request_id keeps resolving to the same document.
        """
        executor.submit(
            self._run_pipeline_background,
            doc_db_id,
            file_path,
            original_filename,
            requested_doc_type,
        )
        logger.info(f"Re-queued interrupted document ID {doc_db_id} for processing.")

    def _run_pipeline_background(
        self,
        doc_db_id: int,
        file_path: Path,
        original_filename: str,
        requested_doc_type: str
    ):
        """Worker thread executing the full Qwen vision pipeline."""
        db = SessionLocal()
        try:
            logger.info(f"Background worker started processing doc ID: {doc_db_id}")
            pipeline.process_file(
                db=db,
                file_path=file_path,
                original_filename=original_filename,
                requested_doc_type=requested_doc_type,
                existing_doc_id=doc_db_id
            )
            logger.info(f"Background worker completed processing doc ID: {doc_db_id}")
        except Exception as e:
            logger.error(f"Background worker failed for doc ID {doc_db_id}: {str(e)}", exc_info=True)
            rec = db.query(DocumentRecord).filter(DocumentRecord.id == doc_db_id).first()
            if rec:
                rec.status = "FAILED"
                rec.error_message = str(e)
                rec.completed_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()


task_service = TaskService()
