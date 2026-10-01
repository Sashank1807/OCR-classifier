import uuid
from pathlib import Path
from typing import Optional, List, Union
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, File, Form, Header, UploadFile, HTTPException, Response, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from app.core.security import validate_file_upload
from app.core.logger import logger
from app.services.ocr_pipeline import pipeline
from app.services.export_service import export_service
from app.models.document import DocumentRecord, DocumentResult, DocumentPage
from app.core.auth import require_api_key
from app.services.erp_payload import erp_from_record, erp_from_pipeline_result

router = APIRouter(prefix="/api/v1/ocr", tags=["OCR Processing Engine"], dependencies=[Depends(require_api_key)])

# What a response carries. `full` is the reviewer's view - markdown, per-field
# verification, classification internals. `erp` is the business projection:
# named values and line items only, no diagnostics (app/services/erp_payload.py).
RESPONSE_FORMATS = ("full", "erp")


def _want_erp(response_format: Optional[str]) -> bool:
    """
    Resolve the requested response shape.

    An unrecognised value is rejected rather than silently treated as `full`:
    a caller who asks for `ERP_DATA` and gets the full envelope back would
    conclude the slim payload does not work, and go looking in the wrong place.
    """
    value = (response_format or "full").strip().lower()
    if value in ("erp", "erp_data", "business"):
        return True
    if value in ("full", "", "default", "complete"):
        return False
    raise HTTPException(
        status_code=400,
        detail=f"Unknown response_format '{response_format}'. Use one of: {', '.join(RESPONSE_FORMATS)}.",
    )


def _api_document(result_data: dict) -> dict:
    """
    Trim a pipeline result down to what the API publishes.

    `tables` repeated the table a consumer already has: its columns and rows
    are the same lists as structured_data.fields.columns / .rows (and, for
    spreadsheets, fields.all_tables), and its markdown is a slice of the
    document markdown. On a 20-row statement the same table was serialised
    four times in one response.

    Dropped here, at the API boundary, rather than in the pipeline - the
    pipeline's return value is also the internal interface the benchmark reads,
    and it still needs the tables list. The one field that lived only on
    `tables`, `column_model_uncertain`, is now published inside
    structured_data.
    """
    return {k: v for k, v in result_data.items() if k != "tables"}


@router.post("/process")
async def process_document(
    request: Request,
    file: Union[UploadFile, List[UploadFile], None] = File(None),
    files: Union[UploadFile, List[UploadFile], None] = File(None),
    document_type: str = Form("AUTO"),
    project_name: Optional[str] = Form("DefaultProject"),
    user_id: str = Form("system"),
    async_mode: bool = Form(False),
    response_format: str = Form("full"),
    x_project_name: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Upload and process single or multiple images / PDF documents.
    Executes full pipeline: preprocessing -> vision classification -> verbatim OCR -> structured extraction.
    Requires project_name parameter for microservice integration tracking (defaults to DefaultProject if omitted).

    Set `response_format=erp` to receive only the business data an ERP
    consumes - named field values and line items, without the markdown,
    verification evidence or classification internals. Same processing, a
    smaller and flatter response.
    """
    erp_only = _want_erp(response_format)
    # Extract project_name dynamically from any form key or header alias
    proj_name = ""
    try:
        form_data = await request.form()
        for key in ["project_name", "Project-name", "project-name", "Project_Name", "ProjectName", "project", "department"]:
            val = form_data.get(key)
            if val and isinstance(val, str) and val.strip():
                proj_name = val.strip()
                break
    except Exception:
        pass

    if not proj_name:
        proj_name = (x_project_name or project_name or "DefaultProject").strip()
    if not proj_name:
        proj_name = "DefaultProject"

    u_id = x_user_id or user_id

    # Collect all uploaded files (single or batch)
    upload_list: List[UploadFile] = []

    def extract_uploads(item):
        if not item:
            return
        if isinstance(item, list):
            for sub in item:
                if sub and hasattr(sub, "filename") and sub.filename:
                    upload_list.append(sub)
        elif hasattr(item, "filename") and item.filename:
            upload_list.append(item)

    extract_uploads(file)
    extract_uploads(files)

    # Deduplicate uploads by filename while preserving order.
    #
    # This guards against a client attaching the same file to both `file` and
    # `files`. But in a BULK ingest two genuinely different documents can
    # share a name ("scan.pdf" from two folders), and silently dropping one is
    # data loss the caller never sees - the batch just comes back smaller.
    # So the drops are counted, logged, and reported in the response.
    unique_uploads = []
    seen = set()
    skipped_duplicates = []
    for f in upload_list:
        if f.filename not in seen:
            seen.add(f.filename)
            unique_uploads.append(f)
        else:
            skipped_duplicates.append(f.filename)

    if skipped_duplicates:
        logger.warning(
            f"Ignored {len(skipped_duplicates)} upload(s) repeating a filename already "
            f"in this request: {sorted(set(skipped_duplicates))[:5]}. Give each document "
            f"a distinct filename if they are different documents."
        )

    if not unique_uploads:
        raise HTTPException(status_code=400, detail="No valid file uploaded in request.")

    try:
        processed_items = []
        for item_file in unique_uploads:
            content = await item_file.read()
            sanitized_name = validate_file_upload(item_file, content)

            unique_prefix = uuid.uuid4().hex[:8]
            saved_filename = f"{unique_prefix}_{sanitized_name}"
            saved_path = settings.UPLOAD_DIR / saved_filename

            with open(saved_path, "wb") as f:
                f.write(content)

            logger.info(f"File '{item_file.filename}' uploaded successfully for project '{proj_name}', user '{u_id}': {saved_path}")

            if async_mode:
                from app.services.task_service import task_service
                req_id = task_service.submit_async_job(
                    file_path=saved_path,
                    original_filename=item_file.filename,
                    project_name=proj_name,
                    user_id=u_id,
                    requested_doc_type=document_type
                )
                processed_items.append({
                    "filename": item_file.filename,
                    "request_id": req_id,
                    "status": "PROCESSING"
                })
            else:
                result_data = await run_in_threadpool(
                    pipeline.process_file,
                    db,
                    saved_path,
                    item_file.filename,
                    requested_doc_type=document_type
                )
                rec = db.query(DocumentRecord).filter(DocumentRecord.id == result_data["id"]).first()
                if rec:
                    rec.project_name = proj_name
                    rec.department = proj_name
                    rec.user_id = u_id
                    rec.request_id = f"req_{uuid.uuid4().hex[:12]}"
                    db.commit()
                    result_data["request_id"] = rec.request_id

                processed_items.append(
                    erp_from_pipeline_result(result_data, project_name=proj_name)
                    if erp_only else _api_document(result_data)
                )

        if len(processed_items) == 1:
            if async_mode:
                return {
                    "success": True,
                    "request_id": processed_items[0]["request_id"],
                    "status": "PROCESSING",
                    "message": "Document submitted for asynchronous OCR processing."
                }
            else:
                return {
                    "success": True,
                    "response_format": "erp" if erp_only else "full",
                    "data": processed_items[0]
                }
        else:
            if async_mode:
                resp = {
                    "success": True,
                    "total_documents": len(processed_items),
                    "batch_requests": processed_items,
                    "message": f"Successfully submitted {len(processed_items)} documents for asynchronous OCR processing."
                }
                if skipped_duplicates:
                    resp["skipped_duplicate_filenames"] = sorted(set(skipped_duplicates))
                    resp["message"] += (
                        f" {len(skipped_duplicates)} upload(s) were ignored for repeating "
                        f"a filename already in this request."
                    )
                return resp
            else:
                resp = {
                    "success": True,
                    "total_documents": len(processed_items),
                    "response_format": "erp" if erp_only else "full",
                    "data": processed_items
                }
                if skipped_duplicates:
                    resp["skipped_duplicate_filenames"] = sorted(set(skipped_duplicates))
                return resp

    except HTTPException as he:
        raise he
    except Exception as e:
        filenames = ", ".join(f.filename for f in unique_uploads) if 'unique_uploads' in locals() and unique_uploads else "upload"
        logger.error(f"Error processing upload '{filenames}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/status/{request_id}")
def get_job_status(request_id: str, response_format: str = "full", db: Session = Depends(get_db)):
    """
    Fetch status and extracted data for asynchronous request_id.

    `response_format=erp` swaps the `document` block for the ERP projection -
    named values and line items only. Progress and failure responses are
    unaffected; there is no extracted data to shape yet.
    """
    erp_only = _want_erp(response_format)
    record = db.query(DocumentRecord).filter(DocumentRecord.request_id == request_id).first()
    if not record and request_id.isdigit():
        # Callers reach for the document id here, because that is the id the
        # rest of the API is addressed by (/document/{id}, /export/{id}). A
        # request_id lookup that misses on an all-digit value is unambiguous -
        # every issued request_id is "req_<hex>" - so fall back rather than
        # returning a 404 that looks like the job was never submitted.
        record = db.query(DocumentRecord).filter(DocumentRecord.id == int(request_id)).first()
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for '{request_id}'. Pass the request_id "
                   f"returned by /process (req_...), or the numeric document id."
        )

    if record.status and record.status.startswith("PROCESSING"):
        pages = db.query(DocumentPage).filter(DocumentPage.document_id == record.id).all()
        completed_count = len(pages)
        total_pages = record.page_count or 1
        pct = int((completed_count / total_pages) * 100) if total_pages > 0 else 0

        return {
            "success": True,
            "request_id": request_id,
            "document_id": record.id,
            "status": record.status,
            "current_page": completed_count,
            "total_pages": total_pages,
            "percent": pct,
            "message": f"Processing Page {min(completed_count + 1, total_pages)} of {total_pages} ({pct}%)...",
            "project_name": record.project_name or record.department,
            "user_id": record.user_id,
            "created_at": record.created_at.isoformat()
        }
    elif record.status == "FAILED":
        return {
            "success": False,
            "request_id": request_id,
            "status": "FAILED",
            "error_message": record.error_message or "OCR processing failed.",
            "project_name": record.project_name or record.department,
            "user_id": record.user_id
        }

    result = db.query(DocumentResult).filter(DocumentResult.document_id == record.id).first()

    if erp_only:
        return {
            "success": True,
            "request_id": request_id,
            "status": "COMPLETED",
            "response_format": "erp",
            "data": erp_from_record(record, result)
        }

    return {
        "success": True,
        "request_id": request_id,
        "status": "COMPLETED",
        "project_name": record.project_name or record.department,
        "user_id": record.user_id,
        "document": {
            "id": record.id,
            "filename": record.original_filename,
            "document_type": record.document_type,
            "language": record.language,
            "processing_time": record.processing_time,
            "created_at": record.created_at.isoformat(),
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            # markdown only - `plain_text` was the same content with the
            # formatting stripped, doubling the payload. Still stored for
            # history search and the txt/csv exports.
            "markdown": result.markdown_text if result else "",
            # The extracted table. Without this a caller polling for an async
            # job got prose back and had to re-parse the table out of it, or
            # make a second call to /export - even though the structured result
            # was already sitting in the row being read here.
            "structured_data": result.structured_json if result else {}
        }
    }


@router.get("/erp/{doc_id}")
def get_erp_data(doc_id: int, db: Session = Depends(get_db)):
    """
    The business data for one document, as an ERP consumes it.

    Same payload as `response_format=erp` elsewhere, on its own URL so an
    integration can be pointed at a single endpoint with no parameters to get
    wrong. Fields carry the names the document yielded; `line_items` are
    objects keyed by column name; `review_required` and `review_fields` say
    what a person should look at before the document is posted.
    """
    record = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    if record.status and record.status.startswith("PROCESSING"):
        raise HTTPException(
            status_code=409,
            detail=f"Document {doc_id} is still processing. Poll /api/v1/ocr/status/{record.request_id or doc_id} first."
        )
    if record.status == "FAILED":
        raise HTTPException(
            status_code=409,
            detail=f"Document {doc_id} failed: {record.error_message or 'OCR processing failed.'}"
        )

    result = db.query(DocumentResult).filter(DocumentResult.document_id == doc_id).first()
    return {
        "success": True,
        "response_format": "erp",
        "data": erp_from_record(record, result)
    }


@router.get("/document/{doc_id}")
def get_document_details(doc_id: int, response_format: str = "full", db: Session = Depends(get_db)):
    """
    Fetch complete OCR processing data for a specific document ID.

    `response_format=erp` returns the business projection instead - see
    GET /api/v1/ocr/erp/{doc_id}.
    """
    erp_only = _want_erp(response_format)
    record = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    result = db.query(DocumentResult).filter(DocumentResult.document_id == doc_id).first()

    if erp_only:
        return {
            "success": True,
            "response_format": "erp",
            "data": erp_from_record(record, result)
        }

    pages = db.query(DocumentPage).filter(DocumentPage.document_id == doc_id).order_by(DocumentPage.page_number.asc()).all()

    return {
        "success": True,
        "document": {
            "id": record.id,
            "request_id": record.request_id,
            "department": record.department,
            "user_id": record.user_id,
            "filename": record.original_filename,
            "document_type": record.document_type,
            "language": record.language,
            "has_handwriting": record.has_handwriting,
            "page_count": record.page_count,
            "processing_time": record.processing_time,
            "created_at": record.created_at.isoformat(),
            # markdown only, here and per page - see the note on /status.
            "markdown": result.markdown_text if result else "",
            # Same reasoning as /status: the extracted table is already in the
            # row being read, so publish it rather than making the caller go
            # to /export for it.
            "structured_data": result.structured_json if result else {},
            "pages": [
                {
                    "page_number": p.page_number,
                    "image_path": p.page_image_path,
                    "markdown": p.markdown,
                    "processing_time": p.processing_time
                } for p in pages
            ]
        }
    }


@router.post("/reprocess/{doc_id}")
async def reprocess_document(
    doc_id: int,
    rotate_deg: int = 0,
    flip_horizontal: bool = False,
    db: Session = Depends(get_db)
):
    """Reprocess an existing document record through the OCR pipeline with optional rotation (90, 180, 270) or mirror flip."""
    record = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = Path(record.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Original source file no longer exists on disk.")

    # Apply rotation or flip if requested
    if rotate_deg in (90, 180, 270) or flip_horizontal:
        from app.services.preprocessor import preprocessor
        transformed_path = preprocessor.rotate_or_flip(file_path, rotate_deg=rotate_deg, flip_horizontal=flip_horizontal)
        file_path = transformed_path
        record.file_path = str(transformed_path)
        db.commit()

    # Delete existing sub-records
    db.query(DocumentPage).filter(DocumentPage.document_id == doc_id).delete()
    db.query(DocumentResult).filter(DocumentResult.document_id == doc_id).delete()
    db.commit()

    # Re-run pipeline while preserving existing Document ID
    requested_type = record.document_type if record.document_type else "AUTO"
    new_result = await run_in_threadpool(
        pipeline.process_file,
        db,
        file_path,
        record.original_filename,
        requested_doc_type=requested_type,
        existing_doc_id=doc_id
    )
    return {
        "success": True,
        "message": f"Document ID {doc_id} successfully reprocessed.",
        "data": _api_document(new_result)
    }


@router.get("/export/{doc_id}")
def export_document(doc_id: int, format: str = "txt", db: Session = Depends(get_db)):
    """Download document output in specified format (txt, md, json, csv)."""
    record = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    result = db.query(DocumentResult).filter(DocumentResult.document_id == doc_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Document processing results missing")

    try:
        content, media_type, filename = export_service.generate_export(record, result, format)
        return Response(
            content=content.encode("utf-8"),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
