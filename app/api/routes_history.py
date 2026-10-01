from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.core.database import get_db
from app.models.document import DocumentRecord, DocumentResult, DocumentPage
from app.core.auth import require_api_key

router = APIRouter(prefix="/api/v1/history", tags=["Document History & Search"], dependencies=[Depends(require_api_key)])


@router.get("/search")
def search_history(
    q: Optional[str] = Query(None, description="Full-text query across text, filename, or document type"),
    doc_type: Optional[str] = Query(None, description="Filter by document type"),
    language: Optional[str] = Query(None, description="Filter by language"),
    db: Session = Depends(get_db)
):
    """Full-text search and filtering across processed OCR document records."""
    query = db.query(DocumentRecord).join(DocumentResult, DocumentRecord.id == DocumentResult.document_id, isouter=True)

    if doc_type and doc_type != "All":
        query = query.filter(DocumentRecord.document_type == doc_type)

    if language and language != "All":
        query = query.filter(DocumentRecord.language == language)

    if q:
        search_term = f"%{q.strip()}%"
        query = query.filter(
            or_(
                DocumentRecord.original_filename.ilike(search_term),
                DocumentRecord.document_type.ilike(search_term),
                DocumentRecord.language.ilike(search_term),
                DocumentResult.raw_text.ilike(search_term),
                DocumentResult.markdown_text.ilike(search_term),
                DocumentResult.structured_json_str.ilike(search_term)
            )
        )

    records = query.order_by(DocumentRecord.created_at.desc()).all()

    results = []
    for r in records:
        results.append({
            "id": r.id,
            "filename": r.original_filename,
            "document_type": r.document_type,
            "language": r.language,
            "has_handwriting": r.has_handwriting,
            "page_count": r.page_count,
            "processing_time": r.processing_time,
            "confidence": r.confidence,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        })

    return {
        "success": True,
        "total": len(results),
        "documents": results
    }


@router.delete("/delete/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    """Delete a document record and its associated pages and results from database."""
    record = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Document record not found")

    db.delete(record)
    db.commit()

    return {
        "success": True,
        "message": f"Document ID {doc_id} successfully deleted."
    }
