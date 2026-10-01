from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db
from app.models.document import DocumentRecord
from app.core.auth import SESSION_COOKIE, require_admin

# Analytics aggregates ACROSS every department - document volumes, token
# spend, per-project activity - so it is gated more tightly than the rest of
# the API. `require_admin` accepts an administrator session OR a valid API
# key: the integrations that poll this are services with a key and no
# session, while the thing being kept out is an ordinary viewer who signed
# in to read one document.
router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["Enterprise Analytics & Department Monitoring"],
    dependencies=[Depends(require_admin)],
)


@router.get("/metrics")
def get_analytics_metrics(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Returns dashboard-ready metrics for project monitoring:
    - Project-wise usage breakdown
    - Total, successful, and failed request counts
    - Token consumption totals & averages
    - Average processing time and confidence scores
    - Document type distribution
    """
    total_requests = db.query(func.count(DocumentRecord.id)).scalar() or 0
    successful_requests = db.query(func.count(DocumentRecord.id)).filter(DocumentRecord.status == "COMPLETED").scalar() or 0
    failed_requests = db.query(func.count(DocumentRecord.id)).filter(DocumentRecord.status == "FAILED").scalar() or 0
    processing_requests = db.query(func.count(DocumentRecord.id)).filter(DocumentRecord.status == "PROCESSING").scalar() or 0

    avg_confidence = db.query(func.avg(DocumentRecord.overall_confidence)).filter(DocumentRecord.status == "COMPLETED").scalar() or 1.0
    avg_processing_time = db.query(func.avg(DocumentRecord.processing_time)).filter(DocumentRecord.status == "COMPLETED").scalar() or 0.0

    total_prompt_tokens = db.query(func.sum(DocumentRecord.prompt_tokens)).scalar() or 0
    total_completion_tokens = db.query(func.sum(DocumentRecord.completion_tokens)).scalar() or 0
    total_tokens = db.query(func.sum(DocumentRecord.total_tokens)).scalar() or 0

    avg_tokens = round(total_tokens / total_requests, 1) if total_requests > 0 else 0

    # Project-wise breakdown
    proj_rows = db.query(
        func.coalesce(DocumentRecord.project_name, DocumentRecord.department).label("proj"),
        func.count(DocumentRecord.id).label("requests"),
        func.sum(DocumentRecord.total_tokens).label("tokens"),
        func.avg(DocumentRecord.overall_confidence).label("avg_conf"),
        func.avg(DocumentRecord.processing_time).label("avg_time")
    ).group_by(func.coalesce(DocumentRecord.project_name, DocumentRecord.department)).all()

    project_metrics = []
    for proj, reqs, toks, conf, ptime in proj_rows:
        project_metrics.append({
            "project_name": proj or "DefaultProject",
            "department": proj or "DefaultProject",
            "total_requests": reqs,
            "total_tokens": toks or 0,
            "average_confidence": round(float(conf or 1.0), 4),
            "average_processing_time": round(float(ptime or 0.0), 2)
        })

    # Document type distribution
    type_rows = db.query(
        DocumentRecord.document_type,
        func.count(DocumentRecord.id).label("count")
    ).group_by(DocumentRecord.document_type).all()

    doc_type_distribution = {dtype or "UNKNOWN": count for dtype, count in type_rows}

    return {
        "success": True,
        "summary": {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "processing_requests": processing_requests,
            "average_confidence": round(float(avg_confidence), 4),
            "average_processing_time": round(float(avg_processing_time), 2),
            "average_tokens_per_request": avg_tokens
        },
        "token_monitoring": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens_consumed": total_tokens
        },
        "project_analytics": project_metrics,
        "department_analytics": project_metrics,
        "document_type_distribution": doc_type_distribution
    }


@router.get("/metrics/{project_name}")
def get_project_analytics_metrics(project_name: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Returns project-restricted metrics for a single specific project (e.g. EmployeePortal, VendorBillsApp, HealthApp).
    Allows project teams to search and view only their project metrics.
    """
    proj_clean = project_name.strip()

    proj_query = db.query(DocumentRecord).filter(
        (func.lower(DocumentRecord.project_name) == proj_clean.lower()) |
        (func.lower(DocumentRecord.department) == proj_clean.lower())
    )

    total_requests = proj_query.count()
    if total_requests == 0:
        return {
            "success": True,
            "project_name": proj_clean,
            "message": f"No requests recorded for project '{proj_clean}' yet.",
            "summary": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "processing_requests": 0,
                "average_confidence": 1.0,
                "average_processing_time": 0.0,
                "average_tokens_per_request": 0
            },
            "token_monitoring": {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens_consumed": 0
            },
            "document_type_distribution": {}
        }

    successful_requests = proj_query.filter(DocumentRecord.status == "COMPLETED").count()
    failed_requests = proj_query.filter(DocumentRecord.status == "FAILED").count()
    processing_requests = proj_query.filter(DocumentRecord.status == "PROCESSING").count()

    avg_confidence = db.query(func.avg(DocumentRecord.overall_confidence)).filter(
        (func.lower(DocumentRecord.project_name) == proj_clean.lower()) |
        (func.lower(DocumentRecord.department) == proj_clean.lower()),
        DocumentRecord.status == "COMPLETED"
    ).scalar() or 1.0

    avg_processing_time = db.query(func.avg(DocumentRecord.processing_time)).filter(
        (func.lower(DocumentRecord.project_name) == proj_clean.lower()) |
        (func.lower(DocumentRecord.department) == proj_clean.lower()),
        DocumentRecord.status == "COMPLETED"
    ).scalar() or 0.0

    total_prompt_tokens = db.query(func.sum(DocumentRecord.prompt_tokens)).filter(
        (func.lower(DocumentRecord.project_name) == proj_clean.lower()) |
        (func.lower(DocumentRecord.department) == proj_clean.lower())
    ).scalar() or 0

    total_completion_tokens = db.query(func.sum(DocumentRecord.completion_tokens)).filter(
        (func.lower(DocumentRecord.project_name) == proj_clean.lower()) |
        (func.lower(DocumentRecord.department) == proj_clean.lower())
    ).scalar() or 0

    total_tokens = db.query(func.sum(DocumentRecord.total_tokens)).filter(
        (func.lower(DocumentRecord.project_name) == proj_clean.lower()) |
        (func.lower(DocumentRecord.department) == proj_clean.lower())
    ).scalar() or 0

    avg_tokens = round(total_tokens / total_requests, 1) if total_requests > 0 else 0

    # Document type distribution for this specific project
    type_rows = db.query(
        DocumentRecord.document_type,
        func.count(DocumentRecord.id).label("count")
    ).filter(
        (func.lower(DocumentRecord.project_name) == proj_clean.lower()) |
        (func.lower(DocumentRecord.department) == proj_clean.lower())
    ).group_by(DocumentRecord.document_type).all()

    doc_type_distribution = {dtype or "UNKNOWN": count for dtype, count in type_rows}

    # Fetch recent processed documents for this project
    recent_recs = proj_query.order_by(DocumentRecord.id.desc()).limit(10).all()
    recent_docs = []
    for r in recent_recs:
        recent_docs.append({
            "id": r.id,
            "filename": r.original_filename,
            "document_type": r.document_type,
            "status": r.status,
            "processing_time": r.processing_time,
            "created_at": r.created_at.isoformat()
        })

    return {
        "success": True,
        "project_name": proj_clean,
        "summary": {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "processing_requests": processing_requests,
            "average_confidence": round(float(avg_confidence), 4),
            "average_processing_time": round(float(avg_processing_time), 2),
            "average_tokens_per_request": avg_tokens
        },
        "token_monitoring": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens_consumed": total_tokens
        },
        "document_type_distribution": doc_type_distribution,
        "recent_documents": recent_docs
    }


# --------------------------------------------------------------------------- #
# Department API keys
#
# Issued at runtime so adding a department is not a config edit and a restart.
# The whole router is already behind `require_admin`, but these three are the
# ones where that matters most: issuing a key hands over read access to every
# extracted document, so it is an administrator action by definition.
# --------------------------------------------------------------------------- #

@router.get("/api-keys")
def get_api_keys(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Every issued key, active and revoked. Prefixes only - see below."""
    from app.services.api_key_service import list_keys
    return {"success": True, "api_keys": list_keys(db)}


@router.post("/api-keys")
def create_api_key(payload: Dict[str, Any], request: Request,
                   db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Issue a key for a department.

    The plaintext is in this response and nowhere else, ever. Only a hash is
    stored, so it cannot be looked up, re-sent or recovered - if it is lost,
    revoke it and issue another. The UI says so at the point of display.
    """
    from app.services.api_key_service import create_key, normalise_department

    department = normalise_department(str(payload.get("department", "")))
    if not department:
        raise HTTPException(status_code=400, detail="A department name is required.")
    if len(department) < 2:
        raise HTTPException(status_code=400, detail="That department name is too short.")

    actor = "admin" if request.cookies.get(SESSION_COOKIE) else "api"
    try:
        issued = create_key(db, department, created_by=actor,
                            note=str(payload.get("note", "")) or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    record = issued["record"]
    return {
        "success": True,
        "message": f"API key issued for {department}. Copy it now - it cannot be shown again.",
        "api_key": issued["key"],
        "key": {
            "id": record.id,
            "department": record.department,
            "prefix": record.prefix,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "active": True,
        },
    }


@router.delete("/api-keys/{key_id}")
def revoke_api_key(key_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Revoke a key. It stops working on the next request.

    The row is kept, not deleted: "who had access, and until when" is the
    question an audit asks, and a deleted row cannot answer it.
    """
    from app.services.api_key_service import revoke_key

    if not revoke_key(db, key_id):
        raise HTTPException(status_code=404, detail="No active key with that id.")
    return {"success": True, "message": "Key revoked. It will be refused on the next request."}
