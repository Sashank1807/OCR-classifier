from pathlib import Path
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Request, Depends, HTTPException, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.config import settings
from app.models.document import DocumentRecord, DocumentPage, DocumentResult
from app.core.auth import (
    ROLE_ADMIN,
    SESSION_COOKIE,
    authenticate_ui,
    check_ui_password,
    request_is_admin,
    session_role,
    issue_media_token,
    issue_session,
    request_has_ui_access,
    ui_auth_required,
)
from app.core.logger import logger
from app.services.erp_payload import erp_from_record, line_item_columns

router = APIRouter(tags=["Frontend Views"])
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


import json


def get_basename(path_str: str) -> str:
    """Jinja filter returning the basename of a path string."""
    return Path(path_str).name if path_str else ""


def get_media_url(path_str: str) -> str:
    """
    Jinja filter returning a SIGNED, short-lived URL for a stored image.

    /uploads and /outputs are no longer public directories, so a bare path is
    not fetchable any more. Each link instead carries a token scoped to that
    one filename and valid for MEDIA_TOKEN_TTL_SECONDS, which means a link
    copied out of the page stops working shortly afterwards rather than
    exposing an ID document indefinitely.
    """
    if not path_str:
        return ""
    filename = Path(path_str).name
    kind = "outputs"
    if not (settings.OUTPUT_DIR / filename).exists() and (settings.UPLOAD_DIR / filename).exists():
        kind = "uploads"
    return f"/media/{kind}/{filename}?t={issue_media_token(filename)}"


def tojson_pretty(val) -> str:
    """Jinja filter returning pretty formatted JSON string."""
    if not val:
        return "{}"
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            return val
    return json.dumps(val, indent=2, ensure_ascii=False)


templates.env.filters["basename"] = get_basename
templates.env.filters["media_url"] = get_media_url
templates.env.filters["tojson_pretty"] = tojson_pretty
# Each extra table carries its own columns, so the template resolves them
# per table rather than reusing the primary table's header.
templates.env.filters["line_item_columns"] = line_item_columns

# Available in every template, so the nav can hide what the viewer cannot
# open rather than offering a link that 404s.
templates.env.globals["is_admin"] = request_is_admin


def static_url(path: str) -> str:
    """
    A /static URL carrying the file's modification time as ?v=.

    StaticFiles does send ETag and Last-Modified, but browsers are free to
    serve a cached stylesheet from memory WITHOUT revalidating, and they
    routinely do. The practical effect is that a CSS change appears not to
    have happened - which is indistinguishable, from the outside, from the
    change not having been made. The mtime makes the URL itself change, so
    there is nothing to decide: a new file is a new resource.
    """
    rel = path.lstrip("/")
    candidate = Path(__file__).parent.parent / rel
    try:
        stamp = int(candidate.stat().st_mtime)
    except OSError:
        return f"/{rel}"
    return f"/{rel}?v={stamp}"


templates.env.globals["static_url"] = static_url



# --------------------------------------------------------------------------- #
# Browser session
#
# The viewer renders extracted PII - names, Aadhaar numbers, addresses - and
# links to the source scans, so it is gated like the API is. A single shared
# password is deliberately modest: it is what this deployment needs, and it is
# vastly better than the nothing that was here before. Swap in SSO when there
# is an identity provider to talk to.
# --------------------------------------------------------------------------- #

def require_ui(request: Request) -> None:
    """Dependency for every page that shows document content."""
    if not request_has_ui_access(request):
        raise HTTPException(
            status_code=307,
            detail="Login required",
            headers={"Location": f"/login?next={request.url.path}"},
        )


@router.get("/login")
def login_view(request: Request, next: str = "/", error: str = ""):
    if not ui_auth_required() or request_has_ui_access(request):
        return RedirectResponse(url=next or "/", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"next": next, "error": error, "app_name": settings.APP_NAME},
    )


@router.post("/login")
def login_submit(request: Request, username: str = Form(""), password: str = Form(""),
                 next: str = Form("/")):
    client = request.client.host if request.client else "unknown"
    role = authenticate_ui(username, password)
    if role is None:
        logger.warning(f"Failed UI login attempt from {client} (username={username!r})")
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            # One message for both a wrong username and a wrong password:
            # saying which was wrong confirms that an account exists.
            context={"next": next, "error": "Those credentials were not recognised.",
                     "app_name": settings.APP_NAME, "username": username},
            status_code=401,
        )
    logger.info(f"UI login from {client} as {role}")
    # Redirect only to a path on this host - an absolute URL in `next` would
    # turn the login form into an open redirect.
    dest = next if next.startswith("/") and not next.startswith("//") else "/"
    resp = RedirectResponse(url=dest, status_code=303)
    resp.set_cookie(
        SESSION_COOKIE,
        issue_session(role),
        max_age=settings.SESSION_TTL_SECONDS,
        httponly=True,                       # not readable from JS
        samesite="lax",                      # not sent on cross-site POSTs
        secure=settings.is_production(),     # HTTPS-only once deployed
    )
    return resp


@router.get("/logout")
def logout(request: Request):
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@router.get("/", dependencies=[Depends(require_ui)])
def dashboard_view(request: Request, db: Session = Depends(get_db)):
    """Render main Dashboard with recent activity and document stats."""
    total_docs = db.query(DocumentRecord).count()
    total_pages = db.query(DocumentPage).count()

    recent_docs = db.query(DocumentRecord).order_by(DocumentRecord.created_at.desc()).limit(5).all()

    # Document type distribution
    doc_types = ["Visiting Card", "Prescription", "Invoice", "Pamphlet", "Certificate", "Report"]
    type_counts = {}
    for dt in doc_types:
        type_counts[dt] = db.query(DocumentRecord).filter(DocumentRecord.document_type == dt).count()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "app_name": settings.APP_NAME,
            "total_docs": total_docs,
            "total_pages": total_pages,
            "recent_docs": recent_docs,
            "type_counts": type_counts
        }
    )


@router.get("/upload", dependencies=[Depends(require_ui)])
def upload_view(request: Request):
    """Render Document Upload page."""
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={
            "app_name": settings.APP_NAME,
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB,
            "allowed_extensions": ", ".join(settings.ALLOWED_EXTENSIONS)
        }
    )


@router.get("/result/{doc_id}", dependencies=[Depends(require_ui)])
def result_view(doc_id: int, request: Request, db: Session = Depends(get_db)):
    """Render detailed Document OCR Result Reader page."""
    record = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Document record not found")

    result = db.query(DocumentResult).filter(DocumentResult.document_id == doc_id).first()
    pages = db.query(DocumentPage).filter(DocumentPage.document_id == doc_id).order_by(DocumentPage.page_number.asc()).all()

    # The same projection the API publishes for response_format=erp, rendered
    # here so what a reviewer signs off on is what the ERP will receive - not
    # a separate view of it that could drift.
    try:
        erp_data = erp_from_record(record, result)
        erp_columns = line_item_columns(erp_data)
    except Exception as exc:
        logger.error(f"Could not build the ERP view of document {doc_id}: {exc}", exc_info=True)
        erp_data, erp_columns = None, []

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "app_name": settings.APP_NAME,
            "record": record,
            "result": result,
            "pages": pages,
            "erp_data": erp_data,
            "erp_columns": erp_columns
        }
    )


@router.get("/history", dependencies=[Depends(require_ui)])
def history_view(request: Request, db: Session = Depends(get_db)):
    """Render OCR Document History page."""
    records = db.query(DocumentRecord).order_by(DocumentRecord.created_at.desc()).all()
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "app_name": settings.APP_NAME,
            "records": records
        }
    )


@router.get("/analytics", dependencies=[Depends(require_ui)])
def analytics_view(request: Request):
    """Render Enterprise Multi-Tenant Analytics Dashboard page (admin only)."""
    if not request_is_admin(request):
        # A signed-in viewer is told the page is not there, rather than that
        # it exists and they may not have it.
        raise HTTPException(status_code=404, detail="Not found")
    return templates.TemplateResponse(
        request=request,
        name="analytics.html",
        context={
            "app_name": settings.APP_NAME
        }
    )
