from pathlib import Path
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import init_db
from app.core.logger import logger
from app.core.exceptions import (
    DocumentProcessingError,
    global_exception_handler,
    document_processing_exception_handler
)
from app.api.routes_views import router as views_router
from app.api.routes_ocr import router as ocr_router
from app.api.routes_history import router as history_router
from app.api.routes_settings import router as settings_router
from app.api.routes_analytics import router as analytics_router
from app.services.model_service import model_engine
from app.core.auth import media_token_is_valid, request_has_ui_access
from app.core.maintenance import recover_stranded_jobs, start_retention_thread
from app.core.ratelimit import rate_limit_middleware

app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise-grade AI Document Understanding & Intelligence Platform powered by Qwen2.5-VL-3B-Instruct",
    version="1.0.0",
    # An open /docs is a tidy inventory of the attack surface. Internally
    # that is fine; on an exposed host it should be switched off.
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
)

# Static assets are public - CSS and JS carry nothing sensitive.
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# uploads/ and outputs/ are NOT mounted.
#
# They were, as plain StaticFiles, which made every document ever processed -
# Aadhaar cards, PAN cards, resumes - downloadable by anyone who could reach
# the host, with no credential and nothing logged. They are now served only
# through the guarded route below.
_MEDIA_ROOTS = {
    "uploads": Path(settings.UPLOAD_DIR).resolve(),
    "outputs": Path(settings.OUTPUT_DIR).resolve(),
}


@app.get("/media/{kind}/{filename:path}", tags=["Media"])
def serve_media(
    kind: str,
    filename: str,
    request: Request,
    t: str = Query(None, description="Signed, short-lived media token"),
):
    """
    Serve one stored page render or original upload.

    Two ways in: a signed token scoped to this exact filename (what the viewer
    embeds), or credentials that already grant access to the document itself -
    a UI session or an API key. Anything else gets a 404 rather than a 403, so
    the endpoint cannot be used to test whether a document exists.
    """
    root = _MEDIA_ROOTS.get(kind)
    if root is None:
        raise HTTPException(status_code=404, detail="Not found")

    # Resolve before comparing: "../" in the path must not escape the root,
    # and a symlink inside it must not either.
    target = (root / filename).resolve()
    if not str(target).startswith(str(root)) or not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")

    if not (media_token_is_valid(filename, t) or request_has_ui_access(request)):
        logger.warning(f"Unauthorized media request for {kind}/{filename}")
        raise HTTPException(status_code=404, detail="Not found")

    return FileResponse(
        target,
        headers={
            # Signed links are shareable by nature; keep them out of shared
            # caches and out of the browser's disk cache.
            "Cache-Control": "private, max-age=300, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )

# Rate limiting runs ahead of the routes so a flood is rejected before it
# reaches the pipeline.
app.middleware("http")(rate_limit_middleware)

if settings.cors_origin_list():
    # Only for browser callers, and only origins named explicitly - a wildcard
    # would let any web page a user visits call this API as them.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type", "X-Project-Name", "X-User-Id"],
    )


@app.get("/health", tags=["Ops"])
def health():
    """Liveness probe. Unauthenticated and cheap by design - it reveals nothing."""
    return {"status": "ok", "app": settings.APP_NAME, "version": "1.0.0"}


@app.get("/readiness", tags=["Ops"])
def readiness():
    """
    Readiness probe: can this instance actually do work right now?

    Distinguishes "process is up" from "database reachable and the vision
    model loaded", which is what a load balancer needs in order to route.
    """
    from sqlalchemy import text as _sql_text
    from app.core.database import SessionLocal

    checks = {"database": False, "model": False}
    db = SessionLocal()
    try:
        db.execute(_sql_text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        logger.error(f"Readiness: database check failed: {e}")
    finally:
        db.close()

    try:
        checks["model"] = bool(getattr(model_engine, "is_ready", lambda: True)())
    except Exception:
        checks["model"] = False

    ready = all(checks.values())
    return JSONResponse(status_code=200 if ready else 503,
                        content={"ready": ready, "checks": checks})


# Register Exception Handlers
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(DocumentProcessingError, document_processing_exception_handler)

# Include API & View Routers
app.include_router(views_router)
app.include_router(ocr_router)
app.include_router(history_router)
app.include_router(settings_router)
app.include_router(analytics_router)


import threading

@app.on_event("startup")
async def startup_event():
    logger.info(f"Initializing {settings.APP_NAME} (env={settings.APP_ENV})...")
    settings.setup_directories()

    # Refuse to serve a production deployment that is missing its guards.
    # Starting anyway is how an "internal" host ends up publicly readable.
    problems = settings.validate_for_production()
    if problems:
        if settings.is_production():
            for pr in problems:
                logger.critical(f"UNSAFE CONFIGURATION: {pr}")
            raise RuntimeError(
                "Refusing to start in production with an unsafe configuration: "
                + "; ".join(problems)
            )
        for pr in problems:
            logger.warning(f"Development mode, would block production start: {pr}")

    init_db()

    # A previous process may have died holding jobs. Nothing else reconciles
    # them, so they would poll as PROCESSING forever.
    recover_stranded_jobs()
    start_retention_thread()

    # Pre-warm vision model in background thread to eliminate first-upload cold-start latency
    threading.Thread(target=model_engine.load_model, daemon=True).start()
    logger.info("Startup complete. Platform ready for requests.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AI Document Intelligence Platform.")
