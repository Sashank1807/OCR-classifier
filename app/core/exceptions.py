import uuid
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.core.logger import logger


class DocumentProcessingError(Exception):
    """Custom exception raised during document processing failures."""
    def __init__(self, message: str, detail: str = None):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class InvalidFileTypeError(HTTPException):
    def __init__(self, filename: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{filename}' has an unsupported extension or corrupted content."
        )


class FileTooLargeError(HTTPException):
    def __init__(self, max_mb: int):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {max_mb} MB."
        )


async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler catching uncaught server errors."""
    # Correlate the client's response with this log line without handing the
    # client the exception text, which carries absolute file paths, SQL
    # fragments and library internals.
    incident = uuid.uuid4().hex[:12]
    logger.error(
        f"Unhandled Exception [{incident}] on {request.method} {request.url.path}: {str(exc)}",
        exc_info=True,
    )
    detail = str(exc) if settings.DEBUG else (
        f"An internal error occurred. Quote reference {incident} when reporting it."
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal System Exception",
            "incident_id": incident,
            "detail": detail,
        },
    )


async def document_processing_exception_handler(request: Request, exc: DocumentProcessingError):
    """Exception handler for document pipeline errors."""
    logger.error(f"Document Processing Error: {exc.message} - Detail: {exc.detail}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": exc.message,
            "detail": exc.detail or "Failed to extract or process document contents."
        }
    )
