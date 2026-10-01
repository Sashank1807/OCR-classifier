import os
import re
from pathlib import Path
from fastapi import UploadFile
from app.core.config import settings
from app.core.exceptions import InvalidFileTypeError, FileTooLargeError
from app.core.logger import logger


def sanitize_filename(filename: str) -> str:
    """Sanitizes uploaded filename to prevent directory traversal attacks."""
    filename = Path(filename).name
    # Keep alphanumeric characters, underscores, hyphens, and single dots
    cleaned = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
    return cleaned if cleaned else "uploaded_document"


def validate_file_upload(file: UploadFile, content: bytes) -> str:
    """
    Validates uploaded file size, extension, and magic bytes.
    Returns sanitized filename.
    """
    cleaned_filename = sanitize_filename(file.filename)
    extension = cleaned_filename.split(".")[-1].lower() if "." in cleaned_filename else ""

    if extension not in settings.ALLOWED_EXTENSIONS:
        logger.warning(f"Rejected file with invalid extension: {cleaned_filename}")
        raise InvalidFileTypeError(cleaned_filename)

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        logger.warning(f"File size {len(content)} bytes exceeds limit {max_bytes}")
        raise FileTooLargeError(settings.MAX_UPLOAD_SIZE_MB)

    # Basic Magic Bytes Validation
    magic_signatures = {
        "pdf": [b"%PDF"],
        "png": [b"\x89PNG"],
        "jpeg": [b"\xff\xd8\xff"],
        "jpg": [b"\xff\xd8\xff"],
        "bmp": [b"BM"],
        "tiff": [b"II*\x00", b"MM\x00*"],
        "webp": [b"RIFF"],
        "xlsx": [b"PK\x03\x04"],
        "xls": [b"\xd0\xcf\x11\xe0"],
        "docx": [b"PK\x03\x04"],
        "doc": [b"\xd0\xcf\x11\xe0"]
    }

    if extension in magic_signatures:
        signatures = magic_signatures[extension]
        matches = any(content.startswith(sig) for sig in signatures)
        if not matches and extension == "webp":
            # WEBP check offset 8
            matches = b"WEBP" in content[:16]
        
        if not matches:
            logger.warning(f"File header mismatch for {cleaned_filename} with extension {extension}")
            raise InvalidFileTypeError(cleaned_filename)

    return cleaned_filename
