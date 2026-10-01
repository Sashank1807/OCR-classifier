#!/usr/bin/env python3
"""
Launcher script for AI Document Intelligence Platform.
Starts the FastAPI Uvicorn server.
"""

import sys
# pyrefly: ignore [missing-import]
import uvicorn
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.core.logger import logger
from app.core.database import init_db


import logging

class EndpointFilter(logging.Filter):
    """Filter out repetitive status check access logs from Uvicorn output."""
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/v1/ocr/status/" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


def main():
    logger.info("Initializing runtime environment and directories...")
    settings.setup_directories()

    logger.info("Verifying database schema...")
    init_db()

    logger.info(f"Starting {settings.APP_NAME} on http://{settings.HOST}:{settings.PORT}")
    # Autoreload watches the project directory and restarts the process on any
    # file change - which silently kills every in-flight OCR job (30-90s each)
    # and leaves those documents stuck in PROCESSING. Development only, and
    # never when APP_ENV says production.
    reload_enabled = settings.DEBUG and not settings.is_production()
    if reload_enabled:
        logger.warning("Autoreload is ON - a file change will abort in-flight OCR jobs.")

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=reload_enabled,
        proxy_headers=True,          # trust X-Forwarded-* from the reverse proxy
        forwarded_allow_ips="*",
        timeout_keep_alive=65,
    )


if __name__ == "__main__":
    main()
