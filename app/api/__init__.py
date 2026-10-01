from app.api.routes_views import router as views_router
from app.api.routes_ocr import router as ocr_router
from app.api.routes_history import router as history_router
from app.api.routes_settings import router as settings_router

__all__ = [
    "views_router",
    "ocr_router",
    "history_router",
    "settings_router"
]
