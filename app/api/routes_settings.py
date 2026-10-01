from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from app.core.config import settings
from app.core.auth import require_api_key

router = APIRouter(prefix="/api/v1/settings", tags=["Platform Settings"], dependencies=[Depends(require_api_key)])


class SettingsUpdateSchema(BaseModel):
    max_upload_size_mb: int
    model_path: str
    device: str
    enable_preprocessing: bool
    deskew_enabled: bool
    clahe_enabled: bool
    noise_removal_enabled: bool
    adaptive_threshold_enabled: bool


@router.get("/")
def get_current_settings():
    """Fetch current system configuration settings."""
    return {
        "success": True,
        "settings": {
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB,
            "model_path": settings.MODEL_PATH,
            "device": settings.DEVICE,
            "local_files_only": settings.LOCAL_FILES_ONLY,
            "enable_preprocessing": settings.ENABLE_PREPROCESSING,
            "deskew_enabled": settings.DESKEW_ENABLED,
            "clahe_enabled": settings.CLAHE_ENABLED,
            "noise_removal_enabled": settings.NOISE_REMOVAL_ENABLED,
            "adaptive_threshold_enabled": settings.ADAPTIVE_THRESHOLD_ENABLED,
            "allowed_extensions": settings.ALLOWED_EXTENSIONS
        }
    }


@router.post("/update")
def update_settings(payload: SettingsUpdateSchema):
    """Update runtime configuration settings."""
    settings.MAX_UPLOAD_SIZE_MB = payload.max_upload_size_mb
    settings.MODEL_PATH = payload.model_path
    settings.DEVICE = payload.device
    settings.ENABLE_PREPROCESSING = payload.enable_preprocessing
    settings.DESKEW_ENABLED = payload.deskew_enabled
    settings.CLAHE_ENABLED = payload.clahe_enabled
    settings.NOISE_REMOVAL_ENABLED = payload.noise_removal_enabled
    settings.ADAPTIVE_THRESHOLD_ENABLED = payload.adaptive_threshold_enabled

    return {
        "success": True,
        "message": "Settings updated successfully."
    }
