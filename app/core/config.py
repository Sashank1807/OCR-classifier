import os
from pathlib import Path
from typing import List, Union
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "AI Document Intelligence Platform"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8080

    DATABASE_URL: str = "sqlite:///./ocr_database.db"

    # Model Settings
    MODEL_BACKEND: str = "ollama"  # 'ollama' or 'transformers'
    MODEL_PATH: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    DEVICE: str = "auto"  # 'cuda', 'cpu', or 'auto'
    LOCAL_FILES_ONLY: bool = True
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5vl:7b"
    OLLAMA_TIMEOUT: int = 180

    # Storage Settings
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "bmp", "tiff", "webp", "jfif", "heic", "pdf", "xlsx", "xls", "csv", "txt", "docx", "doc"]
    UPLOAD_DIR: Path = Path("./uploads")
    OUTPUT_DIR: Path = Path("./outputs")
    LOG_DIR: Path = Path("./logs")

    # Image Preprocessing Settings
    ENABLE_PREPROCESSING: bool = True
    DESKEW_ENABLED: bool = True
    CLAHE_ENABLED: bool = True
    NOISE_REMOVAL_ENABLED: bool = True
    ADAPTIVE_THRESHOLD_ENABLED: bool = False

    # RapidOCR Detector Sensitivity Settings
    # Library defaults (det_box_thresh=0.5, det_unclip_ratio=1.6, text_score=0.5)
    # silently drop faint/small text regions before the app's own confidence
    # metrics ever see them. Lowered box_thresh/text_score and raised
    # unclip_ratio recover more of that faint/small text, deferring the
    # accept/reject decision to the app's own cell-level cascade
    # (is_cell_visually_empty / rank_candidates) instead of the OCR
    # detector's default cutoff.
    RAPIDOCR_DET_BOX_THRESH: float = 0.35
    RAPIDOCR_DET_UNCLIP_RATIO: float = 2.0
    RAPIDOCR_TEXT_SCORE: float = 0.35

    # ------------------------------------------------------------------ #
    # Security
    #
    # These default to the SAFE value, not the convenient one: a missing
    # secret or key list means the app refuses to serve rather than serving
    # openly. is_production() gates that check, so development still runs
    # with no configuration at all.
    # ------------------------------------------------------------------ #

    # HMAC secret for media tokens and UI session cookies. MUST be set (and
    # kept stable) in production - rotating it logs everyone out and
    # invalidates outstanding media links.
    SECRET_KEY: str = ""

    # Comma-separated API keys accepted on /api/v1/*. Callers send one as the
    # X-API-Key header.
    API_KEYS: str = ""

    # Password for the browser UI. The UI shows extracted PII, so it is not
    # left open even though it is "just" the viewer.
    UI_PASSWORD: str = ""

    # The administrator account. Analytics is admin-only: it aggregates
    # across every department's documents and exposes volumes, token spend
    # and per-project activity, which is a different thing to see than one
    # document you were sent.
    #
    # Kept in configuration, never in source. A password committed to a repo
    # is a password published to everyone who can read the repo, and it
    # cannot be rotated without a code change.
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = ""

    # How long a signed media link stays valid. Short, because the link is
    # the only thing standing between an Aadhaar scan and anyone holding it.
    MEDIA_TOKEN_TTL_SECONDS: int = 900
    SESSION_TTL_SECONDS: int = 43200  # 12h

    # Per-client request ceiling. One 50MB upload buys 30-90s of CPU and VLM
    # time, so the write endpoints are cheap to abuse.
    RATE_LIMIT_PER_MINUTE: int = 60
    UPLOAD_RATE_LIMIT_PER_MINUTE: int = 10

    # Interactive API docs. Harmless internally, an inventory of the attack
    # surface on an exposed host.
    ENABLE_DOCS: bool = True

    # Browser-based callers (Hoppscotch, a future web front-end). Empty means
    # no cross-origin access, which is what a server-to-server API wants.
    CORS_ORIGINS: str = ""

    # ------------------------------------------------------------------ #
    # Retention
    #
    # outputs/ reached 1.8GB and uploads/ 172MB with no policy at all. These
    # hold page renders and original ID documents, so unbounded retention is
    # both a disk problem and a privacy one.
    # ------------------------------------------------------------------ #
    RETENTION_DAYS: int = 30
    RETENTION_SWEEP_HOURS: int = 6
    # Defaults OFF. Turning this on deletes uploads, page renders and DB rows
    # older than RETENTION_DAYS on the next sweep - permanently, with no
    # undo. Enable it deliberately, after checking RETENTION_DAYS against how
    # long you actually need documents kept.
    ENABLE_RETENTION_SWEEP: bool = False

    # Debug renders (per-page bbox/grid overlays) are a development aid and
    # multiply output size per document.
    SAVE_DEBUG_IMAGES: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def is_production(self) -> bool:
        return self.APP_ENV.strip().lower() in {"production", "prod", "staging"}

    def api_key_set(self) -> set:
        return {k.strip() for k in self.API_KEYS.split(",") if k.strip()}

    def cors_origin_list(self) -> list:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    def validate_for_production(self) -> list:
        """
        Configuration that would be unsafe to serve with. Returned rather than
        raised so the caller can decide: startup refuses in production and
        logs a warning in development.
        """
        problems = []
        if not self.SECRET_KEY or len(self.SECRET_KEY) < 32:
            problems.append("SECRET_KEY is unset or shorter than 32 chars")
        if not self.api_key_set():
            problems.append("API_KEYS is empty - every /api/v1 route would be open")
        if not self.UI_PASSWORD:
            problems.append("UI_PASSWORD is empty - the document viewer would be open")
        if not self.ADMIN_PASSWORD:
            # Not fatal: with no admin password there is simply no admin role,
            # and analytics falls back to any signed-in user - which is what
            # it was before roles existed. Worth saying out loud, though.
            problems.append(
                "ADMIN_PASSWORD is empty - analytics will be visible to every "
                "signed-in user rather than administrators only"
            )
        if self.DEBUG:
            problems.append("DEBUG is true - enables autoreload, which kills in-flight OCR jobs")
        return problems

    def setup_directories(self) -> None:
        """Ensure all required runtime directories exist."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.setup_directories()
