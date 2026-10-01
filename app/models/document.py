import json
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.mysql import LONGTEXT

# MySQL TEXT is capped at 65,535 bytes and truncates without error.
# A multi-page document's markdown and structured JSON both exceed it.
LongText = Text().with_variant(LONGTEXT, "mysql")
from app.core.database import Base


class DocumentRecord(Base):
    """SQLAlchemy model representing a processed document entry."""
    __tablename__ = "document_records"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(100), unique=True, index=True, nullable=True)
    project_name = Column(String(100), default="DefaultProject", index=True)
    department = Column(String(100), default="General", index=True)
    user_id = Column(String(100), default="system")
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    page_count = Column(Integer, default=1)
    document_type = Column(String(100), default="Unknown")
    language = Column(String(100), default="English")
    has_handwriting = Column(Boolean, default=False)
    processing_time = Column(Float, default=0.0)
    confidence = Column(Float, default=1.0)
    overall_confidence = Column(Float, default=1.0)
    needs_manual_review = Column(Boolean, default=False)
    document_quality = Column(String(50), default="Good")
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    status = Column(String(50), default="COMPLETED")
    # Incremented each time a job is picked up. Startup recovery re-queues
    # an interrupted document, and this is what stops a document that
    # crashes the pipeline from being retried forever.
    processing_attempts = Column(Integer, default=0)
    error_message = Column(LongText, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    pages = relationship("DocumentPage", back_populates="document", cascade="all, delete-orphan")
    result = relationship("DocumentResult", back_populates="document", uselist=False, cascade="all, delete-orphan")


class DocumentPage(Base):
    """SQLAlchemy model representing per-page extraction results for multi-page documents."""
    __tablename__ = "document_pages"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("document_records.id", ondelete="CASCADE"), nullable=False)
    page_number = Column(Integer, nullable=False)
    page_image_path = Column(String(500), nullable=False)
    plain_text = Column(LongText, default="")
    markdown = Column(LongText, default="")
    confidence = Column(Float, default=1.0)
    processing_time = Column(Float, default=0.0)

    document = relationship("DocumentRecord", back_populates="pages")


class DocumentResult(Base):
    """SQLAlchemy model storing combined OCR outputs (Raw Text, Markdown, Structured JSON, Tables)."""
    __tablename__ = "document_results"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("document_records.id", ondelete="CASCADE"), nullable=False, unique=True)
    raw_text = Column(LongText, default="")
    markdown_text = Column(LongText, default="")
    structured_json_str = Column(LongText, default="{}")
    tables_json_str = Column(LongText, default="[]")

    document = relationship("DocumentRecord", back_populates="result")

    @property
    def structured_json(self):
        try:
            return json.loads(self.structured_json_str) if self.structured_json_str else {}
        except Exception:
            return {}

    @property
    def tables_json(self):
        try:
            return json.loads(self.tables_json_str) if self.tables_json_str else []
        except Exception:
            return []


class ApiKey(Base):
    """
    An API key issued to a department, at runtime, from the admin UI.

    WHAT IS STORED IS A HASH, never the key itself. An API key is a bearer
    credential: anyone holding it can read every extracted document. Keeping
    the plaintext in a table means a database dump, a backup file or a stray
    SELECT hands over live access to the whole corpus, and there is no
    legitimate reason to ever read one back - verification only needs to
    know whether a presented key matches.

    So the key is shown to the administrator ONCE, at creation, and after
    that only `prefix` survives for display ("dik_7f3a…"), which is enough
    to tell two keys apart in a list without being enough to use one.

    The hash is HMAC-SHA256 under SECRET_KEY - deterministic, so a presented
    key can be looked up directly rather than by scanning every row. The
    trade is that rotating SECRET_KEY invalidates every issued key, exactly
    as it already invalidates sessions and media links.
    """

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    department = Column(String(120), nullable=False, index=True)
    # Unique: two identical hashes would mean two identical keys.
    key_hash = Column(String(64), nullable=False, unique=True, index=True)
    prefix = Column(String(24), nullable=False)
    note = Column(String(255), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(80), nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    # Revoked rather than deleted: who held access, and until when, is the
    # question an audit asks, and a deleted row cannot answer it.
    revoked_at = Column(DateTime, nullable=True)

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None
