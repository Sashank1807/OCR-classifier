from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from app.core.config import settings
from app.core.logger import logger

# SQLite or MySQL connection engine configuration
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> None:
    """Initialize database tables and apply missing column migrations."""
    try:
        Base.metadata.create_all(bind=engine)
        
        with engine.connect() as conn:
            from sqlalchemy import inspect, text
            inspector = inspect(engine)
            if "document_records" in inspector.get_table_names():
                existing_cols = {c["name"] for c in inspector.get_columns("document_records")}
                new_columns = [
                    ("request_id", "VARCHAR(100)"),
                    ("project_name", "VARCHAR(100) DEFAULT 'DefaultProject'"),
                    ("department", "VARCHAR(100) DEFAULT 'General'"),
                    ("user_id", "VARCHAR(100) DEFAULT 'system'"),
                    ("overall_confidence", "FLOAT DEFAULT 1.0"),
                    ("needs_manual_review", "BOOLEAN DEFAULT 0"),
                    ("document_quality", "VARCHAR(50) DEFAULT 'Good'"),
                    ("prompt_tokens", "INTEGER DEFAULT 0"),
                    ("completion_tokens", "INTEGER DEFAULT 0"),
                    ("total_tokens", "INTEGER DEFAULT 0"),
                    ("error_message", "TEXT"),
                    ("completed_at", "DATETIME"),
                    ("processing_attempts", "INTEGER DEFAULT 0")
                ]
                for col_name, col_type in new_columns:
                    if col_name not in existing_cols:
                        try:
                            conn.execute(text(f"ALTER TABLE document_records ADD COLUMN {col_name} {col_type}"))
                            conn.commit()
                            logger.info(f"Added missing column '{col_name}' to document_records.")
                        except Exception as ex:
                            logger.warning(f"Could not add column '{col_name}': {str(ex)}")

        logger.info("Database tables & schema migrations initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise


def get_db() -> Generator[Session, None, None]:
    """Dependency injection to provide database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
