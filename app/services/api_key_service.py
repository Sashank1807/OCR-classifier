"""
Issuing, verifying and revoking department API keys.

Keys used to exist only as a comma-separated list in `.env`, which meant
adding one for a new department was a config edit and a restart. These are
issued at runtime from the admin screen and work on the next request.

Three decisions worth stating, because each is the kind of thing that is
awkward to change later:

**Only a hash is stored.** An API key is a bearer credential - whoever holds
it can read every extracted document. Keeping the plaintext means a database
dump, a backup file or a careless SELECT hands over live access to the whole
corpus. Verification never needs to read a key back, only to compare, so the
plaintext is shown once at creation and then exists nowhere on this server.

**The hash is HMAC-SHA256 under SECRET_KEY**, not bcrypt. Deterministic
hashing is what lets a presented key be looked up by index instead of
scanning and re-hashing every row on every request. The trade is that
rotating SECRET_KEY invalidates all issued keys - which it already does to
sessions and media links, so the behaviour is at least consistent.

**Revoked, never deleted.** "Who had access, and until when" is the question
an audit asks, and a deleted row cannot answer it.

Keys from `.env` keep working exactly as before. They are the bootstrap
credential - something has to be able to call the API before anyone has
logged in to issue a key.
"""

import hashlib
import hmac
import re
import secrets
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logger import logger

# A recognisable, greppable prefix. When a key turns up in a log file or a
# pasted config, it should be obvious what it is and where to revoke it.
KEY_PREFIX = "dik"
_TOKEN_BYTES = 32

# Verification happens on every single API request, so the active hashes are
# held in process rather than fetched each time. The cache is invalidated
# whenever a key is issued or revoked, and expires on its own as well, so a
# second worker process picks up a change within the TTL.
_CACHE_TTL_SECONDS = 30
_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {"hashes": frozenset(), "loaded_at": 0.0}


def _secret() -> bytes:
    if settings.SECRET_KEY:
        return settings.SECRET_KEY.encode("utf-8")
    if settings.is_production():
        raise RuntimeError("SECRET_KEY must be set in production")
    # Mirrors app.core.auth: a per-process development secret.
    from app.core.auth import _secret as dev_secret
    return dev_secret()


def hash_key(raw_key: str) -> str:
    return hmac.new(_secret(), (raw_key or "").encode("utf-8"), hashlib.sha256).hexdigest()


def normalise_department(name: str) -> str:
    """
    Trim and collapse whitespace. Deliberately NOT slugified or lowercased:
    this is a label a human chose and will recognise in a list, and the
    analytics project names it sits beside are free text too.
    """
    cleaned = re.sub(r"\s+", " ", (name or "").strip())
    return cleaned[:120]


def invalidate_cache() -> None:
    with _cache_lock:
        _cache["loaded_at"] = 0.0


def _active_hashes() -> frozenset:
    now = time.time()
    with _cache_lock:
        if now - _cache["loaded_at"] < _CACHE_TTL_SECONDS:
            return _cache["hashes"]

    hashes = frozenset()
    try:
        from app.core.database import SessionLocal
        from app.models.document import ApiKey

        db = SessionLocal()
        try:
            rows = db.query(ApiKey.key_hash).filter(ApiKey.revoked_at.is_(None)).all()
            hashes = frozenset(r[0] for r in rows)
        finally:
            db.close()
    except Exception as exc:
        # A database hiccup must not take authentication down for callers
        # using an .env key, so this degrades to "no issued keys" rather
        # than raising. It also must not be cached as success.
        logger.error(f"Could not load issued API keys, falling back to configured keys: {exc}")
        return frozenset()

    with _cache_lock:
        _cache["hashes"] = hashes
        _cache["loaded_at"] = now
    return hashes


def issued_key_is_valid(candidate: str) -> bool:
    """True when `candidate` matches a live, non-revoked issued key."""
    if not candidate or not candidate.startswith(KEY_PREFIX + "_"):
        return False
    digest = hash_key(candidate)
    hashes = _active_hashes()
    if not hashes:
        return False
    # compare_digest against each, rather than a set membership test, so the
    # time taken does not depend on the value presented.
    return any(hmac.compare_digest(digest, known) for known in hashes)


def create_key(db, department: str, created_by: Optional[str] = None,
               note: Optional[str] = None) -> Dict[str, Any]:
    """
    Issue a key for a department.

    Returns the record AND the plaintext under "key". That is the only time
    the plaintext exists outside the caller's hands - it is not stored and
    cannot be shown again.
    """
    from app.models.document import ApiKey

    dept = normalise_department(department)
    if not dept:
        raise ValueError("A department name is required.")

    raw = f"{KEY_PREFIX}_{secrets.token_urlsafe(_TOKEN_BYTES)}"
    record = ApiKey(
        department=dept,
        key_hash=hash_key(raw),
        # Enough to tell two keys apart in a list, not enough to use one.
        prefix=raw[:12],
        note=(note or "").strip()[:255] or None,
        created_by=created_by,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    invalidate_cache()

    logger.info(f"API key issued for department '{dept}' ({record.prefix}...) by {created_by or 'admin'}")
    return {"record": record, "key": raw}


def revoke_key(db, key_id: int) -> bool:
    from app.models.document import ApiKey

    record = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not record or record.revoked_at is not None:
        return False
    record.revoked_at = datetime.utcnow()
    db.commit()
    invalidate_cache()
    logger.info(f"API key {record.prefix}... for '{record.department}' revoked")
    return True


def list_keys(db) -> List[Dict[str, Any]]:
    from app.models.document import ApiKey

    rows = db.query(ApiKey).order_by(ApiKey.created_at.desc()).all()
    return [{
        "id": r.id,
        "department": r.department,
        "prefix": r.prefix,
        "note": r.note,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "created_by": r.created_by,
        "last_used_at": r.last_used_at.isoformat() if r.last_used_at else None,
        "revoked_at": r.revoked_at.isoformat() if r.revoked_at else None,
        "active": r.revoked_at is None,
    } for r in rows]


def touch_last_used(raw_key: str) -> None:
    """
    Record that a key was used, at most once a minute per key.

    Writing on every request would add a database round trip to the hot
    path to store a timestamp nobody reads at that resolution. "Last used
    today" is the question this answers, so minute precision is plenty.
    """
    digest = hash_key(raw_key)
    now = time.time()
    seen = _cache.setdefault("touched", {})
    if now - seen.get(digest, 0) < 60:
        return
    seen[digest] = now
    try:
        from app.core.database import SessionLocal
        from app.models.document import ApiKey

        db = SessionLocal()
        try:
            db.query(ApiKey).filter(ApiKey.key_hash == digest).update(
                {"last_used_at": datetime.utcnow()}, synchronize_session=False)
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.warning(f"Could not record API key usage: {exc}")
