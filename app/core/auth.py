"""
Authentication and signed-link issuing.

Three separate things guard three separate doors, because they have different
callers:

  * API keys       - other departments' services calling /api/v1/*.
  * Session cookie - a person using the browser viewer.
  * Media tokens   - a single image, for a short time, embedded in a page the
                     viewer already authenticated to fetch.

Everything is HMAC-SHA256 over stdlib. No new dependency, nothing to vet, and
no key material at rest beyond SECRET_KEY.

WHY MEDIA TOKENS EXIST: /uploads and /outputs used to be public static mounts,
so every Aadhaar and PAN scan ever processed could be fetched by anyone who
reached the host - no credential, just the filename. The viewer still needs to
show those images, so it now asks for a short-lived signature per file instead
of the directory being world-readable.
"""

import hashlib
import hmac
import secrets
import time
from typing import Optional

# pyrefly: ignore [missing-import]
from fastapi import Header, HTTPException, Query, Request

from app.core.config import settings
from app.core.logger import logger

_SEP = "."


def _secret() -> bytes:
    """
    The signing secret.

    In production a missing secret is a hard failure - signing with a
    predictable value is worse than not signing, because it looks protected.
    In development a random per-process secret keeps things working; links and
    sessions simply do not survive a restart, which is the correct trade there.
    """
    if settings.SECRET_KEY:
        return settings.SECRET_KEY.encode("utf-8")
    if settings.is_production():
        raise RuntimeError("SECRET_KEY must be set in production")
    global _DEV_SECRET
    try:
        return _DEV_SECRET
    except NameError:
        _DEV_SECRET = secrets.token_bytes(32)
        logger.warning(
            "SECRET_KEY is unset - using a random development secret. "
            "Sessions and media links will not survive a restart."
        )
        return _DEV_SECRET


def _sign(payload: str) -> str:
    digest = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}{_SEP}{digest}"


def _unsign(token: str) -> Optional[str]:
    """Return the payload if the signature is valid and unexpired, else None."""
    if not token or _SEP not in token:
        return None
    payload, _, digest = token.rpartition(_SEP)
    expected = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    # compare_digest, not == : a plain comparison leaks the correct prefix
    # length through timing.
    if not hmac.compare_digest(expected, digest):
        return None
    _, _, expiry = payload.rpartition(_SEP)
    try:
        if float(expiry) < time.time():
            return None
    except ValueError:
        return None
    return payload


# --------------------------------------------------------------------------- #
# API keys
# --------------------------------------------------------------------------- #

def _key_is_valid(candidate: str) -> bool:
    """
    Two sources, both accepted.

    Keys in `.env` are the BOOTSTRAP credential - something has to be able
    to call the API before anyone has logged in to issue a key - and they
    keep working exactly as before. Keys issued from the admin screen live
    in the database as hashes and are checked second, against a short-lived
    in-process cache so this stays cheap on the hot path.
    """
    if not candidate:
        return False

    keys = settings.api_key_set()
    # Check every configured key rather than short-circuiting, so the time
    # taken does not depend on which one matched.
    if keys and any(hmac.compare_digest(candidate, k) for k in keys):
        return True

    try:
        from app.services.api_key_service import issued_key_is_valid, touch_last_used
        if issued_key_is_valid(candidate):
            touch_last_used(candidate)
            return True
    except Exception as exc:
        # An issued-key lookup failing must not revoke access for callers
        # holding a configured key, which was already answered above.
        logger.error(f"Issued API key check failed: {exc}")

    return False


async def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = Query(None, description="Alternative to the X-API-Key header"),
) -> str:
    """
    FastAPI dependency guarding /api/v1/*.

    Accepts the key as a header (preferred) or a query parameter, because some
    callers - browser tools, a <img src>, a quick curl - cannot set headers.
    """
    if not settings.api_key_set():
        if settings.is_production():
            # Fail closed. An empty key list in production means misconfigured,
            # not "allow everyone".
            raise HTTPException(status_code=503, detail="Service is not configured for authenticated access.")
        return "dev-open"

    supplied = x_api_key or api_key
    if not supplied or not _key_is_valid(supplied):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key. Send it as the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return supplied


# --------------------------------------------------------------------------- #
# Browser session
# --------------------------------------------------------------------------- #

SESSION_COOKIE = "di_session"

# Two roles, because two different things are being protected.
#
#   viewer - may read documents they were sent. This is what UI_PASSWORD
#            has always granted.
#   admin  - additionally sees analytics, which aggregates ACROSS every
#            department: volumes, token spend, per-project activity. That is
#            a different kind of access from reading one document, so it
#            gets a different credential.
ROLE_VIEWER = "viewer"
ROLE_ADMIN = "admin"


def issue_session(role: str = ROLE_VIEWER) -> str:
    """
    Sign a session carrying its role.

    The role is INSIDE the signed payload, not a separate cookie, so it
    cannot be edited by the holder: changing "viewer" to "admin" invalidates
    the signature.
    """
    role = ROLE_ADMIN if role == ROLE_ADMIN else ROLE_VIEWER
    expiry = time.time() + settings.SESSION_TTL_SECONDS
    return _sign(f"ui{_SEP}{role}{_SEP}{expiry:.0f}")


def session_role(cookie: Optional[str]) -> Optional[str]:
    """The role of a valid session, or None when there is not one."""
    payload = _unsign(cookie or "")
    if not payload or not payload.startswith("ui" + _SEP):
        return None
    parts = payload.split(_SEP)
    # Sessions issued before roles existed were "ui.<expiry>". They stay
    # valid as viewers rather than being logged out by an upgrade.
    if len(parts) < 3:
        return ROLE_VIEWER
    return ROLE_ADMIN if parts[1] == ROLE_ADMIN else ROLE_VIEWER


def session_is_valid(cookie: Optional[str]) -> bool:
    return session_role(cookie) is not None


def ui_auth_required() -> bool:
    """False when no UI password is configured (development convenience)."""
    return bool(settings.UI_PASSWORD)


def check_ui_password(supplied: str) -> bool:
    if not settings.UI_PASSWORD:
        return True
    return hmac.compare_digest(supplied or "", settings.UI_PASSWORD)


def authenticate_ui(username: str, password: str) -> Optional[str]:
    """
    Resolve a sign-in to a role, or None when the credentials do not match.

    The admin check is tried first and is the only path that returns admin.
    Both comparisons use compare_digest so neither leaks its correct prefix
    length through timing.
    """
    supplied_user = (username or "").strip().lower()
    admin_user = (settings.ADMIN_USERNAME or "admin").strip().lower()

    if settings.ADMIN_PASSWORD and supplied_user == admin_user:
        if hmac.compare_digest(password or "", settings.ADMIN_PASSWORD):
            return ROLE_ADMIN
        # A wrong password on the admin account is not then offered the
        # viewer role as a consolation prize.
        return None

    if check_ui_password(password):
        return ROLE_VIEWER
    return None


def request_is_admin(request: Request) -> bool:
    """
    True when this caller may see analytics.

    An API key counts. Analytics is a reporting endpoint and the integrations
    that poll it are services, not people - they hold a key issued per
    department and have no session to carry a role. What this gate is for is
    stopping an ordinary VIEWER, who signed in to read a document, from
    reading the whole organisation's activity.
    """
    if session_role(request.cookies.get(SESSION_COOKIE)) == ROLE_ADMIN:
        return True
    supplied = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if supplied and _key_is_valid(supplied):
        return True
    # With no admin password configured there is no admin role to withhold,
    # so the gate would lock everyone out of a page they used to have.
    return not settings.ADMIN_PASSWORD


def request_has_ui_access(request: Request) -> bool:
    """
    True when the caller may see the viewer and its media.

    An API key counts: a service that is already trusted to pull the extracted
    JSON is trusted to pull the page image it came from.
    """
    if not ui_auth_required():
        return True
    if session_is_valid(request.cookies.get(SESSION_COOKIE)):
        return True
    supplied = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    return bool(supplied and _key_is_valid(supplied))


# --------------------------------------------------------------------------- #
# Signed media links
# --------------------------------------------------------------------------- #

def issue_media_token(filename: str, ttl_seconds: Optional[int] = None) -> str:
    """
    A signature covering ONE filename for a short window.

    The filename is part of the signed payload, so a token minted for a
    thumbnail cannot be replayed against someone else's Aadhaar scan.
    """
    ttl = settings.MEDIA_TOKEN_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    expiry = time.time() + ttl
    digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]
    return _sign(f"{digest}{_SEP}{expiry:.0f}")


def media_token_is_valid(filename: str, token: Optional[str]) -> bool:
    payload = _unsign(token or "")
    if not payload:
        return False
    expected = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]
    return hmac.compare_digest(payload.split(_SEP)[0], expected)


async def require_admin(request: Request) -> str:
    """
    FastAPI dependency for analytics.

    Admin session or API key. Returns 404 rather than 403 for a signed-in
    viewer, for the same reason the media route does: 403 confirms the page
    exists and is worth attacking, while 404 tells someone who should not
    have it exactly as much as they need to know.
    """
    if request_is_admin(request):
        return ROLE_ADMIN
    if session_role(request.cookies.get(SESSION_COOKIE)) is not None:
        raise HTTPException(status_code=404, detail="Not found.")
    raise HTTPException(
        status_code=401,
        detail="Analytics requires an administrator session or an API key.",
        headers={"WWW-Authenticate": "ApiKey"},
    )
