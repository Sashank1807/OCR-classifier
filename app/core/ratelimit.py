"""
Per-client request ceiling.

A single 50MB upload buys 30-90 seconds of CPU and VLM time on this box, and
the submit endpoint needs no credential beyond an API key, so a caller in a
loop can saturate the machine without trying. A fixed window per client is
crude but it is the part that matters: it bounds the damage.

In-memory and per-process, so it is a guard rail rather than a quota - with
several uvicorn workers each holds its own counters. That is the honest limit
of doing this without Redis, and it is still far better than no ceiling.
"""

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict

# pyrefly: ignore [missing-import]
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logger import logger

_WINDOW_SECONDS = 60.0

_hits: Dict[str, Deque[float]] = defaultdict(deque)
_lock = threading.Lock()

# Liveness and static assets are polled by monitors and browsers; counting
# them would exhaust a client's budget on things that cost nothing to serve.
_EXEMPT_PREFIXES = ("/health", "/static", "/favicon")


def _client_id(request: Request) -> str:
    """
    Identify the caller by API key when there is one, else by address.

    Keying on the API key means one department's loop cannot consume another
    department's allowance, and that several services behind one NAT are not
    lumped together.
    """
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if key:
        return f"key:{key[:12]}"
    fwd = request.headers.get("X-Forwarded-For")
    if fwd:
        return f"ip:{fwd.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def _limit_for(request: Request) -> int:
    path = request.url.path
    if request.method == "POST" and ("/process" in path or "/reprocess" in path):
        return settings.UPLOAD_RATE_LIMIT_PER_MINUTE
    return settings.RATE_LIMIT_PER_MINUTE


async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith(_EXEMPT_PREFIXES):
        return await call_next(request)

    limit = _limit_for(request)
    if limit <= 0:
        return await call_next(request)

    cid = _client_id(request)
    now = time.time()
    cutoff = now - _WINDOW_SECONDS

    with _lock:
        bucket = _hits[cid]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(bucket[0] + _WINDOW_SECONDS - now))
            logger.warning(f"Rate limit hit for {cid} on {request.method} {path} ({limit}/min)")
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": "Too Many Requests",
                    "detail": f"Limit is {limit} requests per minute for this endpoint.",
                },
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)

        # Buckets are keyed by caller, so an unbounded key space is a slow
        # memory leak under churning client addresses. Sweep the idle ones.
        if len(_hits) > 2048:
            for k in [k for k, v in _hits.items() if not v or v[-1] < cutoff]:
                _hits.pop(k, None)

    return await call_next(request)
