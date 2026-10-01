"""
Administrator role, and real token accounting.

Two things that were quietly wrong:

ANALYTICS WAS OPEN TO ANY SIGNED-IN USER. It aggregates across every
department - document volumes, token spend, per-project activity - which is
a different kind of access from reading one document you were sent.

TOKEN USAGE WAS A CONSTANT. Every document recorded 512/256/768, the
placeholder baked into the envelope template, while Ollama's real counts
(`prompt_eval_count` / `eval_count`) were logged and thrown away. The
analytics "token ledger" was therefore that constant multiplied by the
document count: it moved only when the document count moved. Measured
afterwards, one PAN card costs 5,297 tokens and an Aadhaar letter 6,107 -
roughly seven times the figure being reported.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.database import init_db
from app.core.auth import ROLE_ADMIN, ROLE_VIEWER, authenticate_ui, issue_session, session_role
from app.main import app

init_db()

TEST_API_KEY = "test-key-do-not-use-in-production"
TEST_UI_PASSWORD = "viewer-password"
TEST_ADMIN_PASSWORD = "admin-password"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", TEST_API_KEY, raising=False)
    monkeypatch.setattr(settings, "UI_PASSWORD", TEST_UI_PASSWORD, raising=False)
    monkeypatch.setattr(settings, "ADMIN_USERNAME", "admin", raising=False)
    monkeypatch.setattr(settings, "ADMIN_PASSWORD", TEST_ADMIN_PASSWORD, raising=False)
    monkeypatch.setattr(settings, "SECRET_KEY", "x" * 48, raising=False)
    monkeypatch.setattr(settings, "APP_ENV", "development", raising=False)
    yield


def _signed_in(username: str, password: str) -> TestClient:
    client = TestClient(app)
    response = client.post("/login",
                           data={"username": username, "password": password, "next": "/"},
                           follow_redirects=False)
    assert response.status_code == 303, response.text
    return client


# --------------------------------------------------------------------------- #
# Who gets which role
# --------------------------------------------------------------------------- #

def test_the_admin_account_resolves_to_admin():
    assert authenticate_ui("admin", TEST_ADMIN_PASSWORD) == ROLE_ADMIN


def test_the_shared_password_resolves_to_viewer():
    assert authenticate_ui("", TEST_UI_PASSWORD) == ROLE_VIEWER


def test_the_admin_username_is_case_insensitive():
    assert authenticate_ui("Admin", TEST_ADMIN_PASSWORD) == ROLE_ADMIN


def test_a_wrong_admin_password_is_refused_outright():
    """
    Not quietly demoted to viewer. Someone typing the admin password wrongly
    must not be silently signed in with fewer rights than they asked for.
    """
    assert authenticate_ui("admin", "not-the-password") is None


def test_the_viewer_password_does_not_grant_admin_via_the_admin_username():
    assert authenticate_ui("admin", TEST_UI_PASSWORD) is None


@pytest.mark.parametrize("bad", ["", "wrong", "Hetero", None])
def test_junk_credentials_are_refused(bad):
    assert authenticate_ui("someone", bad or "") is None


# --------------------------------------------------------------------------- #
# The role lives inside the signature
# --------------------------------------------------------------------------- #

def test_a_session_carries_its_role():
    assert session_role(issue_session(ROLE_ADMIN)) == ROLE_ADMIN
    assert session_role(issue_session(ROLE_VIEWER)) == ROLE_VIEWER


def test_the_role_cannot_be_edited_by_the_holder():
    """
    The role is inside the signed payload, so rewriting it invalidates the
    signature rather than granting the rights.
    """
    tampered = issue_session(ROLE_VIEWER).replace("viewer", "admin", 1)
    assert session_role(tampered) is None


def test_a_session_from_before_roles_existed_still_works_as_a_viewer():
    """An upgrade must not sign everybody out."""
    from app.core.auth import _sign
    import time

    legacy = _sign(f"ui.{time.time() + 3600:.0f}")
    assert session_role(legacy) == ROLE_VIEWER


def test_garbage_is_not_a_session():
    for junk in ("", "not-a-cookie", "ui.admin.9999999999.deadbeef"):
        assert session_role(junk) is None


# --------------------------------------------------------------------------- #
# What each role can reach
# --------------------------------------------------------------------------- #

def test_an_admin_sees_analytics():
    client = _signed_in("admin", TEST_ADMIN_PASSWORD)
    assert client.get("/analytics").status_code == 200
    assert client.get("/api/v1/analytics/metrics").status_code == 200


def test_a_viewer_cannot_reach_analytics():
    client = _signed_in("", TEST_UI_PASSWORD)
    # 404 rather than 403: telling a viewer the page exists and is forbidden
    # confirms it is worth attacking.
    assert client.get("/analytics").status_code == 404
    assert client.get("/api/v1/analytics/metrics").status_code == 404


def test_a_viewer_keeps_everything_else():
    """The gate is on analytics only; the viewer role is not a downgrade."""
    client = _signed_in("", TEST_UI_PASSWORD)
    for path in ("/", "/upload", "/history"):
        assert client.get(path).status_code == 200, path


def test_the_analytics_link_is_hidden_from_a_viewer():
    """An offered link that 404s reads as a broken app."""
    assert "/analytics" not in _signed_in("", TEST_UI_PASSWORD).get("/").text
    assert "/analytics" in _signed_in("admin", TEST_ADMIN_PASSWORD).get("/").text


def test_an_api_key_still_reaches_analytics():
    """
    The integrations that poll this are services: they hold a key issued per
    department and have no session to carry a role.
    """
    anonymous = TestClient(app)
    response = anonymous.get("/api/v1/analytics/metrics",
                             headers={"X-API-Key": TEST_API_KEY})
    assert response.status_code == 200


def test_analytics_refuses_an_anonymous_caller():
    assert TestClient(app).get("/api/v1/analytics/metrics").status_code == 401


# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #

def test_usage_starts_empty_and_accumulates():
    from app.services.model_service import (
        get_token_usage, record_token_usage, reset_token_usage,
    )

    reset_token_usage()
    assert get_token_usage() == {"prompt_tokens": 0, "completion_tokens": 0,
                                 "calls": 0, "total_tokens": 0}

    # Two calls per document is the normal shape: transcription, then
    # structured extraction.
    record_token_usage(1707, 116)
    record_token_usage(3330, 144)
    usage = get_token_usage()
    assert usage == {"prompt_tokens": 5037, "completion_tokens": 260,
                     "calls": 2, "total_tokens": 5297}


def test_resetting_clears_the_previous_document():
    from app.services.model_service import (
        get_token_usage, record_token_usage, reset_token_usage,
    )

    reset_token_usage()
    record_token_usage(999, 99)
    reset_token_usage()
    assert get_token_usage()["total_tokens"] == 0


def test_counts_are_per_thread():
    """
    Async jobs run in a ThreadPoolExecutor with four workers. A shared
    counter would bill one document for another's tokens.
    """
    import threading

    from app.services.model_service import (
        get_token_usage, record_token_usage, reset_token_usage,
    )

    reset_token_usage()
    record_token_usage(100, 10)
    other = {}

    def worker():
        reset_token_usage()
        record_token_usage(7000, 700)
        other["total"] = get_token_usage()["total_tokens"]

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert other["total"] == 7700
    assert get_token_usage()["total_tokens"] == 110, "another thread's tokens leaked in"


def test_the_pipeline_records_measured_usage_not_the_placeholder():
    """
    512/256/768 is the envelope template's placeholder. If it reappears in
    the stored record, the real counts have stopped being read again.
    """
    import inspect

    from app.services.ocr_pipeline import OCRPipeline

    src = inspect.getsource(OCRPipeline.process_file)
    assert "get_token_usage()" in src
    assert "reset_token_usage()" in src
    assert 'meta_tokens.get("prompt_tokens", 512)' not in src


def test_the_model_service_records_both_call_sites():
    """Transcription and structured extraction are separate calls; the
    second is often the larger."""
    import inspect

    from app.services import model_service

    src = inspect.getsource(model_service)
    assert src.count("record_token_usage(") >= 3  # definition + two call sites
