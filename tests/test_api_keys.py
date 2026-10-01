"""
Runtime-issued department API keys.

Adding a department used to mean editing `API_KEYS` in `.env` and
restarting. These are issued from the admin screen and work on the next
request.

The property that matters most is that **the plaintext is never stored**.
An API key is a bearer credential: whoever holds it can read every extracted
document. Keeping it in a table means a database dump, a backup file or a
careless SELECT hands over live access to the whole corpus, and verification
never needs to read one back - only to compare. So it is shown once and
exists nowhere on the server afterwards.
"""

import pytest
from fastapi.testclient import TestClient

import app.models.document  # noqa: F401 - registers ApiKey before create_all
from app.core.auth import _key_is_valid
from app.core.config import settings
from app.core.database import SessionLocal, init_db
from app.main import app
from app.models.document import ApiKey
from app.services.api_key_service import (
    KEY_PREFIX,
    create_key,
    hash_key,
    invalidate_cache,
    normalise_department,
    revoke_key,
)

init_db()

CONFIGURED_KEY = "configured-bootstrap-key"
TEST_UI_PASSWORD = "viewer-password"
TEST_ADMIN_PASSWORD = "admin-password"
SEED_DEPT = "ApiKeyTest Department"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", CONFIGURED_KEY, raising=False)
    monkeypatch.setattr(settings, "UI_PASSWORD", TEST_UI_PASSWORD, raising=False)
    monkeypatch.setattr(settings, "ADMIN_USERNAME", "admin", raising=False)
    monkeypatch.setattr(settings, "ADMIN_PASSWORD", TEST_ADMIN_PASSWORD, raising=False)
    monkeypatch.setattr(settings, "SECRET_KEY", "k" * 48, raising=False)
    monkeypatch.setattr(settings, "APP_ENV", "development", raising=False)
    invalidate_cache()
    yield
    invalidate_cache()


@pytest.fixture(autouse=True)
def _clean_up_issued_keys():
    """
    Remove the keys these tests issue.

    They go into the real table, and a live key left behind is a working
    credential nobody meant to grant.
    """
    yield
    db = SessionLocal()
    try:
        db.query(ApiKey).filter(ApiKey.department.like("ApiKeyTest%")).delete(
            synchronize_session=False)
        db.commit()
    finally:
        db.close()
    invalidate_cache()


def _admin() -> TestClient:
    client = TestClient(app)
    r = client.post("/login",
                    data={"username": "admin", "password": TEST_ADMIN_PASSWORD, "next": "/"},
                    follow_redirects=False)
    assert r.status_code == 303
    return client


def _viewer() -> TestClient:
    client = TestClient(app)
    client.post("/login",
                data={"username": "", "password": TEST_UI_PASSWORD, "next": "/"},
                follow_redirects=False)
    return client


# --------------------------------------------------------------------------- #
# The plaintext is never stored
# --------------------------------------------------------------------------- #

def test_only_a_hash_is_persisted():
    db = SessionLocal()
    try:
        issued = create_key(db, SEED_DEPT, created_by="test")
        raw = issued["key"]
        row = db.query(ApiKey).filter(ApiKey.id == issued["record"].id).first()

        assert row.key_hash == hash_key(raw)
        assert raw not in row.key_hash
        # Nothing on the row carries enough to reconstruct the key.
        for column in (row.key_hash, row.prefix, row.department, row.note or ""):
            assert column != raw
        # The prefix identifies, it does not authenticate.
        assert row.prefix in raw and len(row.prefix) < len(raw)
    finally:
        db.close()


def test_the_prefix_alone_is_not_a_working_key():
    db = SessionLocal()
    try:
        issued = create_key(db, SEED_DEPT)
        assert _key_is_valid(issued["key"]) is True
        assert _key_is_valid(issued["record"].prefix) is False
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #

def test_a_new_key_works_immediately():
    """No restart, no config edit - the point of the feature."""
    db = SessionLocal()
    try:
        raw = create_key(db, SEED_DEPT)["key"]
    finally:
        db.close()

    fresh = TestClient(app)
    assert fresh.get("/api/v1/analytics/metrics",
                     headers={"X-API-Key": raw}).status_code == 200


def test_revoking_stops_it_on_the_next_request():
    db = SessionLocal()
    try:
        issued = create_key(db, SEED_DEPT)
        raw = issued["key"]
        assert _key_is_valid(raw) is True
        assert revoke_key(db, issued["record"].id) is True
    finally:
        db.close()

    assert _key_is_valid(raw) is False
    assert TestClient(app).get("/api/v1/analytics/metrics",
                               headers={"X-API-Key": raw}).status_code == 401


def test_a_revoked_key_is_kept_for_the_audit_trail():
    """Deleted rows cannot answer 'who had access, and until when'."""
    db = SessionLocal()
    try:
        issued = create_key(db, SEED_DEPT)
        revoke_key(db, issued["record"].id)
        row = db.query(ApiKey).filter(ApiKey.id == issued["record"].id).first()
        assert row is not None
        assert row.revoked_at is not None
        assert row.is_active is False
    finally:
        db.close()


def test_revoking_twice_is_refused():
    db = SessionLocal()
    try:
        issued = create_key(db, SEED_DEPT)
        assert revoke_key(db, issued["record"].id) is True
        assert revoke_key(db, issued["record"].id) is False
    finally:
        db.close()


def test_configured_env_keys_keep_working():
    """
    They are the bootstrap credential: something has to call the API before
    anyone has logged in to issue a key.
    """
    assert _key_is_valid(CONFIGURED_KEY) is True


def test_an_unrelated_string_is_not_a_key():
    for junk in ("", "nonsense", KEY_PREFIX + "_", "dik_" + "a" * 43):
        assert _key_is_valid(junk) is False


# --------------------------------------------------------------------------- #
# Only an administrator may issue one
# --------------------------------------------------------------------------- #

def test_an_admin_can_issue_a_key():
    response = _admin().post("/api/v1/analytics/api-keys",
                             json={"department": SEED_DEPT})
    assert response.status_code == 200
    body = response.json()
    assert body["api_key"].startswith(KEY_PREFIX + "_")
    assert body["key"]["department"] == SEED_DEPT
    assert "cannot be shown again" in body["message"]


def test_a_viewer_cannot_issue_a_key():
    assert _viewer().post("/api/v1/analytics/api-keys",
                          json={"department": "ApiKeyTest Sneaky"}).status_code == 404


def test_an_anonymous_caller_cannot_issue_a_key():
    assert TestClient(app).post("/api/v1/analytics/api-keys",
                                json={"department": "ApiKeyTest Anon"}).status_code == 401


def test_a_viewer_cannot_list_or_revoke():
    viewer = _viewer()
    assert viewer.get("/api/v1/analytics/api-keys").status_code == 404
    assert viewer.delete("/api/v1/analytics/api-keys/1").status_code == 404


@pytest.mark.parametrize("bad", ["", "   ", "a"])
def test_a_department_name_is_required(bad):
    assert _admin().post("/api/v1/analytics/api-keys",
                         json={"department": bad}).status_code == 400


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #

def test_the_listing_never_carries_a_plaintext_key():
    admin = _admin()
    raw = admin.post("/api/v1/analytics/api-keys",
                     json={"department": SEED_DEPT}).json()["api_key"]

    listing = admin.get("/api/v1/analytics/api-keys")
    assert listing.status_code == 200
    assert raw not in listing.text, "a plaintext key leaked into the listing"
    assert any(k["department"] == SEED_DEPT for k in listing.json()["api_keys"])


def test_department_names_are_tidied_not_mangled():
    """
    Whitespace collapses, but the label a human chose is otherwise kept -
    it has to be recognisable in the list they revoke from.
    """
    assert normalise_department("  Radiology   Dept  ") == "Radiology Dept"
    assert normalise_department("HR") == "HR"
    assert normalise_department("Cardiology & ICU") == "Cardiology & ICU"
    assert normalise_department("") == ""


def test_two_keys_for_one_department_are_distinct():
    db = SessionLocal()
    try:
        first = create_key(db, SEED_DEPT)["key"]
        second = create_key(db, SEED_DEPT)["key"]
        assert first != second
        assert _key_is_valid(first) and _key_is_valid(second)
    finally:
        db.close()
