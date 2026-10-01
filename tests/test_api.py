"""
API surface tests.

Every route here is now behind a credential. The tests therefore assert two
things per endpoint - that it REFUSES an anonymous caller, and that it works
for an authenticated one. Testing only the happy path would let the guard be
removed without a single test going red, which is exactly the regression
worth preventing: /uploads and /outputs were publicly readable for months.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.database import init_db
from app.main import app

init_db()
client = TestClient(app)

# A key and password just for the tests, installed before the app reads them.
TEST_API_KEY = "test-key-do-not-use-in-production"
TEST_UI_PASSWORD = "test-ui-password"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", TEST_API_KEY, raising=False)
    monkeypatch.setattr(settings, "UI_PASSWORD", TEST_UI_PASSWORD, raising=False)
    monkeypatch.setattr(settings, "SECRET_KEY", "x" * 48, raising=False)
    # The session cookie is marked Secure in production, so a browser - and
    # TestClient - will not send it back over plain http. Tests run as
    # development for that reason; the Secure flag itself is asserted below.
    monkeypatch.setattr(settings, "APP_ENV", "development", raising=False)
    yield


def api(path: str, **kw):
    return client.get(path, headers={"X-API-Key": TEST_API_KEY}, **kw)


def ui_client() -> TestClient:
    """A client holding a valid viewer session."""
    c = TestClient(app)
    r = c.post("/login", data={"password": TEST_UI_PASSWORD, "next": "/"},
               follow_redirects=False)
    assert r.status_code == 303, r.text
    return c


# --------------------------------------------------------------------------- #
# The guards themselves
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("path", [
    "/api/v1/settings/",
    "/api/v1/analytics/metrics",
    "/api/v1/analytics/metrics/hr",
    "/api/v1/history/search",
    "/api/v1/ocr/status/req_nonexistent",
    "/api/v1/ocr/document/1",
    "/api/v1/ocr/erp/1",
])
def test_api_routes_reject_anonymous_callers(path):
    assert client.get(path).status_code == 401, f"{path} is not guarded"


def test_api_routes_reject_a_wrong_key():
    r = client.get("/api/v1/analytics/metrics", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


@pytest.mark.parametrize("path", ["/", "/upload", "/history", "/analytics"])
def test_viewer_pages_redirect_anonymous_callers_to_login(path):
    r = client.get(path, follow_redirects=False)
    assert r.status_code == 307, f"{path} is not gated"
    assert "/login" in r.headers.get("location", "")


def test_media_is_not_publicly_mounted():
    """
    /uploads and /outputs must not be served as static directories. This is
    the regression that mattered most: every Aadhaar and PAN scan processed
    was downloadable by anyone who could reach the host.
    """
    for path in ("/uploads/anything.jpg", "/outputs/anything.png"):
        assert client.get(path).status_code == 404, f"{path} is publicly mounted again"


def test_media_route_requires_a_token_or_credentials():
    r = client.get("/media/uploads/anything.jpg")
    assert r.status_code == 404


def test_media_token_is_bound_to_one_filename():
    from app.core.auth import issue_media_token, media_token_is_valid
    token = issue_media_token("mine.jpg")
    assert media_token_is_valid("mine.jpg", token)
    assert not media_token_is_valid("someone_elses.jpg", token)
    assert not media_token_is_valid("mine.jpg", token + "x")
    assert not media_token_is_valid("mine.jpg", None)


def test_media_token_expires():
    from app.core.auth import issue_media_token, media_token_is_valid
    assert not media_token_is_valid("mine.jpg", issue_media_token("mine.jpg", ttl_seconds=-1))


def test_media_path_traversal_is_blocked():
    assert client.get("/media/uploads/../../.env").status_code == 404
    assert client.get("/media/nosuchkind/x.jpg").status_code == 404


# --------------------------------------------------------------------------- #
# Authenticated behaviour
# --------------------------------------------------------------------------- #

def test_dashboard_endpoint():
    response = ui_client().get("/")
    assert response.status_code == 200
    assert "Document Intelligence Dashboard" in response.text


def test_upload_view_endpoint():
    response = ui_client().get("/upload")
    assert response.status_code == 200
    assert "Upload Document for Intelligence Extraction" in response.text


def test_history_view_endpoint():
    response = ui_client().get("/history")
    assert response.status_code == 200
    assert "OCR Document History" in response.text


def test_session_cookie_is_hardened(monkeypatch):
    """HttpOnly always; Secure once deployed, so it never crosses plain http."""
    monkeypatch.setattr(settings, "APP_ENV", "production", raising=False)
    c = TestClient(app)
    r = c.post("/login", data={"password": TEST_UI_PASSWORD, "next": "/"},
               follow_redirects=False)
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert "secure" in cookie


def test_login_rejects_a_wrong_password():
    r = client.post("/login", data={"password": "nope"}, follow_redirects=False)
    assert r.status_code == 401


def test_login_will_not_redirect_off_site():
    """An absolute `next` would make the login form an open redirect."""
    c = TestClient(app)
    r = c.post("/login", data={"password": TEST_UI_PASSWORD, "next": "https://evil.example/x"},
               follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/"


def test_settings_api_get():
    response = api("/api/v1/settings/")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "max_upload_size_mb" in data["settings"]


def test_analytics_metrics_api():
    response = api("/api/v1/analytics/metrics")
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_department_analytics_metrics_api():
    response = api("/api/v1/analytics/metrics/hr")
    assert response.status_code == 200
    assert response.json()["success"] is True


# --------------------------------------------------------------------------- #
# Ops
# --------------------------------------------------------------------------- #

def test_health_is_public_and_cheap():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readiness_reports_its_checks():
    r = client.get("/readiness")
    assert r.status_code in (200, 503)
    assert set(r.json()["checks"]) == {"database", "model"}


def test_production_config_validation_catches_unsafe_settings(monkeypatch):
    """Startup refuses these in production rather than serving openly."""
    monkeypatch.setattr(settings, "SECRET_KEY", "", raising=False)
    monkeypatch.setattr(settings, "API_KEYS", "", raising=False)
    monkeypatch.setattr(settings, "UI_PASSWORD", "", raising=False)
    monkeypatch.setattr(settings, "DEBUG", True, raising=False)
    problems = settings.validate_for_production()
    assert len(problems) == 4
    joined = " ".join(problems).lower()
    for expected in ("secret_key", "api_keys", "ui_password", "debug"):
        assert expected in joined


def test_error_responses_do_not_leak_internals(monkeypatch):
    import asyncio, json as _json
    from app.core.exceptions import global_exception_handler

    monkeypatch.setattr(settings, "DEBUG", False, raising=False)

    class _Req:
        method = "GET"
        class url:
            path = "/api/v1/ocr/boom"

    resp = asyncio.new_event_loop().run_until_complete(
        global_exception_handler(_Req(), RuntimeError(r"cannot open D:\secret\creds.db"))
    )
    body = _json.loads(resp.body)
    assert "secret" not in body["detail"].lower()
    assert body["incident_id"] in body["detail"]


# --------------------------------------------------------------------------- #
# The ERP response format
#
# These fix the request contract rather than the payload's contents, which
# tests/test_erp_payload.py covers: an unknown format must be REFUSED rather
# than quietly served the full envelope, and asking for `erp` must not leave
# diagnostics in the response.
# --------------------------------------------------------------------------- #

SEED_PROJECT = "ErpContractTest"


@pytest.fixture(scope="module", autouse=True)
def _clean_up_seeded_documents():
    """
    Remove the documents these tests insert.

    They go into the real database, and a row left as PROCESSING is not inert:
    startup re-queues interrupted jobs, so every suite run would hand the
    worker pool another document whose file never existed.
    """
    yield

    from app.core.database import SessionLocal
    from app.models.document import DocumentRecord, DocumentPage, DocumentResult

    db = SessionLocal()
    try:
        ids = [r.id for r in db.query(DocumentRecord)
               .filter(DocumentRecord.project_name == SEED_PROJECT).all()]
        if ids:
            db.query(DocumentPage).filter(DocumentPage.document_id.in_(ids)).delete(
                synchronize_session=False)
            db.query(DocumentResult).filter(DocumentResult.document_id.in_(ids)).delete(
                synchronize_session=False)
            db.query(DocumentRecord).filter(DocumentRecord.id.in_(ids)).delete(
                synchronize_session=False)
            db.commit()
    finally:
        db.close()


def _seed_document(doc_type="PAN", structured=None, status="COMPLETED", with_result=True):
    """
    Insert one document straight into the database; return (id, request_id).

    stored_filename is unique, so every call needs its own name - otherwise
    the second document seeded in a run collides instead of testing anything.
    """
    import json as _json
    import uuid as _uuid

    from app.core.database import SessionLocal
    from app.models.document import DocumentRecord, DocumentResult

    tag = _uuid.uuid4().hex[:10]
    db = SessionLocal()
    try:
        record = DocumentRecord(
            original_filename=f"erp_contract_{tag}.jpg",
            stored_filename=f"erp_contract_{tag}.jpg",
            file_path=f"uploads/erp_contract_{tag}.jpg",
            file_type="jpg",
            file_size=1024,
            document_type=doc_type,
            status=status,
            project_name=SEED_PROJECT,
            request_id=f"req_erp{tag}",
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        if not with_result:
            return record.id, record.request_id
        envelope = structured if structured is not None else {
            "document_type": doc_type,
            "classification_confidence": 0.9,
            "overall_confidence": 1.0,
            "needs_manual_review": False,
            "field_verification": {"counts": {}, "flagged": []},
            "column_model_uncertain": False,
            "fields": {"name": "Sanjay Kumar Prajapati", "pan_number": "BJQPP5524G"},
        }
        # structured_json is a read-only property over this column.
        db.add(DocumentResult(
            document_id=record.id,
            markdown_text="# Heading\n\nsome text",
            structured_json_str=_json.dumps(envelope, ensure_ascii=False),
        ))
        db.commit()
        return record.id, record.request_id
    finally:
        db.close()


def test_erp_endpoint_returns_only_business_data():
    doc_id, req_id = _seed_document()
    r = api(f"/api/v1/ocr/erp/{doc_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["response_format"] == "erp"
    payload = body["data"]

    assert payload["data"] == {"name": "Sanjay Kumar Prajapati", "pan_number": "BJQPP5524G"}
    assert payload["document_type"] == "PAN"
    assert payload["review_required"] is False
    # No reviewer-facing material anywhere in the response.
    for leaked in ("markdown", "field_verification", "classification_confidence",
                   "column_model_uncertain", "pages", "structured_data"):
        assert leaked not in payload, f"{leaked} leaked into the ERP payload"
    assert "Heading" not in r.text


@pytest.mark.parametrize("path", [
    "/api/v1/ocr/document/{id}?response_format=erp",
    "/api/v1/ocr/status/{req}?response_format=erp",
])
def test_response_format_erp_is_accepted_on_the_read_endpoints(path):
    doc_id, req_id = _seed_document()
    r = api(path.format(id=doc_id, req=req_id))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["response_format"] == "erp"
    assert "field_verification" not in body["data"]
    assert "document" not in body


def test_the_full_format_stays_the_default():
    """An existing caller that sends no response_format must see no change."""
    doc_id, req_id = _seed_document()
    body = api(f"/api/v1/ocr/document/{doc_id}").json()
    assert "document" in body
    assert "markdown" in body["document"]
    assert "structured_data" in body["document"]


@pytest.mark.parametrize("bad", ["erpdata", "slim", "ERP-DATA", "json"])
def test_an_unknown_response_format_is_refused_not_silently_ignored(bad):
    """
    Serving the full envelope for a misspelled `erp` would look like the slim
    payload does not work, and send the caller looking in the wrong place.
    """
    doc_id, req_id = _seed_document()
    r = api(f"/api/v1/ocr/document/{doc_id}?response_format={bad}")
    assert r.status_code == 400
    assert "response_format" in r.json()["detail"]


def test_erp_export_format_matches_the_api_payload():
    """A mapping built from a downloaded sample must hold when it goes live."""
    import json as _json

    doc_id, req_id = _seed_document()
    exported = _json.loads(api(f"/api/v1/ocr/export/{doc_id}?format=erp").text)
    served = api(f"/api/v1/ocr/erp/{doc_id}").json()["data"]
    assert exported == served


def test_erp_endpoint_will_not_serve_an_unfinished_document():
    """
    A 200 with an empty payload reads as "the document had no data". A caller
    polling for a result needs to be told it is not ready yet.
    """
    doc_id, _ = _seed_document(status="PROCESSING", with_result=False)
    r = api(f"/api/v1/ocr/erp/{doc_id}")
    assert r.status_code == 409
    assert "still processing" in r.json()["detail"].lower()


def test_erp_payload_is_materially_smaller_than_the_full_one():
    """The point of the format. A stock statement is where it matters most."""
    doc_id, _ = _seed_document(doc_type="SPREADSHEET", structured={
        "document_type": "SPREADSHEET",
        "classification_confidence": 1.0,
        "overall_confidence": 1.0,
        "needs_manual_review": False,
        "field_verification": {
            "counts": {"exact": 20, "verified": 0, "unchecked": 0, "flagged": 0},
            "field_count": 20,
            "source_legibility": {"legible": True, "reasons": [], "recognised_words": 15},
            "flagged": [],
        },
        "column_model_uncertain": False,
        "fields": {
            "sheet_title": "Sheet1",
            "columns": ["Product", "Qty", "Price"],
            "rows": [["Medicine A", "10", "100.5"], ["Medicine B", "20", "200.75"]],
            "all_tables": [{
                "table_index": 1,
                "columns": ["Product", "Qty", "Price"],
                "rows": [["Medicine A", "10", "100.5"], ["Medicine B", "20", "200.75"]],
            }],
        },
    })
    full = len(api(f"/api/v1/ocr/document/{doc_id}").text)
    erp = len(api(f"/api/v1/ocr/erp/{doc_id}").text)
    assert erp < full * 0.6, f"erp={erp} bytes vs full={full} bytes"


def test_process_accepts_response_format_as_a_form_field(monkeypatch):
    """
    The one binding the read endpoints do not cover.

    /process reads `await request.form()` itself to find project_name under
    any of its aliases, so a Form parameter added to the signature could
    plausibly fail to bind. This drives the real route with the pipeline
    stubbed out, because the point is the request contract, not the OCR.
    """
    from app.core import ratelimit
    from app.services import ocr_pipeline

    # The upload limiter is 10/minute per key and every test here shares one
    # key, so by this point in the file the bucket is already partly spent.
    # Clear it - this test is about the request contract, not the limiter,
    # which has its own coverage.
    ratelimit._hits.clear()

    canned = {
        "id": None,
        "filename": "stub.jpg",
        "document_type": "PAN",
        "markdown": "# Heading",
        "structured_data": {
            "document_type": "PAN",
            "overall_confidence": 1.0,
            "needs_manual_review": False,
            "field_verification": {"counts": {}, "flagged": []},
            "fields": {"name": "Sanjay Kumar Prajapati", "pan_number": "BJQPP5524G"},
        },
        "tables": [{"columns": ["x"], "rows": [["y"]]}],
        "processing_time": 1.0,
        "status": "COMPLETED",
        "created_at": "2026-09-30T00:00:00",
    }

    doc_id, _ = _seed_document()

    def fake_process_file(db, path, filename, **kw):
        return dict(canned, id=doc_id)

    monkeypatch.setattr(ocr_pipeline.pipeline, "process_file", fake_process_file)

    def submit(fmt):
        return client.post(
            "/api/v1/ocr/process",
            headers={"X-API-Key": TEST_API_KEY},
            files={"file": ("stub.jpg", b"\xff\xd8\xff\xe0stub-jpeg-bytes", "image/jpeg")},
            data={"async_mode": "false", "response_format": fmt},
        )

    r = submit("erp")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["response_format"] == "erp"
    assert body["data"]["data"] == {"name": "Sanjay Kumar Prajapati",
                                    "pan_number": "BJQPP5524G"}
    assert "markdown" not in body["data"]
    assert "tables" not in body["data"]

    r = submit("full")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["response_format"] == "full"
    assert "markdown" in body["data"]
    assert "tables" not in body["data"]

    assert submit("slim").status_code == 400
