import os
import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.document import DocumentRecord, DocumentResult, DocumentPage
from app.services.preprocessor import ImagePreprocessor
from app.services.export_service import ExportService
from app.utils.file_utils import parse_json_from_llm_response, extract_markdown_tables
from app.utils.structured_schemas import normalize_structured_data

# Test Database setup
TEST_DB_URL = "sqlite:///./test_ocr_pipeline.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def setup_module():
    Base.metadata.create_all(bind=engine)


def teardown_module():
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    if os.path.exists("./test_ocr_pipeline.db"):
        try:
            os.remove("./test_ocr_pipeline.db")
        except Exception:
            pass


def test_parse_json_from_llm_response():
    raw_markdown_json = "```json\n{\"document_type\": \"Invoice\", \"language\": \"English\"}\n```"
    parsed = parse_json_from_llm_response(raw_markdown_json)
    assert parsed.get("document_type") == "Invoice"
    assert parsed.get("language") == "English"


def test_extract_markdown_tables():
    md = """
| Name | Role | Salary |
| --- | --- | --- |
| Alice | Architect | $120,000 |
| Bob | Lead Dev | $110,000 |
"""
    tables = extract_markdown_tables(md)
    assert len(tables) == 1
    assert len(tables[0]) == 3  # Header + 2 data rows
    assert tables[0][1][0] == "Alice"


def test_convert_markdown_table_to_spreadsheet_json():
    from app.utils.file_utils import convert_markdown_table_to_spreadsheet_json
    md = """
| Topic | Cluster | Date |
| --- | --- | --- |
| Teams | Cluster E | 20-03-2026 |
| Visio | Cluster D | |
"""
    res = convert_markdown_table_to_spreadsheet_json(md)
    assert res["columns"] == ["Topic", "Cluster", "Date"]
    assert len(res["rows"]) == 2
    assert res["rows"][0] == ["Teams", "Cluster E", "20-03-2026"]
    assert res["rows"][1] == ["Visio", "Cluster D", ""]


def test_sanitize_row_wise_table():
    from app.utils.file_utils import sanitize_row_wise_table
    raw_rows = [
        ["Item 1", "50.00"],
        ["Item 2", "1", "100.00"]
    ]
    sanitized = sanitize_row_wise_table(raw_rows, 3)
    assert len(sanitized[0]) == 3
    assert sanitized[0] == ["Item 1", "50.00", ""]
    assert sanitized[1] == ["Item 2", "1", "100.00"]


def test_auto_align_billing_table_columns_flags_mismatched_rows():
    from app.utils.file_utils import auto_align_billing_table_columns
    headers = ["S.NO", "OPENQ", "INQTY", "OUTQTY", "CLS.QTY"]
    rows = [
        ["1", "50", "0", "10", "40"],       # matches header count exactly
        ["2", "30", "5", "35"],             # one cell short - should be flagged
    ]
    aligned, mismatches = auto_align_billing_table_columns(headers, rows)
    assert len(aligned[0]) == 5 and len(aligned[1]) == 5
    assert mismatches == [1], "Row whose cell count didn't match the header must be flagged, not silently padded"

    # Non-billing header: must pass rows through unchanged with no mismatches reported
    aligned2, mismatches2 = auto_align_billing_table_columns(["A", "B"], [["x", "y", "z"]])
    assert aligned2 == [["x", "y", "z"]]
    assert mismatches2 == []


def test_map_invoice_column_roles_handles_reordered_columns():
    from app.utils.file_utils import map_invoice_column_roles
    # Deliberately non-standard order: Qty, Description, Rate, Disc%, Amount
    columns = ["Qty", "Description", "Rate", "Disc%", "Amount"]
    role_map = map_invoice_column_roles(columns)
    assert role_map["quantity"] == 0
    assert role_map["description"] == 1
    assert role_map["unit_price"] == 2
    assert role_map["amount"] == 4


def test_deduplicate_repeated_lines():
    from app.services.model_service import model_engine
    text = "Cluster D\nCluster D\nCluster D\nCluster D\nCluster D\nCluster D"
    deduped = model_engine._deduplicate_repeated_lines(text, max_consecutive_repeats=3)
    assert deduped.count("Cluster D") == 3


def test_normalize_structured_data():
    raw = {"name": "Dr. Smith", "phone": "1234567890"}
    norm = normalize_structured_data("VISITING_CARD", raw)
    assert norm["document_type"] == "VISITING_CARD"
    assert norm["fields"]["name"] == "Dr. Smith"
    assert "company" not in norm["fields"]
    assert norm["fields"]["phone"] == "1234567890"


def test_export_service():
    db = TestingSessionLocal()
    rec = DocumentRecord(
        original_filename="test_doc.pdf",
        stored_filename="stored_test_doc.pdf",
        file_path="/tmp/test_doc.pdf",
        file_type="pdf",
        file_size=1024,
        page_count=1,
        document_type="Invoice",
        language="English",
        has_handwriting=False,
        processing_time=1.23,
        confidence=0.98
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    res = DocumentResult(
        document_id=rec.id,
        raw_text="Sample text content",
        markdown_text="# Sample Markdown\nContent",
        structured_json_str='{"invoice_number": "INV-100"}'
    )
    db.add(res)
    db.commit()

    exporter = ExportService()
    txt_content, media_txt, filename_txt = exporter.generate_export(rec, res, "txt")
    assert "STRUCTURED JSON DATA" in txt_content
    assert "PLAIN TEXT EXTRACTED" in txt_content
    assert media_txt == "text/plain"

    json_content, media_json, filename_json = exporter.generate_export(rec, res, "json")
    assert "INV-100" in json_content
    assert media_json == "application/json"

    db.close()


def test_normalize_no_text_response():
    from app.utils.file_utils import normalize_no_text_response
    assert normalize_no_text_response("") == "No text has been found."
    assert normalize_no_text_response("The image shows a beautiful sunset over the mountains.") == "No text has been found."
    assert normalize_no_text_response("I cannot find any readable text in this picture.") == "No text has been found."
    assert normalize_no_text_response("This is a photo of a blue chair in a room.") == "No text has been found."
    valid_text = "| Product | Price |\n| Item A | 100 |"
    assert normalize_no_text_response(valid_text) == valid_text


def test_sanitize_batch_notes_and_ledger_healing():
    from app.utils.file_utils import sanitize_extracted_markdown
    corrupted_md = """
# ASHOK MEDICAL HALL

## Stock Statement Of HETERO DERMA GLOW For The Month Of June Page No: 1

| SL NO | PRODUCT | PACK | OPENI QTY | RECEV QTY | S A L E QTY | AMOUNT | EXPAI LOSS | CLOSING QTY | CLOSING AMOUNT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MOISTE CREAM | 100GM | batch no.BR1D4020Quantity: 101 | Expiry with in six month: 11/26 | 8 | 0.00 | 0 | 101 | 16725.60 |
| MOISTE CREAM | 100GM | batch no.BR1D4020Quantity: 101 | Expiry with in six month: 11/26 | 8 | 0.00 | 0 | 101 | 16725.60 |  |
| 2 | MOISTE LOTION | 100ML | 45 | 0 | 8 | 1313.28 | 0 | 8 | -696.00 |
| 3 | TOFUS TAB | 10'S | 45 | 0 | 8 | 1313.28 | 0 | 37 | 6073.92 |
"""
    sanitized = sanitize_extracted_markdown(corrupted_md)

    # Verify duplicate unnumbered batch row was removed
    lines = [l for l in sanitized.splitlines() if l.strip().startswith("|")]
    # Header + separator + 3 data rows = 5 rows
    assert len(lines) == 5

    # Row 1: MOISTE CREAM Opening should be 101, Sale Qty 0, Closing 101
    r1 = [c.strip() for c in lines[2].split("|")[1:-1]]
    assert r1[1] == "MOISTE CREAM"
    assert r1[3] == "101"
    assert r1[5] == "0"
    assert r1[8] == "101"

    # Row 2: MOISTE LOTION should be healed from bleed: Opening 8, Sale 0, Closing 8
    r2 = [c.strip() for c in lines[3].split("|")[1:-1]]
    assert r2[1] == "MOISTE LOTION"
    assert r2[3] == "8"
    assert r2[5] == "0"
    assert r2[8] == "8"

    # Row 3: TOFUS TAB unchanged: Opening 45, Sale 8, Closing 37
    r3 = [c.strip() for c in lines[4].split("|")[1:-1]]
    assert r3[1] == "TOFUS TAB"
    assert r3[3] == "45"
    assert r3[5] == "8"
    assert r3[8] == "37"

    # Verify batch notes were NOT outputted as a separate section
    assert "### Batch" not in sanitized


