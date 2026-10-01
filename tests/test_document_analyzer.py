import pytest
from pathlib import Path
from app.services.document_analyzer import document_analyzer


def test_document_analyzer_office_docs():
    # Excel
    res = document_analyzer.analyze(Path("dummy_file.xlsx"))
    assert res["source_type"] == "OFFICE_DOC"
    assert res["document_type"] == "SPREADSHEET"
    assert res["is_table_heavy"] is True
    assert res["recommended_pipeline"] == "OFFICE_DOC_PIPELINE"

    # CSV
    res_csv = document_analyzer.analyze(Path("dummy_file.csv"))
    assert res_csv["source_type"] == "OFFICE_DOC"
    assert res_csv["document_type"] == "SPREADSHEET"

    # Docx
    res_docx = document_analyzer.analyze(Path("candidate_cv.docx"))
    assert res_docx["source_type"] == "OFFICE_DOC"
    assert res_docx["document_type"] == "RESUME"

    # Text
    res_txt = document_analyzer.analyze(Path("log_output.txt"))
    assert res_txt["source_type"] == "OFFICE_DOC"
    assert res_txt["document_type"] == "SPREADSHEET"


def test_document_analyzer_id_filenames():
    # PAN
    res_pan = document_analyzer.analyze(Path("my_pan_card.jpg"))
    assert res_pan["document_type"] == "PAN"
    assert res_pan["recommended_pipeline"] == "ID_CARD_PIPELINE"

    # Aadhaar
    res_aadhaar = document_analyzer.analyze(Path("aadhaar_front.png"))
    assert res_aadhaar["document_type"] == "AADHAAR"
    assert res_aadhaar["recommended_pipeline"] == "ID_CARD_PIPELINE"

    # Passport
    res_pass = document_analyzer.analyze(Path("passport_scan.jpg"))
    assert res_pass["document_type"] == "PASSPORT"
    assert res_pass["recommended_pipeline"] == "ID_CARD_PIPELINE"


def test_document_analyzer_stock_filenames():
    res_stock = document_analyzer.analyze(Path("lifecare_june_stock_statement.jpg"))
    assert res_stock["document_type"] == "STOCK_STATEMENT"
    assert res_stock["recommended_pipeline"] == "COORDINATE_TABLE_PIPELINE"


def test_document_analyzer_explicit_request():
    res = document_analyzer.analyze(Path("generic_document.png"), requested_type="INVOICE")
    assert res["document_type"] == "INVOICE"
    assert res["classification_confidence"] == 1.0
