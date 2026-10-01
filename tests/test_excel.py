import pytest
import pandas as pd
import pathlib
from app.services.excel_service import excel_service
from app.services.ocr_pipeline import pipeline
from app.core.database import SessionLocal


def test_excel_service_extraction(tmp_path):
    # Create sample Excel spreadsheet
    excel_path = tmp_path / "sample_stock.xlsx"
    df1 = pd.DataFrame({
        "ITEM": ["HETRAN 20MG TAB", "HETRIVA 9MG", "TOTAL VALUE"],
        "PACK": ["10S", "1*S", ""],
        "OPENING": [178, 1, 20393.6],
        "PURCHASE": [120, 370, 23781.0],
        "SALE": [0, 180, 7122.15],
        "CLOSING": [298, 191, 43604.6]
    })
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="Stock_And_Sales", index=False)

    md, plain, pages = excel_service.extract_excel_content(excel_path)
    assert "Stock_And_Sales" in md
    assert "HETRAN 20MG TAB" in md
    assert "20393.6" in md
    assert len(pages) == 1


def test_pipeline_excel_processing(tmp_path):
    excel_path = tmp_path / "test_pipeline.xlsx"
    df = pd.DataFrame({
        "Product": ["Medicine A", "Medicine B"],
        "Qty": [10, 20],
        "Price": [100.50, 200.75]
    })
    df.to_excel(excel_path, index=False, engine="openpyxl")

    db = SessionLocal()
    try:
        res = pipeline.process_file(db, excel_path, "test_pipeline.xlsx", requested_doc_type="SPREADSHEET")
        assert res["document_type"] == "SPREADSHEET"
        assert "Medicine A" in res["markdown"]
        assert "Medicine B" in res["markdown"]
        assert res["status"] == "COMPLETED"
    finally:
        db.close()
