import json
import pytest
from pathlib import Path

GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"


def test_ground_truth_files_exist_and_valid():
    """Validates that all expected ground truth JSON files exist and have valid structure."""
    expected_files = [
        "1000411295.json",
        "1000411296.json",
        "1000517666.json",
        "1000517796.json",
        "agarwal_jaipur.json",
        "bansal_barelly.json",
        "anshul_sikar.json",
        "saraswati_drug.json",
        "hetroder.json",
        "pan_card.json",
        "aadhaar_card.json"
    ]
    for f in expected_files:
        p = GROUND_TRUTH_DIR / f
        assert p.exists(), f"Ground truth file missing: {f}"
        with open(p, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        # Check required schema fields
        assert "document" in data
        assert "document_type" in data
        assert "table_type" in data
        assert "physical_line_count" in data
        assert "logical_row_count" in data
        assert "product_row_count" in data
        assert "subtotal_row_count" in data
        assert "total_row_count" in data
        assert "columns" in data
        assert isinstance(data["columns"], list)
        assert "rows" in data
        assert isinstance(data["rows"], list)

        # Validate column specification
        for col in data["columns"]:
            assert "name" in col
            assert "type" in col
            assert col["type"] in ["text", "integer", "decimal", "date", "percentage"]
            assert "x_order" in col

        # Validate row specification
        for row in data["rows"]:
            assert "row_type" in row
            assert row["row_type"] in ["product", "subtotal", "total"]
            assert "cells" in row
            assert isinstance(row["cells"], dict)

        # Validate three row concepts consistency:
        # logical_row_count == product_row_count + subtotal_row_count + total_row_count
        expected_logical = (
            data["product_row_count"] +
            data["subtotal_row_count"] +
            data["total_row_count"]
        )
        assert data["logical_row_count"] == expected_logical, (
            f"Row count mismatch in {f}: logical {data['logical_row_count']} != sum {expected_logical}"
        )
