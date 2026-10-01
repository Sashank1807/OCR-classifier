"""
Content-integrity checks for the ground-truth fixtures.

test_immutable_ground_truth.py validates fixture STRUCTURE (required keys,
row-count bookkeeping). These checks validate fixture CONTENT: a fixture that
is internally self-contradictory silently corrupts every accuracy measurement
taken against it, and tuning the pipeline against it chases noise.

Both defects below were found by hand after benchmark scores stopped making
sense, so they are encoded here to be caught automatically instead.

KNOWN_BAD records fixtures whose defects are already identified but not yet
corrected - correcting a fixture changes the measurement standard for all
historical numbers, so it is a deliberate decision, not a drive-by edit. When
a fixture is fixed, delete its entry; if a NEW fixture breaks, these fail.
"""

import json
from pathlib import Path

import pytest

GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"

# fixture -> why it currently fails. Empty is the goal state.
# CORRECTED 2026-09-10: 1000411296.json total row (In 120.000 -> 90.000,
# Balance 222.000 -> 192.000) to match both its own product rows and the image.
KNOWN_BAD_ARITHMETIC = {}

KNOWN_BAD_PLACEHOLDER = {
    "bansal_barelly.json": (
        "Contains fabricated product rows 'EXTRA PRODUCT 1'/'EXTRA PRODUCT 2' "
        "and files the 'HETERO-DERMA GLOW' banner line as a product, while "
        "omitting real rows present in the image (HETERONOX CREAM 5GM, "
        "MOISTE LOTION 100ML, NAFBOR CREAM 30GM)."
    ),
}

PLACEHOLDER_MARKERS = ("extra product", "placeholder", "dummy", "sample product", "test product")

# CORRECTED 2026-09-10: pan_card.json and aadhaar_card.json field values
# transcribed from the source images in uploads/ (they previously held the field
# LABEL, e.g. 'PAN'/'DOB', so no extraction could ever match and both documents
# scored a meaningless flat 50%).
KNOWN_BAD_LABEL_AS_VALUE = {}

# Columns that should foot: total row == sum of product rows.
FOOTING_COLUMNS = ("Opening", "In", "Out", "Balance")


def _fixtures():
    return sorted(GROUND_TRUTH_DIR.glob("*.json"))


def _num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _arithmetic_discrepancies(data):
    """Returns {column: (sum_of_product_rows, declared_total)} for columns that don't foot."""
    rows = data.get("rows") or []
    total = next((r.get("cells", {}) for r in rows if r.get("row_type") == "total"), None)
    if not total:
        return {}

    cols = [c["name"] for c in data.get("columns", []) if c["name"] in FOOTING_COLUMNS]
    out = {}
    for col in cols:
        declared = _num(total.get(col))
        if declared is None:
            continue
        parts = [_num(r.get("cells", {}).get(col)) for r in rows if r.get("row_type") == "product"]
        parts = [p for p in parts if p is not None]
        if not parts:
            continue
        got = round(sum(parts), 3)
        if abs(got - declared) > 0.05:
            out[col] = (got, declared)
    return out


@pytest.mark.parametrize("path", _fixtures(), ids=lambda p: p.name)
def test_total_row_foots_against_product_rows(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    bad = _arithmetic_discrepancies(data)

    if path.name in KNOWN_BAD_ARITHMETIC:
        assert bad, (
            f"{path.name} is listed in KNOWN_BAD_ARITHMETIC but now foots correctly - "
            f"remove its entry. Reason recorded: {KNOWN_BAD_ARITHMETIC[path.name]}"
        )
        return

    assert not bad, (
        f"{path.name}: total row contradicts the sum of its own product rows "
        f"(column: sum vs declared) -> {bad}. Either the fixture's product rows or its "
        f"total row is wrong; measurements taken against it are unreliable until fixed."
    )


@pytest.mark.parametrize("path", _fixtures(), ids=lambda p: p.name)
def test_no_placeholder_product_rows(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    found = []
    for i, row in enumerate(data.get("rows") or []):
        cells = row.get("cells") or {}
        first = str(next(iter(cells.values()), "")).strip().lower()
        if any(m in first for m in PLACEHOLDER_MARKERS):
            found.append((i, first))

    if path.name in KNOWN_BAD_PLACEHOLDER:
        assert found, (
            f"{path.name} is listed in KNOWN_BAD_PLACEHOLDER but has no placeholder rows now - "
            f"remove its entry. Reason recorded: {KNOWN_BAD_PLACEHOLDER[path.name]}"
        )
        return

    assert not found, (
        f"{path.name}: fabricated placeholder rows present {found}. The pipeline is scored "
        f"as 'dropping rows' for not inventing documents' content that isn't on the page."
    )


def _label_as_value_rows(data):
    """Field-value cells that just repeat the field's own name/label instead of the card's text."""
    cols = [c["name"] for c in data.get("columns", [])]
    if "field_name" not in cols or "field_value" not in cols:
        return []
    bad = []
    for i, row in enumerate(data.get("rows") or []):
        cells = row.get("cells") or {}
        name = str(cells.get("field_name", "")).strip().lower().replace("_", " ")
        value = str(cells.get("field_value", "")).strip().lower().replace("_", " ")
        if not value:
            continue
        # "pan_number" -> value "PAN", "address" -> value "ADDRESS": the value is
        # the label (or a word of it), never something read off the document.
        if value == name or value in name.split() or name.startswith(value):
            bad.append((i, cells.get("field_name"), cells.get("field_value")))
    return bad


@pytest.mark.parametrize("path", _fixtures(), ids=lambda p: p.name)
def test_field_values_are_not_just_field_labels(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    bad = _label_as_value_rows(data)

    if path.name in KNOWN_BAD_LABEL_AS_VALUE:
        assert bad, (
            f"{path.name} is listed in KNOWN_BAD_LABEL_AS_VALUE but its field values look "
            f"real now - remove its entry. Reason recorded: {KNOWN_BAD_LABEL_AS_VALUE[path.name]}"
        )
        return

    assert not bad, (
        f"{path.name}: field_value repeats the field label instead of the document's own text "
        f"{bad}. Nothing extracted from the document can ever match, so the score reflects the "
        f"fixture rather than the pipeline."
    )
