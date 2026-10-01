"""
Canonical cell-state constants shared by the table/cell OCR pipeline and the
structured-output schema. Both `cell_ocr_service.py` and `structured_schemas.py`
previously used their own divergent sets of string literals for the same concept
(cell state) — this module is the single source of truth going forward.

Cell state is distinct from CellType (column data type, e.g. INTEGER_QTY): state
answers "what do we actually know about this cell's content", type answers
"what kind of value belongs here".

    VALUE   - a real, OCR/re-OCR-confirmed value is present.
    ZERO    - the value is the number zero (kept distinct from VALUE so zero
              vs blank vs dash confusion in stock ledgers is traceable).
    DASH    - a printed dash/NA marker is present (business meaning: "not
              applicable"), distinct from EMPTY.
    EMPTY   - visually confirmed: no ink/content in the cell.
    UNKNOWN - ink/content may be present but could not be reliably read, OR
              no cell-specific OCR evidence exists to support a value (e.g. a
              blank packing cell that has no visual evidence for any specific
              packing string). UNKNOWN must never be silently collapsed into
              EMPTY or backfilled with a guessed value - see cell_ocr_service.py's
              extract_cell_value/refine_table_cells cascade and the project's
              evidence-integrity rule: prefer UNKNOWN over a wrong or
              fabricated value, and prefer UNKNOWN over EMPTY when ink might
              be present but unreadable.

`NOT_REPORTED` is kept as an alias of DASH for backward compatibility with
existing structured-output consumers that already expect that string.
"""

from typing import Any, Dict, Optional


VALUE = "VALUE"
ZERO = "ZERO"
DASH = "DASH"
EMPTY = "EMPTY"
UNKNOWN = "UNKNOWN"

# Back-compat alias: structured_schemas.create_semantic_cell historically emitted
# "NOT_REPORTED" for dash/NA cells. New code should use DASH; this alias exists
# so existing consumers of the structured JSON output are not broken.
NOT_REPORTED = DASH

ALL_STATES = (VALUE, ZERO, DASH, EMPTY, UNKNOWN)


def make_cell_meta(
    raw: str,
    state: str,
    normalized: Optional[float] = None,
    repaired: bool = False,
    repair_source: Optional[str] = None,
    repair_reason: Optional[str] = None,
    engine: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds a cell metadata dict with a canonical, unambiguous state.

    repair_source distinguishes *why* a value differs from the first-pass OCR
    read, e.g.:
      - "split_from_description_same_row": value relocated from this row's own
        description cell (evidence-preserving, not fabrication).
      - "arithmetic_conflict_reocr_confirmed": arithmetic flagged a conflict,
        a targeted re-OCR was run, and its result is what's stored here.
      - "narrow_column_reocr": a focused re-crop/re-OCR of the cell itself.
    A value must never be attributed a repair_source of "arithmetic" alone
    (i.e. computed and written with no independent OCR/visual confirmation) -
    that case must resolve to UNKNOWN instead, per the evidence-integrity rule.
    """
    meta: Dict[str, Any] = {
        "raw": raw,
        "normalized": normalized,
        "status": state,
        "repaired": repaired,
    }
    if repair_source:
        meta["repair_source"] = repair_source
    if repair_reason:
        meta["repair_reason"] = repair_reason
    if engine:
        meta["engine"] = engine
    return meta
