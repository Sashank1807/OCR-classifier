"""
The ERP view of an extraction.

The full API response is built for a reviewer: it carries the markdown
rendering, per-field verification evidence, classification internals and the
raw table geometry, because a human deciding whether to trust an extraction
needs all of that. An ERP does not. It needs the business values, keyed by
name, plus enough signal to decide whether to post the document or route it to
a person.

So this module answers one question - "what would an ERP actually write to a
record?" - and drops everything else. Three concrete reductions:

  * Diagnostics go.  `field_verification`, `classification_confidence`,
    `column_model_uncertain`, token counts and the audit block are evidence
    ABOUT the extraction, not data FROM the document. What survives is
    `confidence`, `review_required` and, when something was flagged, the
    NAMES of the suspect fields - which is the part an ERP can act on (hold
    those, post the rest).

  * Tables become objects.  `fields.rows` is positional (`[["Medicine A",
    "10", "100.5"], ...]`) with the header in a parallel `fields.columns`
    list. Every consumer then has to re-zip them, and any consumer that
    hardcodes an index breaks the first time a column order changes. Here the
    two are joined once, into `line_items` of `{"Product": "Medicine A", ...}`.

  * Empties go.  A schema template contributes every key its document type
    could have; unfilled ones came through as "" or {} (a resume's
    `contact_info` carried three empty social links). An absent key is a
    clearer statement than an empty one.

Values themselves are published VERBATIM - no number parsing, no date
reformatting. A string that looks like "1,234.50" stays that string. This
module is a projection, and a projection that silently rewrites values would
repeat the mistake already recorded for arithmetic auto-repair in this
codebase: the correction corrupted cells the OCR had read correctly. Typing
and mapping belong to the integration, which knows its own ERP's schema.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Keys that describe the extraction rather than the document. Dropped.
_DIAGNOSTIC_KEYS = {
    "classification_confidence",
    "field_verification",
    "column_model_uncertain",
    "overall_confidence",
    "needs_manual_review",
    "document_type",
    "overall_document_quality",
    "processing_summary",
    "token_usage",
    "metadata",
    "_audit",
}

# The key holding the document's PRIMARY table, in order of preference. Only
# one list becomes `line_items`; every other list-valued field keeps its own
# name inside `data`, because a resume's `education` and `work_experience` are
# not line items of each other and collapsing them would lose which was which.
_PRIMARY_TABLE_KEYS = ("items", "line_items", "transactions", "rows")

# Structural companions of the primary table - consumed to build `line_items`,
# never published on their own.
_TABLE_SUPPORT_KEYS = {"columns", "all_tables"}


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _prune(value: Any) -> Any:
    """Drop empty leaves, recursively. Returns None when nothing is left."""
    if isinstance(value, dict):
        cleaned = {k: _prune(v) for k, v in value.items()}
        cleaned = {k: v for k, v in cleaned.items() if not _is_empty(v)}
        return cleaned or None
    if isinstance(value, list):
        items = [_prune(v) for v in value]
        items = [v for v in items if not _is_empty(v)]
        return items or None
    return None if _is_empty(value) else value


def _unique_headers(columns: Sequence[Any], width: int) -> List[str]:
    """
    Column names usable as object keys.

    Two failure cases here are real, not hypothetical: a header cell can be
    blank, and a header name can repeat (a PDF statement in this project had
    six columns all named "Qty"). Either one silently loses columns when a row
    is turned into a dict, so blanks get a positional name and repeats get a
    suffix. Rows wider than the header get named columns too - the extra cell
    is still data.
    """
    headers: List[str] = []
    seen: Dict[str, int] = {}
    for idx in range(max(len(columns), width)):
        raw = columns[idx] if idx < len(columns) else ""
        name = str(raw).strip() if raw is not None else ""
        if not name:
            name = f"column_{idx + 1}"
        count = seen.get(name, 0) + 1
        seen[name] = count
        headers.append(name if count == 1 else f"{name}_{count}")
    return headers


def _objectify(columns: Sequence[Any], rows: Sequence[Any]) -> List[Dict[str, Any]]:
    """Join a positional table with its header into a list of objects."""
    if not rows:
        return []

    # A list of dicts is already object-shaped (invoice line_items, resume
    # education). Pass it through, pruned.
    if all(isinstance(r, dict) for r in rows):
        out = []
        for r in rows:
            cleaned = _prune(r)
            if cleaned:
                out.append(cleaned)
        return out

    widest = max((len(r) for r in rows if isinstance(r, (list, tuple))), default=0)
    headers = _unique_headers(columns or [], widest)

    out = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            # A stray scalar row: keep it rather than dropping data.
            if not _is_empty(row):
                out.append({headers[0] if headers else "value": row})
            continue
        item = {}
        for idx, cell in enumerate(row):
            if _is_empty(cell):
                continue
            item[headers[idx] if idx < len(headers) else f"column_{idx + 1}"] = cell
        if item:
            out.append(item)
    return out


def _split_tables(fields: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[str]]:
    """
    Return (line_items, additional_tables, consumed_keys).

    `all_tables` wins when present: it is the authoritative list for a
    multi-sheet spreadsheet, and its first table is the same data `rows` and
    `columns` hold - publishing both would serialise the whole table twice,
    which is exactly what was stripped out of the API response earlier.
    """
    consumed: Set[str] = set()

    all_tables = fields.get("all_tables")
    if isinstance(all_tables, list) and all_tables:
        usable = [t for t in all_tables if isinstance(t, dict) and t.get("rows")]
        if usable:
            consumed |= _TABLE_SUPPORT_KEYS | {"rows"}
            primary = _objectify(usable[0].get("columns") or [], usable[0].get("rows") or [])
            extra = []
            for pos, tbl in enumerate(usable[1:], start=2):
                items = _objectify(tbl.get("columns") or [], tbl.get("rows") or [])
                if items:
                    extra.append({
                        "table": tbl.get("table_name") or tbl.get("table_index") or pos,
                        "line_items": items,
                    })
            return primary, extra, consumed

    for key in _PRIMARY_TABLE_KEYS:
        value = fields.get(key)
        if isinstance(value, list) and value:
            consumed |= {key} | _TABLE_SUPPORT_KEYS
            return _objectify(fields.get("columns") or [], value), [], consumed

    # No table at all. `columns` without rows describes nothing.
    consumed |= _TABLE_SUPPORT_KEYS
    return [], [], consumed


def build_erp_payload(
    structured_data: Optional[Dict[str, Any]],
    *,
    document_id: Optional[int] = None,
    request_id: Optional[str] = None,
    filename: Optional[str] = None,
    document_type: Optional[str] = None,
    project_name: Optional[str] = None,
    processed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Project a structured extraction onto the fields an ERP consumes.

    The shape is stable across document types - an integration reads
    `payload["data"]` and `payload["line_items"]` without branching on
    `document_type` first - while the KEYS inside `data` stay the ones the
    document actually yielded.
    """
    envelope = structured_data if isinstance(structured_data, dict) else {}
    fields = envelope.get("fields")
    if not isinstance(fields, dict):
        # Tolerate a bare field map, which is what an older stored row or a
        # direct pipeline call can hand over.
        fields = {k: v for k, v in envelope.items() if k not in _DIAGNOSTIC_KEYS}

    line_items, additional_tables, consumed = _split_tables(fields)

    data: Dict[str, Any] = {}
    for key, value in fields.items():
        if key in _DIAGNOSTIC_KEYS or key in consumed:
            continue
        cleaned = _prune(value)
        if cleaned is not None:
            data[key] = cleaned

    payload: Dict[str, Any] = {
        "document_id": document_id,
        "request_id": request_id,
        "source_file": filename,
        "document_type": document_type or envelope.get("document_type") or "UNKNOWN",
        "project_name": project_name,
        "extracted_at": processed_at,
        "confidence": envelope.get("overall_confidence"),
        "review_required": bool(envelope.get("needs_manual_review", False)),
        "data": data,
        "line_items": line_items,
    }

    # Which fields to hold, not the evidence for holding them. An ERP can post
    # the rest of the document and queue these for a person.
    verification = envelope.get("field_verification")
    if isinstance(verification, dict):
        flagged = verification.get("flagged") or []
        names = [f.get("field") for f in flagged if isinstance(f, dict) and f.get("field")]
        if names:
            payload["review_fields"] = names

        # WHY the document needs review, when the reason is something the
        # sender can act on. "review_required: true" on its own tells an
        # operator nothing; "the image is 180x261, text below ~400px is not
        # reliably legible" tells them to rescan. That was a real case - a
        # pamphlet thumbnail scored 0.1 and nobody could see the cause
        # without opening the verification internals this payload drops.
        legibility = verification.get("source_legibility")
        if isinstance(legibility, dict) and not legibility.get("legible", True):
            reasons = [r for r in (legibility.get("reasons") or []) if r]
            if reasons:
                payload["review_reason"] = "; ".join(reasons)

    if additional_tables:
        payload["additional_tables"] = additional_tables

    # An identifier the caller did not supply is noise, not information.
    # `data` and `line_items` are kept even when empty, so the shape a
    # consumer indexes into is always there.
    always = {"data", "line_items", "review_required", "document_type"}
    return {k: v for k, v in payload.items() if v is not None or k in always}


def line_item_columns(payload: Optional[Dict[str, Any]]) -> List[str]:
    """
    Column order for rendering `line_items` as a table.

    Items are dicts with empty cells dropped, so no single item is guaranteed
    to carry every column. Taking the keys of the first item would silently
    hide a column that only appears further down, so this walks all of them
    and keeps first-seen order.
    """
    if not isinstance(payload, dict):
        return []
    columns: List[str] = []
    for item in payload.get("line_items") or []:
        if not isinstance(item, dict):
            continue
        for key in item:
            if key not in columns:
                columns.append(key)
    return columns


def erp_from_record(record, result) -> Dict[str, Any]:
    """Build the ERP payload from the stored DocumentRecord / DocumentResult."""
    stamp = record.completed_at or record.created_at
    return build_erp_payload(
        result.structured_json if result else {},
        document_id=record.id,
        request_id=record.request_id,
        filename=record.original_filename,
        document_type=record.document_type,
        project_name=record.project_name or record.department,
        processed_at=stamp.isoformat() if stamp else None,
    )


def erp_from_pipeline_result(
    result_data: Dict[str, Any],
    project_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the ERP payload from a synchronous pipeline return value."""
    return build_erp_payload(
        result_data.get("structured_data") or {},
        document_id=result_data.get("id"),
        request_id=result_data.get("request_id"),
        filename=result_data.get("filename"),
        document_type=result_data.get("document_type"),
        project_name=project_name,
        processed_at=result_data.get("created_at"),
    )
