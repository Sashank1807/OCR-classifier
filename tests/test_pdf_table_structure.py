"""
Digital-PDF table structure.

These documents are extracted natively (exact text, no OCR), so the values were
always right - what was wrong was the SHAPE of the JSON built around them:

  * a stacked header was read only at its leaf tier, so six different columns
    all came back named "Qty". Role mapping could not tell opening from sale
    and emitted null for every quantity on every line.
  * the summary block, print stamp and software credit line below the grid
    became table rows, because the table was terminated by matching one
    vendor's wording ("tot sale", "generated at").
  * figures are right-aligned under labels of a different width, so a boundary
    taken from the header midpoint sliced through a column: two values landed
    in one cell ("0 0") and its neighbour came back empty.

Every rule tested here keys on the SHAPE of a line, never on a vendor's or a
product's wording, so an ERP nobody has seen still reads correctly.
"""

import re
# Imported by name: `app.services.__init__` binds `pdf_service` to the
# singleton, which shadows the submodule of the same name.
from app.services.pdf_service import (
    PDFService,
    _is_summary_line,
    _looks_like_footer,
    _looks_like_report_period,
)
from app.services.ocr_pipeline import pipeline


def test_summary_block_is_recognised_by_shape():
    assert _is_summary_line("OP.Stk Val (PTS) : 138995.86 Pur.Val(PTS+Tax) : 39761.77")
    assert _is_summary_line("CL.Stk Val(PTS): 157306.05 LMS(PTR+Tax) : 37981.01")
    # A product row has one label and then bare figures - never colon pairs.
    assert not _is_summary_line("3 BILASET 20-10's 65 107 0 0 40 12 2,343.52")
    # A batch/expiry continuation has ONE pair and is real table data.
    assert not _is_summary_line("BATCH: 25S2GCA517")


def test_footer_chrome_is_recognised_without_vendor_names():
    assert _looks_like_footer("Printed By : REKHA - COUNTER1, 01-Jun-26 9:07:57 AM")
    assert _looks_like_footer("Software @BlueFox Systems, Kannur www.bluefoxsystems.net")
    assert _looks_like_footer("Ph:-0497-2761096, 9447684374")
    assert not _looks_like_footer("18 OPOX 100 DT-10's 38 14 66 0 30 6 3,041.45")


def test_report_title_is_not_mistaken_for_a_super_header_tier():
    # It sits directly above the header, carries no bare figures and spans the
    # grid, so on shape alone it looks exactly like a super tier - absorbed as
    # one it prepends "Stock And Sales from" to the first column's name.
    assert _looks_like_report_period("Stock And Sales from 01-May-2026 to 31-May-2026")
    assert _looks_like_report_period("Summary from 01/05/2026 To 29/05/2026")
    # The genuine super tier must NOT be rejected.
    assert not _looks_like_report_period("Op. Pur Pur Sale Sale Repl Ret. Adj Sale Bal.")


def test_column_boundary_is_recut_between_the_two_lanes_of_figures():
    # Header labels put the boundary at 100, but the figures of the two columns
    # actually occupy 60-80 and 120-140 - so a value at 120 fell on the wrong
    # side, joining its neighbour as "0 0" and leaving one cell empty.
    col_bounds = [("Sale Qty", 0.0, 100.0), ("Sale F.Qty", 100.0, 200.0)]
    lines_dict = {float(y): [] for y in range(10, 60, 10)}
    sorted_y = sorted(lines_dict)
    for y in sorted_y:
        lines_dict[y] = [
            (60.0, y, 80.0, y + 8, "0"),
            (120.0, y, 140.0, y + 8, "0"),
        ]
    out = PDFService._refine_bounds_from_data(col_bounds, lines_dict, sorted_y, hdr_y=5.0)
    assert 80.0 <= out[0][2] <= 120.0, out
    assert out[0][2] == out[1][1]

    # With only one lane of figures there is no evidence of a split, so the
    # header's boundary is left exactly where it was.
    for y in sorted_y:
        lines_dict[y] = [(60.0, y, 80.0, y + 8, "0")]
    same = PDFService._refine_bounds_from_data(col_bounds, lines_dict, sorted_y, hdr_y=5.0)
    assert same == col_bounds


def test_summary_totals_are_parsed_and_the_print_stamp_is_not():
    md = "\n".join([
        "| SlNo | Product Name | Bal.Val |",
        "| --- | --- | --- |",
        "| 1 | AD 10 SACHET-1gm | 92.60 |",
        "",
        "OP.Stk Val (PTS) : 138995.86 Pur.Val(PTS+Tax) : 39761.77",
        "CL.Stk Val(PTS): 157306.05 Sal.Free.Val(MRP) : 11286.95",
        "Printed By : REKHA - COUNTER1, 01-Jun-26 9:07:57 AM Page 1 of 1",
    ])
    totals = pipeline._parse_summary_totals(md)
    assert totals["OP.Stk Val (PTS)"] == 138995.86
    assert totals["CL.Stk Val(PTS)"] == 157306.05
    assert totals["Sal.Free.Val(MRP)"] == 11286.95
    # The clock time in the print stamp parses as {"Jun-26 9": 7.0} if allowed.
    assert not any("Jun" in k for k in totals), totals
    assert 7.0 not in totals.values(), totals
    # Table rows are not summary lines.
    assert not any("SACHET" in k for k in totals)


def test_company_name_comes_from_the_division_banner_not_the_address():
    # "the line after the distributor" is a guess about layout order, and on
    # any letterhead carrying a street address it returns the address.
    md = "\n".join([
        "# ARADHANA AGENCIES",
        "51/2434, GRAND ICON",
        "NETHAJI ROAD, KANNUR",
        "Stock And Sales from 01-May-2026 to 31-May-2026",
        "",
        "| SlNo | Product Name | Bal.Val |",
        "| --- | --- | --- |",
        "|  | HETERO HEALTHCARE (GENX) |  |",
        "| 1 | AD 10 SACHET-1gm | 92.60 |",
    ])
    dist, comp, per = pipeline._extract_stock_header_metadata(md)
    assert dist == "ARADHANA AGENCIES"
    assert comp == "HETERO HEALTHCARE (GENX)", comp
    assert "01-May-2026" in (per or "")


def test_api_items_do_not_republish_the_cells_already_in_rows():
    """
    `raw_cells` used to be published on every item in the API response, as a
    byte-for-byte copy of the matching entry in `rows` - so every cell of the
    table crossed the wire twice. On a 16-column stock statement that was 23%
    of the payload, and the cost grows with table width.

    It stays an internal field on the validated row (validation_service builds
    it, and product_name falls back to it); this pins that it is not serialised
    out again.
    """
    import inspect
    from app.services.ocr_pipeline import OCRPipeline

    src = inspect.getsource(OCRPipeline.process_file)
    published = re.search(
        r'stock_items\.append\(\{(.*?)\n\s*\}\)', src, re.S
    )
    assert published, "stock_items payload block not found - has it been renamed?"
    # Match it as a published KEY only. The block legitimately mentions
    # raw_cells inside the product_name fallback, which reads the internal
    # field and does not re-emit it.
    assert not re.search(r'^\s*"raw_cells"\s*:', published.group(1), re.M), (
        "raw_cells is being published again; consumers should read `rows`, "
        "which is positionally aligned with `columns`."
    )
    # The fallback that legitimately reads it must still be there.
    assert "raw_cells" in published.group(1), (
        "product_name's raw_cells fallback disappeared - unrelated regression"
    )


def test_api_tables_do_not_carry_internal_cell_geometry():
    """
    `tables[].cells_metadata` holds, for every cell, the source OCR tokens with
    pixel bboxes, quad_slope, per-token confidence and glyph dimensions. It
    drives the closed-loop cell re-OCR and is meaningless to an API consumer -
    but it was 93% of the table payload (30,309 bytes of 32,370 on a 20-row
    statement) and grows with every cell on the page.

    It must keep flowing internally and must not be serialised out.
    """
    import inspect
    from app.services.ocr_pipeline import OCRPipeline

    src = inspect.getsource(OCRPipeline.process_file)

    # Stripped once, at the serialization boundary...
    assert 'k != "cells_metadata"' in src, (
        "the cells_metadata strip disappeared from process_file"
    )
    # ...and both the stored blob and the response use the stripped list.
    assert re.search(r'tables_str\s*=\s*json\.dumps\(\s*published_tables', src), (
        "stored tables_json_str is not using the stripped list"
    )
    assert re.search(r'"tables":\s*published_tables', src), (
        "the API response is not using the stripped list"
    )
    # ...while the internal repair pass still reads the full metadata.
    repair = inspect.getsource(OCRPipeline._attempt_closed_loop_reocr)
    assert 'cells_metadata' in repair, (
        "closed-loop re-OCR lost access to cells_metadata"
    )


def test_api_response_does_not_republish_the_table_as_a_separate_block():
    """
    The top-level `tables` block repeated a table the consumer already has:
    its columns/rows are the same lists as structured_data.fields.columns/.rows
    (and fields.all_tables for spreadsheets), and its markdown is a slice of the
    document markdown. One 20-row statement carried the same table four times.

    It is dropped at the API boundary, NOT in the pipeline - process_file's
    return is also the internal interface the benchmark reads, and that still
    needs the tables list.
    """
    import inspect
    from app.api import routes_ocr

    strip = inspect.getsource(routes_ocr._api_document)
    assert 'k != "tables"' in strip, "the tables strip disappeared from _api_document"

    src = inspect.getsource(routes_ocr)
    assert "_api_document(result_data)" in src, (
        "the sync /process response stopped going through _api_document"
    )
    assert "_api_document(new_result)" in src, (
        "the /reprocess response stopped going through _api_document"
    )

    # The ERP format is the other way out of /process, so it needs the same
    # guarantee - asserted on the projection itself rather than on the source,
    # since `tables` is not a field key and so cannot survive it.
    from app.services.erp_payload import erp_from_pipeline_result

    projected = erp_from_pipeline_result({
        "id": 1,
        "filename": "x.pdf",
        "document_type": "STOCK_STATEMENT",
        "structured_data": {"fields": {"columns": ["Item"], "rows": [["Tablet"]]}},
        "tables": [{"columns": ["Item"], "rows": [["Tablet"]], "markdown": "| Item |"}],
    })
    assert "tables" not in projected
    assert projected["line_items"] == [{"Item": "Tablet"}]

    # The pipeline must still hand `tables` to internal callers.
    from app.services.ocr_pipeline import OCRPipeline
    assert '"tables": published_tables' in inspect.getsource(OCRPipeline.process_file), (
        "process_file stopped returning tables - the benchmark reads it"
    )


def test_column_model_uncertain_survives_the_tables_removal():
    """
    `column_model_uncertain` lived only on the `tables` block, and it is a real
    review signal: a candidate column was detected but rejected for want of
    corroboration, so the table may be MISSING a column. Dropping `tables`
    without rehoming it would silently lose that warning.
    """
    import inspect
    from app.services.ocr_pipeline import OCRPipeline

    envelope = re.search(
        r'client_envelope\s*=\s*\{(.*?)\n\s*\}', inspect.getsource(OCRPipeline.process_file), re.S
    )
    assert envelope, "client_envelope block not found"
    assert '"column_model_uncertain"' in envelope.group(1), (
        "column_model_uncertain is no longer published in structured_data"
    )


def test_result_endpoints_publish_the_extracted_table():
    """
    /status and /document returned markdown only, so a caller polling an async
    job got prose back and had to either re-parse the table out of it or make a
    second call to /export - even though the structured result was already in
    the very row being read.
    """
    import inspect
    from app.api import routes_ocr

    status_src = inspect.getsource(routes_ocr.get_job_status)
    assert '"structured_data": result.structured_json if result else {}' in status_src, (
        "/status stopped publishing structured_data"
    )

    doc_src = inspect.getsource(routes_ocr.get_document_details)
    assert '"structured_data": result.structured_json if result else {}' in doc_src, (
        "/document stopped publishing structured_data"
    )


def test_status_accepts_a_numeric_document_id_as_well_as_a_request_id():
    """
    Callers reach for the document id on /status, because that is the id the
    rest of the API is addressed by. Every issued request_id is "req_<hex>", so
    an all-digit value that misses the request_id lookup is unambiguously a
    document id - a 404 there looks like the job was never submitted.
    """
    import inspect
    from app.api import routes_ocr

    src = inspect.getsource(routes_ocr.get_job_status)
    assert "request_id.isdigit()" in src, "the numeric document-id fallback is gone"
    assert "DocumentRecord.id == int(request_id)" in src, (
        "the fallback no longer looks the document up by id"
    )
    # The request_id path must still be tried first.
    assert src.index("DocumentRecord.request_id == request_id") < src.index("request_id.isdigit()")
