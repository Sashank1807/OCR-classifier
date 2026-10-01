"""
The ERP projection.

What this guards is a contract an integration is written against, so the
tests are about SHAPE as much as content: an ERP that reads
`payload["line_items"][0]["Qty"]` must keep working, and a diagnostic field
must never reappear in a payload someone has already mapped.

Fixtures below are trimmed copies of real stored envelopes from this
deployment, not invented ones - the resume's three empty social links, the
visiting card's six flagged fields and the spreadsheet's duplicated
`all_tables` are all things the pipeline actually produced.
"""

import pytest

from app.services.erp_payload import (
    build_erp_payload,
    line_item_columns,
)

# The full envelope as stored, diagnostics and all.
PAN_ENVELOPE = {
    "document_type": "PAN",
    "classification_confidence": 0.9,
    "overall_confidence": 1.0,
    "needs_manual_review": False,
    "field_verification": {
        "counts": {"exact": 0, "verified": 4, "unchecked": 0, "flagged": 0},
        "field_count": 4,
        "source_legibility": {"legible": True, "reasons": [], "min_dimension": 862},
        "flagged": [],
    },
    "column_model_uncertain": False,
    "fields": {
        "name": "Sanjay Kumar Prajapati",
        "father_name": "Dukhiya Prashad Prajapati",
        "dob": "02/06/1990",
        "pan_number": "BJQPP5524G",
    },
}

SPREADSHEET_ENVELOPE = {
    "document_type": "SPREADSHEET",
    "overall_confidence": 1.0,
    "needs_manual_review": False,
    "field_verification": {"counts": {}, "flagged": []},
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
}


# --------------------------------------------------------------------------- #
# What must be dropped
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("diagnostic", [
    "field_verification",
    "classification_confidence",
    "column_model_uncertain",
    "fields",
])
def test_diagnostics_never_reach_the_erp(diagnostic):
    payload = build_erp_payload(PAN_ENVELOPE)
    assert diagnostic not in payload
    assert diagnostic not in payload["data"]


def test_the_business_values_all_survive():
    payload = build_erp_payload(PAN_ENVELOPE)
    assert payload["data"] == PAN_ENVELOPE["fields"]
    assert payload["document_type"] == "PAN"
    assert payload["confidence"] == 1.0
    assert payload["review_required"] is False


def test_empty_values_are_omitted_not_published_as_blanks():
    """
    A schema template contributes every key its type could have. The resume's
    contact_info arrived with three empty social links; an absent key states
    "not on the document" more clearly than "".
    """
    envelope = {
        "overall_confidence": 1.0,
        "fields": {
            "candidate_name": "Subhajit Pati",
            "contact_info": {
                "email": "subhajitpati79@gmail.com",
                "linkedin": "",
                "github": "",
                "portfolio": "",
            },
            "certifications": [],
            "professional_summary": "   ",
        },
    }
    data = build_erp_payload(envelope)["data"]
    assert data["contact_info"] == {"email": "subhajitpati79@gmail.com"}
    assert "certifications" not in data
    assert "professional_summary" not in data


def test_the_shape_is_present_even_when_nothing_was_extracted():
    """
    A consumer indexes into `data` and `line_items` unconditionally. On a
    document that yielded nothing they must be empty, not missing.
    """
    payload = build_erp_payload({"overall_confidence": 0.1, "needs_manual_review": True,
                                 "fields": {"summary_totals": {}}})
    assert payload["data"] == {}
    assert payload["line_items"] == []
    assert payload["review_required"] is True


# --------------------------------------------------------------------------- #
# Tables: positional rows become objects
# --------------------------------------------------------------------------- #

def test_rows_are_joined_to_their_column_names():
    payload = build_erp_payload(SPREADSHEET_ENVELOPE)
    assert payload["line_items"] == [
        {"Product": "Medicine A", "Qty": "10", "Price": "100.5"},
        {"Product": "Medicine B", "Qty": "20", "Price": "200.75"},
    ]
    # ...and the raw parallel lists are not ALSO published, which would send
    # the same table twice.
    assert "rows" not in payload["data"]
    assert "columns" not in payload["data"]
    assert payload["data"] == {"sheet_title": "Sheet1"}


def test_duplicate_column_names_do_not_collapse_columns():
    """
    A PDF statement in this project produced six columns all named "Qty".
    Keyed naively, five of the six values would vanish.
    """
    envelope = {"fields": {
        "columns": ["Item", "Qty", "Qty", "Qty"],
        "rows": [["Tablet", "1", "2", "3"]],
    }}
    item = build_erp_payload(envelope)["line_items"][0]
    assert item == {"Item": "Tablet", "Qty": "1", "Qty_2": "2", "Qty_3": "3"}


def test_a_blank_header_cell_still_yields_a_usable_key():
    envelope = {"fields": {"columns": ["Item", "", "Amount"],
                           "rows": [["Tablet", "5", "50.00"]]}}
    item = build_erp_payload(envelope)["line_items"][0]
    assert item == {"Item": "Tablet", "column_2": "5", "Amount": "50.00"}


def test_a_row_wider_than_the_header_keeps_its_extra_cell():
    envelope = {"fields": {"columns": ["Item"], "rows": [["Tablet", "5"]]}}
    item = build_erp_payload(envelope)["line_items"][0]
    assert item["Item"] == "Tablet"
    assert "5" in item.values()


def test_already_object_shaped_line_items_pass_through():
    envelope = {"fields": {
        "invoice_number": "INV-1",
        "line_items": [{"description": "Tablet", "quantity": "2", "amount": ""}],
    }}
    payload = build_erp_payload(envelope)
    assert payload["line_items"] == [{"description": "Tablet", "quantity": "2"}]
    assert payload["data"] == {"invoice_number": "INV-1"}


def test_extra_sheets_are_published_separately_not_merged():
    envelope = {"fields": {"all_tables": [
        {"table_index": 1, "columns": ["A"], "rows": [["1"]]},
        {"table_index": 2, "columns": ["B"], "rows": [["2"]]},
    ]}}
    payload = build_erp_payload(envelope)
    assert payload["line_items"] == [{"A": "1"}]
    assert payload["additional_tables"] == [{"table": 2, "line_items": [{"B": "2"}]}]


def test_only_the_primary_table_becomes_line_items():
    """
    A resume's education and work_experience are both lists of objects, but
    they are not line items of each other. Collapsing them would lose which
    entry came from which section.
    """
    envelope = {"fields": {
        "candidate_name": "A",
        "education": [{"degree": "B.Pharm"}],
        "work_experience": [{"role": "Intern"}],
    }}
    payload = build_erp_payload(envelope)
    assert payload["line_items"] == []
    assert payload["data"]["education"] == [{"degree": "B.Pharm"}]
    assert payload["data"]["work_experience"] == [{"role": "Intern"}]


# --------------------------------------------------------------------------- #
# Review routing
# --------------------------------------------------------------------------- #

def test_flagged_field_names_are_published_but_not_the_evidence():
    """
    An ERP can act on "hold phone and website". It cannot act on the
    legibility reasoning behind that, and does not need to carry it.
    """
    envelope = {
        "overall_confidence": 0.1,
        "needs_manual_review": True,
        "field_verification": {"flagged": [
            {"field": "phone", "value": "+123-456-7890", "status": "flagged",
             "reason": "source page was not legible enough to verify"},
            {"field": "website", "value": "www.example.com", "status": "flagged",
             "reason": "source page was not legible enough to verify"},
        ]},
        "fields": {"phone": "+123-456-7890", "website": "www.example.com"},
    }
    payload = build_erp_payload(envelope)
    assert payload["review_fields"] == ["phone", "website"]
    assert payload["review_required"] is True
    assert "reason" not in str(payload)


def test_review_fields_is_absent_when_nothing_was_flagged():
    assert "review_fields" not in build_erp_payload(PAN_ENVELOPE)


# --------------------------------------------------------------------------- #
# Values are projected, never rewritten
# --------------------------------------------------------------------------- #

def test_values_are_published_verbatim():
    """
    No number parsing and no date reformatting. Auto-correction is already
    recorded in this codebase as having corrupted cells the OCR read right;
    a projection is not the place to reintroduce it. Typing belongs to the
    integration, which knows its own ERP's schema.
    """
    envelope = {"fields": {
        "grand_total": "1,234.50",
        "invoice_date": "02/06/1990",
        "columns": ["Qty"],
        "rows": [["1,000"]],
    }}
    payload = build_erp_payload(envelope)
    assert payload["data"]["grand_total"] == "1,234.50"
    assert payload["data"]["invoice_date"] == "02/06/1990"
    assert payload["line_items"][0]["Qty"] == "1,000"


def test_devanagari_and_other_non_ascii_survive():
    envelope = {"fields": {"name": "Rajesh Kumar (राजेश कुमार)"}}
    assert build_erp_payload(envelope)["data"]["name"] == "Rajesh Kumar (राजेश कुमार)"


# --------------------------------------------------------------------------- #
# Robustness - this runs on every document, including malformed ones
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [None, {}, [], "not a dict", 0])
def test_a_missing_or_malformed_envelope_does_not_raise(bad):
    payload = build_erp_payload(bad)
    assert payload["document_type"] == "UNKNOWN"
    assert payload["data"] == {}
    assert payload["line_items"] == []


def test_a_bare_field_map_without_the_envelope_wrapper_is_tolerated():
    """An older stored row, or a direct pipeline call, can hand one over."""
    payload = build_erp_payload({"name": "Someone", "dob": "01/01/1980"})
    assert payload["data"] == {"name": "Someone", "dob": "01/01/1980"}


def test_unsupplied_identifiers_are_omitted_rather_than_sent_as_null():
    payload = build_erp_payload(PAN_ENVELOPE)
    assert "document_id" not in payload
    assert "request_id" not in payload

    payload = build_erp_payload(PAN_ENVELOPE, document_id=7, request_id="req_abc")
    assert payload["document_id"] == 7
    assert payload["request_id"] == "req_abc"


# --------------------------------------------------------------------------- #
# Rendering helper
# --------------------------------------------------------------------------- #

def test_column_order_spans_every_item_not_just_the_first():
    """
    Empty cells are dropped per item, so the first item is not a reliable
    header. Taking its keys alone would hide a column that only appears
    further down the table.
    """
    payload = {"line_items": [{"Item": "A"}, {"Item": "B", "Batch": "X"}]}
    assert line_item_columns(payload) == ["Item", "Batch"]


def test_column_helper_tolerates_junk():
    assert line_item_columns(None) == []
    assert line_item_columns({}) == []
    assert line_item_columns({"line_items": ["not a dict"]}) == []


# --------------------------------------------------------------------------- #
# "N/A" means absent
#
# A model writes "N/A" to say a field is not on the document. Published as a
# VALUE it did damage twice: an ERP writes the literal text "N/A" into a
# record, and field verification flags it as placeholder filler - which took a
# correctly extracted resume from 1.0 to 0.78 and marked it for review over
# two fields the document simply does not have.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("marker", ["N/A", "n/a", "NA", "None", "null", "-", "--",
                                    "Not Available", "not applicable", "unknown"])
def test_absent_markers_never_reach_the_erp(marker):
    from app.utils.structured_schemas import normalize_structured_data

    envelope = normalize_structured_data(
        "VISITING_CARD", {"name": "Arjun Mehta", "website": marker}, 0.99)
    payload = build_erp_payload(envelope)
    # The key is gone, not merely blanked. Compared on the values rather than
    # by substring: "NA" occurs inside "name", which is not a leak.
    assert payload["data"] == {"name": "Arjun Mehta"}
    assert marker not in payload["data"].values()


def test_absent_markers_are_stripped_inside_nested_entries():
    """Where they actually appear: a resume's education objects."""
    from app.utils.structured_schemas import normalize_structured_data

    envelope = normalize_structured_data("RESUME", {
        "candidate_name": "Subhajit Pati",
        "contact_info": {"email": "a@b.com", "linkedin": "N/A", "github": "None"},
        "education": [
            {"degree": "B.Pharm", "institution": "JNTUK",
             "graduation_year": "2025", "gpa_or_grade": "7.29"},
            {"degree": "Higher Secondary", "institution": "N/A",
             "field_of_study": "N/A", "graduation_year": "2021", "gpa_or_grade": "71%"},
        ],
    }, 0.99)
    data = build_erp_payload(envelope)["data"]

    assert data["contact_info"] == {"email": "a@b.com"}
    assert data["education"][0]["gpa_or_grade"] == "7.29"
    assert data["education"][1] == {"degree": "Higher Secondary",
                                    "graduation_year": "2021", "gpa_or_grade": "71%"}


def test_an_entry_that_is_entirely_absent_markers_is_dropped():
    from app.utils.structured_schemas import normalize_structured_data

    envelope = normalize_structured_data("RESUME", {
        "candidate_name": "Subhajit Pati",
        "certifications": [{"certification_name": "N/A", "issuing_organization": "N/A",
                            "issue_date": "N/A"}],
    }, 0.99)
    assert "certifications" not in build_erp_payload(envelope)["data"]


def test_a_real_value_that_merely_contains_na_is_kept():
    """`strip_absent_markers` matches the WHOLE value, never a substring."""
    from app.utils.structured_schemas import normalize_structured_data

    envelope = normalize_structured_data(
        "VISITING_CARD", {"name": "Nandini Nair", "city": "Nanded",
                          "company": "NA Logistics Pvt Ltd"}, 0.99)
    data = build_erp_payload(envelope)["data"]
    assert data["name"] == "Nandini Nair"
    assert data["city"] == "Nanded"
    assert data["company"] == "NA Logistics Pvt Ltd"


# --------------------------------------------------------------------------- #
# Why a document needs review, not just that it does
# --------------------------------------------------------------------------- #

def test_an_illegible_source_says_so_in_terms_the_sender_can_act_on():
    """
    "review_required: true" tells an operator nothing. "the image is 180x261"
    tells them to rescan. A real 180x261 pamphlet thumbnail scored 0.1 and
    the cause was only visible in the verification internals this payload
    deliberately drops.
    """
    envelope = {
        "overall_confidence": 0.1,
        "needs_manual_review": True,
        "field_verification": {
            "flagged": [],
            "source_legibility": {
                "legible": False,
                "reasons": ["source image is 180x261; text below ~400px on the "
                            "short edge is not reliably legible"],
            },
        },
        "fields": {"clinic_or_hospital": "BEST MEDICAL SERVICE"},
    }
    payload = build_erp_payload(envelope)
    assert payload["review_required"] is True
    assert "180x261" in payload["review_reason"]


def test_a_legible_source_carries_no_review_reason():
    envelope = {
        "overall_confidence": 1.0,
        "needs_manual_review": False,
        "field_verification": {"flagged": [],
                               "source_legibility": {"legible": True, "reasons": []}},
        "fields": {"name": "Arjun Mehta"},
    }
    assert "review_reason" not in build_erp_payload(envelope)
