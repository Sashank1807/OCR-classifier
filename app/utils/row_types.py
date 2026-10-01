"""
Minimal logical-row taxonomy for table row assembly.

This is intentionally small: it covers exactly the row categories the current
benchmark documents need to distinguish, rather than a large speculative
taxonomy. In particular PARTY_OR_HEADER is a single bucket for "this physical
line is a real row in the table but not a transaction" - both customer/party
name rows and section/division banner lines - because both are handled
identically downstream: kept in the row stream with description text
preserved, numeric columns left EMPTY (never fabricated), never merged into a
neighboring PRODUCT row, and never dropped.

Row classification here is deliberately geometry/position/structure-driven
(numeric fill ratio, position in the table, ruling-line boundaries), not
vocabulary-driven - matching the project's rule that banner-vs-data decisions
must not rely on company/product name keyword lists.
"""

from typing import Optional


PRODUCT = "PRODUCT"
CONTINUATION = "CONTINUATION"
SUBTOTAL = "SUBTOTAL"
TOTAL = "TOTAL"
PARTY_OR_HEADER = "PARTY_OR_HEADER"
UNKNOWN_ROW = "UNKNOWN_ROW"

ALL_ROW_TYPES = (PRODUCT, CONTINUATION, SUBTOTAL, TOTAL, PARTY_OR_HEADER, UNKNOWN_ROW)


def score_is_banner(
    numeric_fill_ratio: float,
    is_first_data_row: bool,
    has_ruling_line_above: bool,
    desc_continues_previous_row: bool,
) -> bool:
    """
    Structural (not vocabulary-based) signal for "this line is a
    PARTY_OR_HEADER row rather than a PRODUCT row".

    A line qualifies when it contributes essentially no numeric/transaction
    data to the table (numeric_fill_ratio near zero) and it does not read as
    a wrapped continuation of the previous row's description. Position and
    ruling-line context are additional corroborating signals but neither
    alone is required, since a party row can appear either before the first
    product (pre-table banner position) or interleaved between products.
    """
    if desc_continues_previous_row:
        return False
    if numeric_fill_ratio > 0.15:
        return False
    return True
