"""
Deterministic Stock & Total Validation Service
Validates row-level stock arithmetic (closing = opening + receipt - issue),
reconciles column totals against summary rows, computes multi-factor confidence,
and flags documents requiring human review without mutating raw OCR extractions.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from app.core.logger import logger


class ValidationService:
    """
    Deterministic validation engine for pharmaceutical stock statements,
    inventory reports, and billing tables.
    """

    def validate_stock_table(
        self,
        columns: List[str],
        rows: List[List[str]],
        ocr_confidence: float = 0.90,
        layout_confidence: float = 0.85
    ) -> Dict[str, Any]:
        """
        Executes row-by-row arithmetic validation and summary totals checks.
        NEVER mutates or overwrites raw OCR values.
        """
        col_map = self._map_stock_columns(columns)
        validated_rows: List[Dict[str, Any]] = []
        arithmetic_matches = 0
        arithmetic_mismatches = 0
        arithmetic_checks = 0
        suspicious_cells: List[Dict[str, Any]] = []

        # Extract rows and validate stock equation: closing = opening + receipt - issue
        for row_idx, r in enumerate(rows):
            r_str = " ".join(r).lower()
            is_total_row = any(k in r_str for k in ["total", "totals:", "grand total", "net total"])

            row_data: Dict[str, Any] = {
                "row_index": row_idx + 1,
                "is_total_row": is_total_row,
                "raw_cells": r,
                "validation_status": "UNVERIFIED",
                "discrepancy": 0.0
            }

            if is_total_row:
                row_data["validation_status"] = "TOTAL_ROW"
                validated_rows.append(row_data)
                continue

            def _get_cell_str(idx: Optional[int]) -> Optional[str]:
                if idx is not None and idx < len(r):
                    s = r[idx].strip()
                    return s if s else None
                return None

            row_data["product_name"] = _get_cell_str(col_map.get("product_name"))
            row_data["item_code"] = _get_cell_str(col_map.get("item_code"))
            row_data["pack"] = _get_cell_str(col_map.get("pack"))
            row_data["batch"] = _get_cell_str(col_map.get("batch"))
            row_data["expiry"] = _get_cell_str(col_map.get("expiry"))
            row_data["rate"] = self._parse_num(r, col_map.get("rate"))

            # Parse numeric quantities if columns exist
            op_val = self._parse_num(r, col_map.get("opening_qty"))
            rec_val = self._parse_num(r, col_map.get("receipt_qty"))
            tot_val = self._parse_num(r, col_map.get("total_qty"))
            iss_val = self._parse_num(r, col_map.get("issue_qty"))
            cl_val = self._parse_num(r, col_map.get("closing_qty"))

            row_data["opening_qty"] = op_val
            row_data["receipt_qty"] = rec_val
            row_data["total_qty"] = tot_val
            row_data["issue_qty"] = iss_val
            row_data["closing_qty"] = cl_val

            # Evaluate arithmetic formula
            rec_term = rec_val if rec_val is not None else 0.0
            iss_term = iss_val if iss_val is not None else 0.0

            if tot_val is not None and iss_val is not None and cl_val is not None:
                arithmetic_checks += 1
                expected_closing = round(tot_val - iss_term, 2)
                row_data["calculated_closing_qty"] = expected_closing
                diff = round(abs(cl_val - expected_closing), 2)
                row_data["discrepancy"] = diff
                if diff <= 0.05:
                    row_data["validation_status"] = "VERIFIED"
                    arithmetic_matches += 1
                else:
                    row_data["validation_status"] = "MISMATCH"
                    arithmetic_mismatches += 1
                    suspicious_cells.append({
                        "row_index": row_idx,
                        "field": "closing_qty",
                        "column_index": col_map.get("closing_qty"),
                        "current_val": cl_val,
                        "expected_val": expected_closing,
                        "discrepancy": diff
                    })
            elif op_val is not None and cl_val is not None:
                arithmetic_checks += 1
                expected_closing = round(op_val + rec_term - iss_term, 2)
                row_data["calculated_closing_qty"] = expected_closing
                diff = round(abs(cl_val - expected_closing), 2)
                row_data["discrepancy"] = diff
                if diff <= 0.05:
                    row_data["validation_status"] = "VERIFIED"
                    arithmetic_matches += 1
                else:
                    row_data["validation_status"] = "MISMATCH"
                    arithmetic_mismatches += 1
                    suspicious_cells.append({
                        "row_index": row_idx,
                        "field": "closing_qty",
                        "column_index": col_map.get("closing_qty"),
                        "current_val": cl_val,
                        "expected_val": expected_closing,
                        "discrepancy": diff
                    })
            elif cl_val is None and op_val is not None:
                row_data["calculated_closing_qty"] = round(op_val + rec_term - iss_term, 2)
                row_data["validation_status"] = "REVIEW"

            validated_rows.append(row_data)

        # Totals Reconciliation
        totals_result = self._validate_column_totals(validated_rows, col_map)

        # Phase 12: Multi-Factor Deterministic Confidence Scoring
        # 1. Row structure confidence: assesses row volume and description coverage
        if len(validated_rows) == 0:
            row_struct_conf = 0.0
        elif len(validated_rows) >= 3:
            row_struct_conf = 1.0
        else:
            row_struct_conf = 0.60

        # 2. Column structure confidence: assesses semantic column completeness
        has_desc_col = ("product_name" in col_map)
        has_num_col = any(k in col_map for k in ["closing_qty", "total_qty", "opening_qty", "receipt_qty", "issue_qty", "rate"])
        if len(columns) >= 4 and has_desc_col and has_num_col:
            col_struct_conf = 1.0
        elif len(columns) >= 2 and (has_desc_col or has_num_col):
            col_struct_conf = 0.60
        else:
            col_struct_conf = 0.20

        # 3. Numeric confidence: checks parseability of numeric cells across numeric columns
        num_cells_total = 0
        num_cells_parsed = 0
        for vr in validated_rows:
            if vr.get("is_total_row"):
                continue
            for k in ["opening_qty", "receipt_qty", "issue_qty", "closing_qty", "rate"]:
                if k in col_map:
                    num_cells_total += 1
                    if vr.get(k) is not None:
                        num_cells_parsed += 1
        num_conf = float(num_cells_parsed / max(1, num_cells_total)) if num_cells_total > 0 else 0.85

        # 4. Cell accuracy confidence: proportion of data rows with valid descriptions
        desc_filled = sum(1 for vr in validated_rows if vr.get("product_name"))
        cell_acc_conf = float(desc_filled / max(1, len(validated_rows))) if validated_rows else 0.0

        # 5. Validation confidence: arithmetic match rate
        if arithmetic_checks > 0:
            val_conf = float(arithmetic_matches / arithmetic_checks)
        else:
            val_conf = 0.85

        # Composite overall confidence:
        composite_conf = (
            0.20 * float(ocr_confidence) +
            0.15 * float(layout_confidence) +
            0.15 * row_struct_conf +
            0.15 * col_struct_conf +
            0.15 * num_conf +
            0.20 * val_conf
        )

        # Structural and Arithmetic Penalties
        if len(validated_rows) == 0:
            composite_conf = 0.25
        elif arithmetic_mismatches > 0:
            mismatch_ratio = arithmetic_mismatches / max(1, arithmetic_checks)
            composite_conf = max(0.40, composite_conf - (0.30 * mismatch_ratio))

        overall_conf = round(composite_conf, 4)

        # Phase 12 Mandatory Human Review Triggers:
        needs_review = (
            arithmetic_mismatches > 0 or
            totals_result.get("mismatches_count", 0) > 0 or
            overall_conf < 0.75 or
            len(validated_rows) == 0 or
            len(columns) < 2
        )

        return {
            "columns": columns,
            "col_map": col_map,
            "validated_rows": validated_rows,
            "suspicious_cells": suspicious_cells,
            "totals_validation": totals_result,
            "arithmetic_checks": arithmetic_checks,
            "arithmetic_matches": arithmetic_matches,
            "arithmetic_mismatches": arithmetic_mismatches,
            "ocr_confidence": round(ocr_confidence, 4),
            "layout_confidence": round(layout_confidence, 4),
            "row_structure_confidence": round(row_struct_conf, 4),
            "col_structure_confidence": round(col_struct_conf, 4),
            "cell_accuracy_confidence": round(cell_acc_conf, 4),
            "numeric_confidence": round(num_conf, 4),
            "validation_confidence": round(val_conf, 4),
            "overall_confidence": overall_conf,
            "needs_manual_review": needs_review
        }


    def _map_stock_columns(self, columns: List[str]) -> Dict[str, int]:
        """Identifies column indices for opening, receipt, issue, closing quantities, and item metadata."""
        col_map: Dict[str, int] = {}
        for idx, col in enumerate(columns):
            c_upper = col.upper().strip()

            # Disambiguate Qty vs Value
            is_val = "VAL" in c_upper or "AMOUNT" in c_upper or "AMT" in c_upper

            if ("OP" in c_upper or "OPENING" in c_upper) and not is_val:
                col_map["opening_qty"] = idx
            elif ("REC" in c_upper or "PUR" in c_upper or "RECEIPT" in c_upper or "PURCHASE" in c_upper or c_upper == "IN") and not is_val:
                col_map["receipt_qty"] = idx
            elif ("ISS" in c_upper or "SALE" in c_upper or "DISPATCH" in c_upper or "SOLD" in c_upper or c_upper == "OUT") and not is_val:
                col_map["issue_qty"] = idx
            elif ("TOTAL" in c_upper or "TOT" in c_upper) and not is_val:
                col_map["total_qty"] = idx
            elif ("CL" in c_upper or "CLOSING" in c_upper or "BAL" in c_upper or "BALANCE" in c_upper) and not is_val:
                col_map["closing_qty"] = idx
            elif is_val and ("OP" in c_upper or "OPENING" in c_upper):
                col_map["opening_val"] = idx
            elif is_val and ("CL" in c_upper or "CLOSING" in c_upper or "BAL" in c_upper):
                col_map["closing_val"] = idx
            elif any(k in c_upper for k in ["PRODUCT", "ITEM DESCRIPTION", "PARTICULARS", "DESCRIPTION"]) or (c_upper == "ITEM"):
                col_map["product_name"] = idx
            elif any(k in c_upper for k in ["CODE", "SLNO", "SL NO", "S.NO", "ITEM CODE"]):
                col_map["item_code"] = idx
            elif any(k in c_upper for k in ["PACK", "UNIT", "PACKING"]):
                col_map["pack"] = idx
            elif any(k in c_upper for k in ["BATCH"]):
                col_map["batch"] = idx
            elif any(k in c_upper for k in ["EXP", "EXPIRY", "N.EXP", "M.EXP"]):
                col_map["expiry"] = idx
            elif any(k in c_upper for k in ["RATE", "PRICE", "MRP"]):
                col_map["rate"] = idx

        # Default fallback for product_name if not explicitly named
        if "product_name" not in col_map and len(columns) > 0:
            # Check if column 0 is slno/code
            if len(columns) > 1 and "item_code" in col_map and col_map["item_code"] == 0:
                col_map["product_name"] = 1
            else:
                col_map["product_name"] = 0

        return col_map

    def _parse_num(self, row: List[str], col_idx: Optional[int]) -> Optional[float]:
        """Parses a float or integer number from a table cell, handling blanks and negatives."""
        if col_idx is None or col_idx >= len(row):
            return None
        val_str = row[col_idx].strip()
        if not val_str or val_str in ["-", "--", "---", "NA", "N/A"]:
            return None
        try:
            # Strip commas and currency symbols
            clean = val_str.replace(",", "").replace("$", "").replace("₹", "")
            return float(clean)
        except ValueError:
            # Handle possible attached suffix like '12TAB'
            import re
            m = re.search(r'^-?\d+(?:\.\d+)?', val_str.replace(",", ""))
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    return None
            return None

    def _validate_column_totals(
        self,
        validated_rows: List[Dict[str, Any]],
        col_map: Dict[str, int]
    ) -> Dict[str, Any]:
        """Calculates column sums across data rows and compares with printed total row."""
        data_rows = [r for r in validated_rows if not r.get("is_total_row")]
        total_rows = [r for r in validated_rows if r.get("is_total_row")]

        res: Dict[str, Any] = {
            "has_printed_totals": len(total_rows) > 0,
            "column_sums": {},
            "printed_totals": {},
            "mismatches_count": 0,
            "status": "NO_TOTALS"
        }

        if not total_rows:
            return res

        # Sum quantities for data rows
        t_row_cells = total_rows[0]["raw_cells"]
        mismatches = 0

        for field in ["opening_qty", "receipt_qty", "issue_qty", "closing_qty"]:
            col_idx = col_map.get(field)
            if col_idx is not None:
                calc_sum = round(sum(r.get(field, 0.0) or 0.0 for r in data_rows), 2)
                res["column_sums"][field] = calc_sum

                printed_val = self._parse_num(t_row_cells, col_idx)
                if printed_val is not None:
                    res["printed_totals"][field] = printed_val
                    if abs(calc_sum - printed_val) > 1.0:
                        mismatches += 1

        res["mismatches_count"] = mismatches
        res["status"] = "VERIFIED" if mismatches == 0 else "MISMATCH"
        return res


validation_service = ValidationService()
