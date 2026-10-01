"""
    Cell-Level OCR Service & Multi-Level Cascade Architecture
Provides fine-grained, localized OCR and image analysis for individual table cells:
1. Spanning Numeric Token Splitting (resolves adjacent column numeric bleed while protecting text)
2. Fused Description & Packing Extraction (preserves 1X10, 10TAB, 15GM, 10'S)
3. Image-Based Dash ('-') vs Empty ('') Morphological Disambiguation
4. Image Quality Scoring per Cell (intensity, contrast, variance, character height, glare)
5. Multi-Level Cascade:
   - Level 1: Fast Accept for high-confidence, well-formatted RapidOCR candidates
   - Level 2: Adaptive 5-Variant Small-Cell Re-OCR & Deterministic Candidate Ranking
   - Level 3: Selective Qwen2.5-VL Exact-Cell Crop Verification (capped per doc)
6. Dot-Matrix Character Disambiguation (0 vs 6/8, 1 vs I, 5 vs S, 3 vs 8, 10.9 vs 10.S)
7. Specialized Narrow Column Extraction (Sn., Unit, Balance)
8. Visual Cell Debug Export with 4-Panel Comparison
"""

import cv2
import numpy as np
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from app.core.config import settings
from app.core.logger import logger
from app.services.rapid_ocr_service import rapid_ocr_service
from app.services.model_service import model_engine


class CellType:
    DESCRIPTION = "DESCRIPTION"
    PACKING = "PACKING"
    INTEGER_QTY = "INTEGER_QTY"
    DECIMAL_VALUE = "DECIMAL_VALUE"
    RATE = "RATE"
    AMOUNT = "AMOUNT"
    DATE = "DATE"
    PERCENTAGE = "PERCENTAGE"
    SERIAL = "SERIAL"
    UNIT = "UNIT"
    DASH_OR_EMPTY = "DASH_OR_EMPTY"


PACKING_REGEX = re.compile(
    r'\b(\d+\s*[xX*]\s*\d+(?:\s*(?:tab|tabs|cap|caps|ml|gm|gms|bot|bottle|s))?|\d+[\'’.]?s|\d+\s*(?:ml|gm|gms|tab|tabs|cap|caps|vial|amp|kg|ltr|bot|bottle|g|mg|sch\.?))\b',
    re.IGNORECASE
)


class CellOCRService:
    """
    Dedicated fine-grained cell-level OCR cascade engine.
    Refines table extraction after global row & column geometry is established.
    """

    def __init__(self):
        self.engine = rapid_ocr_service.engine

    # =========================================================================
    # 0. Fine-Grained Cell Type Classification (Priority 2)
    # =========================================================================
    def classify_cell_type(
        self,
        col_name: str,
        sample_val: str = "",
        neighbor_types: Optional[List[str]] = None
    ) -> str:
        """
        Classifies each cell before OCR into one of 11 fine-grained types:
        DESCRIPTION, PACKING, INTEGER_QTY, DECIMAL_VALUE, RATE, AMOUNT,
        DATE, PERCENTAGE, SERIAL, UNIT, DASH_OR_EMPTY.
        Uses word-boundary column matching and sample value inspection.
        """
        cl = (col_name or "").strip().lower()
        val = (sample_val or "").strip()

        # 1. Serial / Ordinal numbers
        if re.search(r'\b(?:sn|sl|sr|s\.?no|sr\.?no|serial|no\.)\b', cl):
            return CellType.SERIAL

        # 2. Unit of measurement
        if re.search(r'\b(?:unit|uom)\b', cl):
            return CellType.UNIT

        # 3. Packaging
        if re.search(r'\b(?:pack|pkg|packing)\b', cl):
            return CellType.PACKING

        # 4. Description / Product Name / Code / Batch
        if re.search(r'\b(?:description|desc|product|item|particulars?|name|trade|code|hsn|batch|lot)\b', cl):
            return CellType.DESCRIPTION

        # 5. Rate / Price
        if re.search(r'\b(?:rate|price|mrp|ptr|pts)\b', cl):
            return CellType.RATE

        # 6. Amount / Valuation
        if re.search(r'\b(?:amount|amt|total|value|val)\b', cl):
            return CellType.AMOUNT

        # 7. Date
        if re.search(r'\b(?:date|dt|exp|mfg)\b', cl):
            return CellType.DATE

        # 8. Percentage / Tax / Discount
        if re.search(r'\b(?:%|percent|percentage|disc|discount|tax|gst|vat)\b', cl) or "%" in cl:
            return CellType.PERCENTAGE

        # 9. Pure discrete integer quantities (Free, Scheme, Box, Case, Pcs)
        if re.search(r'\b(?:free|scheme|sch|case|box|pcs)\b', cl):
            if "." in val and not val.endswith(".00"):
                return CellType.DECIMAL_VALUE
            return CellType.INTEGER_QTY

        # 10. General Stock Quantities (Opening, Receipt, Issue, Closing, Balance, In, Out, Qty)
        if re.search(r'\b(?:qty|quantity|opening|op\.?bal|receipt|issue|closing|cl\.?bal|balance|bal|in|out|stock)\b', cl):
            # If explicit non-zero decimal observed in sample, classify as decimal
            if "." in val and not val.endswith(".00"):
                return CellType.DECIMAL_VALUE
            return CellType.DECIMAL_VALUE

        # Fallback based on value content
        if val in ["-", "--", "---", "一", "—"]:
            return CellType.DASH_OR_EMPTY
        if val and self._is_valid_num(val):
            return CellType.DECIMAL_VALUE
        return CellType.DESCRIPTION

    # =========================================================================
    # 1. Spanning Numeric Token Splitting (Gated strictly to numeric components)
    # =========================================================================
    def split_spanning_tokens(
        self,
        tokens: List[Dict[str, Any]],
        col_bounds: List[Tuple[str, float, float]]
    ) -> List[Dict[str, Any]]:
        """
        Detects tokens that span across multiple column boundaries (e.g. '9 39' spanning
        Issue and Closing columns, or '101 16725.60' spanning Closing Qty and Amount).
        Splits them logically into sub-tokens assigned to their proper columns.

        CRITICAL GUARD: Only splits if ALL parts match numeric or dash representations.
        Never splits text descriptions (e.g. 'BORIT SB 130', 'MINOSTRONG 60ML').
        """
        if not tokens or not col_bounds or len(col_bounds) < 2:
            return tokens

        refined_tokens: List[Dict[str, Any]] = []

        def is_numeric_dash_or_packing(s: str) -> bool:
            clean = s.replace(",", "").replace("$", "").replace("₹", "").strip()
            if bool(re.match(r'^-?\d+(?:\.\d+)?$', clean)) or clean in ["-", "--", "---", "一", "—"]:
                return True
            return bool(PACKING_REGEX.match(clean))

        for tok in tokens:
            x0 = tok.get("x0", 0.0)
            x1 = tok.get("x1", 0.0)
            text = (tok.get("text") or "").strip()

            # Identify which columns x0 and x1 fall into
            col_start = None
            col_end = None
            for c_idx, (c_name, b_left, b_right) in enumerate(col_bounds):
                if b_left <= x0 < b_right:
                    col_start = c_idx
                if b_left <= x1 <= b_right or (c_idx == len(col_bounds) - 1 and x1 >= b_left):
                    col_end = c_idx

            # If token starts and ends in the same column, no split needed
            if col_start is not None and col_end is not None and col_start == col_end:
                refined_tokens.append(tok)
                continue

            parts = text.split()
            # STRICT GUARD: Only split if parts are numeric/dash/packing and span across different columns
            if (
                len(parts) >= 2 and
                col_start is not None and col_end is not None and col_end > col_start and
                all(is_numeric_dash_or_packing(p) for p in parts)
            ):
                total_chars = max(1, sum(len(p) for p in parts))
                span_w = max(1.0, x1 - x0)
                cur_x = x0

                sub_tokens = []
                for p_idx, part in enumerate(parts):
                    part_w = (len(part) / total_chars) * span_w
                    part_x0 = cur_x
                    part_x1 = cur_x + part_w
                    cur_x = part_x1 + (span_w * 0.05)  # space advance

                    sub_tokens.append({
                        "text": part,
                        "x0": round(part_x0, 1),
                        "x1": round(part_x1, 1),
                        "y0": tok.get("y0", 0.0),
                        "y1": tok.get("y1", 0.0),
                        "xc": round((part_x0 + part_x1) / 2.0, 1),
                        "yc": tok.get("yc", tok.get("y_center", 0.0)),
                        "y_center": tok.get("yc", tok.get("y_center", 0.0)),
                        "height": tok.get("height", 10.0),
                        "width": round(part_w, 1),
                        "confidence": tok.get("confidence", 0.85),
                        "split_from": text
                    })

                refined_tokens.extend(sub_tokens)
                logger.debug(f"Split numeric spanning token '{text}' into {[st['text'] for st in sub_tokens]}")
            else:
                refined_tokens.append(tok)

        return refined_tokens

    # =========================================================================
    # 2. Fused Packing Extraction from Description Strings
    # =========================================================================
    def extract_fused_packing(self, desc_text: str) -> Tuple[str, str]:
        """
        Detects packaging patterns (e.g. '1X10', '1X60ML', '10'S', '10TAB', '15GM')
        fused within a product description string.
        Returns: (cleaned_description, extracted_packing)
        """
        if not desc_text:
            return "", ""

        # Find all packaging occurrences
        matches = list(PACKING_REGEX.finditer(desc_text))
        if not matches:
            return desc_text, ""

        # Ignore pure dosage strengths (e.g. '1.25MG', '2.5MG', '40MG', '500MG', '100MCG')
        # because active ingredient strength is part of the drug trade name, not packaging
        valid_matches = [m for m in matches if not re.match(r'^\d+(?:\.\d+)?\s*(?:mg|mcg|iu)$', m.group(0).strip(), re.IGNORECASE)]
        if not valid_matches:
            return desc_text, ""

        # Use the last packaging token (typically at the end of the product description)
        last_m = valid_matches[-1]
        packing_val = last_m.group(0).strip()
        cleaned_desc = (desc_text[:last_m.start()] + desc_text[last_m.end():]).strip()
        cleaned_desc = re.sub(r'\s+', ' ', cleaned_desc).strip()

        # If description became empty, keep original description
        if not cleaned_desc:
            return desc_text, packing_val

        return cleaned_desc, packing_val

    # =========================================================================
    # 3. Image-Based Dash ('-') vs Empty ('') Disambiguation
    # =========================================================================
    def detect_cell_dash_vs_empty(
        self,
        image: np.ndarray,
        cell_bbox: Tuple[int, int, int, int]
    ) -> Tuple[str, str, float]:
        """
        Inspects exact pixel crop for an empty cell.
        Returns ('-', 'NOT_REPORTED', conf) if a horizontal printed dash is found,
        or ('', 'EMPTY', 1.0) if cell is truly blank.
        """
        x0, y0, x1, y1 = [int(v) for v in cell_bbox]
        h_img, w_img = image.shape[:2]

        # Clamp boundaries
        x0 = max(0, min(x0, w_img - 1))
        x1 = max(0, min(x1, w_img))
        y0 = max(0, min(y0, h_img - 1))
        y1 = max(0, min(y1, h_img))

        if x1 - x0 < 8 or y1 - y0 < 6:
            return "", "EMPTY", 1.0

        crop = image[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # Remove cell border ruling line artifacts (margins of 2-3px)
        ch, cw = gray.shape[:2]
        if ch > 6 and cw > 6:
            gray_inner = gray[2:ch-2, 2:cw-2]
        else:
            gray_inner = gray

        # Dynamic background and ink thresholding
        bg_median = float(np.median(gray_inner))
        min_val = float(np.min(gray_inner))
        contrast = int(np.max(gray_inner)) - int(min_val)
        var_val = float(np.var(gray_inner))

        # Truly flat paper without ink
        if contrast < 14 and var_val < 25.0:
            return "", "EMPTY", 1.0

        dark_diff = max(14.0, (bg_median - min_val) * 0.45)
        dark_mask = (gray_inner < (bg_median - dark_diff)).astype(np.uint8)
        total_dark = int(np.sum(dark_mask))
        if total_dark == 0:
            return "", "EMPTY", 1.0

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        inner_h, inner_w = gray_inner.shape[:2]

        dash_found = False
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]

            y_top = stats[i, cv2.CC_STAT_TOP]
            y_bot = y_top + h
            x_left = stats[i, cv2.CC_STAT_LEFT]
            x_right = x_left + w

            # Reject table ruling lines touching cell margins or spanning cell width
            if y_top <= 1 or y_bot >= inner_h - 1:
                continue
            if float(w) / max(1.0, float(inner_w)) >= 0.65:
                continue
            if x_left <= 1 and x_right >= inner_w - 2:
                continue

            area = stats[i, cv2.CC_STAT_AREA]
            aspect = w / max(1.0, float(h))
            if aspect >= 1.8 and 8 <= w <= 50 and 2 <= h <= 12 and area >= 20:
                v_dist = abs(cy - (inner_h / 2.0)) / max(1.0, float(inner_h))
                h_dist = abs(cx - (inner_w / 2.0)) / max(1.0, float(inner_w))
                if v_dist <= 0.38 and h_dist <= 0.45:
                    # Verify ink darkness: component ink must be substantially darker than surrounding paper
                    mask = (labels == i)
                    comp_mean = float(np.mean(gray_inner[mask]))
                    bg_mean = float(np.mean(gray_inner[~mask])) if np.any(~mask) else 255.0
                    if bg_mean - comp_mean >= 22.0:
                        dash_found = True
                        break

        if dash_found:
            return "-", "NOT_REPORTED", 0.90

        return "", "EMPTY", 1.0

    def is_cell_visually_empty(
        self,
        crop: np.ndarray,
        col_name: str = "",
        expected_type: str = "text",
        candidates: Optional[List[Dict[str, Any]]] = None,
        neighbor_info: Optional[Dict[str, Any]] = None,
        row_type: str = "product"
    ) -> bool:
        """
        Determines whether a cell is genuinely empty based on multi-feature visual and contextual evidence:
        1. Expected column type & row type (e.g. Total rows have expected blank serial number / unit)
        2. Connected component stroke analysis (area >= 16, width >= 4, height >= 7, stroke darkness >= 25)
        3. Screen photo texture discrimination (repetitive 1-2px scan lines without character connectivity)
        4. OCR candidate existence across variants (if no candidate found and no character stroke exists)
        5. Neighboring row occupancy pattern
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
            return True

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        ch, cw = gray.shape[:2]
        if ch > 6 and cw > 6:
            inner = gray[2:ch-2, 2:cw-2]
        else:
            inner = gray

        if inner.size == 0:
            return True

        # Total rows frequently have empty serial number, unit, etc.
        if row_type == "total" and any(k in col_name.lower() for k in ["sl", "sn", "sr", "pack", "unit", "product", "item", "particular"]):
            if not candidates or all(c.get("confidence", 0.0) < 0.40 for c in candidates):
                return True

        # Dynamic flat paper check: extremely low contrast and tiny variance
        contrast = int(np.max(inner)) - int(np.min(inner))
        variance = float(np.var(inner))
        if contrast < 12 and variance < 20.0:
            return True

        bg_median = float(np.median(inner))
        min_val = float(np.min(inner))
        dark_diff = max(13.0, (bg_median - min_val) * 0.45)
        dark_mask = inner < (bg_median - dark_diff)
        total_dark = int(np.sum(dark_mask))
        if total_dark == 0:
            return True

        ink_density = total_dark / max(1, inner.size)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask.astype(np.uint8), connectivity=8)
        char_components = []
        stripe_dark_pixels = 0
        stripe_boxes = []  # (x0, y0, w, h, area) - candidates for the merge-check below

        for i in range(1, num_labels):
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y0 = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            aspect = w / max(1.0, float(h))

            # Screen scan lines / CRT subpixel stripes have width <= 2
            if w <= 2 and h >= 8:
                stripe_dark_pixels += area
                stripe_boxes.append((x0, y0, w, h, area))
            elif w >= 3 and h >= 5 and area >= 12 and 0.15 <= aspect <= 4.0:
                char_components.append((w, h, area))

        # A genuine faint/broken dot-matrix stroke can fragment into several narrow
        # (width <= 2) pieces stacked in the same x-band - which the stripe-rejection
        # bucket above would otherwise discard entirely as "screen noise", permanently
        # losing real ink. A true periodic scan-line/moire pattern instead spreads its
        # stripe components evenly across most of the cell width. Distinguish the two
        # by clustering stripe x-positions: 2-4 fragments confined to a single narrow
        # x-band (not spread across the cell) are reclassified as one merged character
        # component rather than noise.
        if not char_components and len(stripe_boxes) >= 2:
            xs = sorted(b[0] for b in stripe_boxes)
            x_span = xs[-1] - xs[0]
            if x_span <= 6.0 and len(stripe_boxes) <= 4:
                merged_w = max(b[0] + b[2] for b in stripe_boxes) - min(b[0] for b in stripe_boxes)
                merged_h = max(b[1] + b[3] for b in stripe_boxes) - min(b[1] for b in stripe_boxes)
                merged_area = sum(b[4] for b in stripe_boxes)
                char_components.append((merged_w, merged_h, merged_area))
                stripe_dark_pixels -= merged_area

        # If stripe pixels dominate screen photo or no character components exist with very low ink density
        if not char_components:
            return True
        if total_dark > 0 and (stripe_dark_pixels / float(total_dark)) > 0.50 and len(char_components) <= 1:
            return True

        # If no candidates were detected by RapidOCR across any variant
        if not candidates or len(candidates) == 0:
            # Only declare visually empty if no character components exist or total ink is negligible
            if not char_components or (total_dark < 8 and ink_density < 0.003):
                return True

        return False

    # =========================================================================
    # 4. Image Quality Metric per Cell Crop
    # =========================================================================
    def compute_cell_quality(self, crop: np.ndarray) -> Dict[str, Any]:
        """
        Calculates localized image quality metrics for a cell crop:
        mean intensity, variance, contrast, foreground ratio, character height, glare.
        """
        if crop is None or crop.size == 0:
            return {
                "mean_intensity": 0.0,
                "variance": 0.0,
                "contrast": 0.0,
                "foreground_ratio": 0.0,
                "char_height": 0.0,
                "glare_ratio": 0.0,
                "is_faint": False,
                "is_glare": False
            }

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        h, w = gray.shape[:2]

        mean_int = float(np.mean(gray))
        variance = float(np.var(gray))
        contrast = float(np.max(gray) - np.min(gray))
        glare_ratio = float(np.sum(gray > 240) / max(1, h * w))

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fg_ratio = float(np.sum(thresh == 255) / max(1, h * w))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        char_heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels) if stats[i, cv2.CC_STAT_HEIGHT] >= 4]
        med_height = float(np.median(char_heights)) if char_heights else 0.0

        is_faint = contrast < 35.0 or variance < 100.0
        is_glare = glare_ratio > 0.20

        return {
            "mean_intensity": round(mean_int, 2),
            "variance": round(variance, 2),
            "contrast": round(contrast, 2),
            "foreground_ratio": round(fg_ratio, 3),
            "char_height": round(med_height, 1),
            "glare_ratio": round(glare_ratio, 3),
            "is_faint": is_faint,
            "is_glare": is_glare
        }

    # =========================================================================
    # 5. Multi-Variant Cell Preprocessing (5 Variants)
    # =========================================================================
    def generate_cell_variants(self, crop: np.ndarray, quality: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
        """
        Generates 5 tailored visual variants preserving thin strokes and dot-matrix pins:
        A: Original crop (or mild upscale)
        B: Grayscale + CLAHE (contrast enhancement)
        C: Adaptive Otsu binarization
        D: Edge-sharpened (unsharp mask)
        E: Mild upscaled original (2x/3x bicubic)
        """
        ch, cw = crop.shape[:2]
        scale = 3.0 if (ch < 25 or cw < 45) else (2.0 if (ch < 38 or cw < 80) else 1.0)

        # Scale crop
        if scale > 1.0:
            crop_up = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
        else:
            crop_up = crop.copy()

        # Add white border padding on small crops so DBNet text detector captures isolated glyphs
        if ch < 32 or cw < 70:
            pad_y = max(12, int(18 * min(1.0, ch / 30.0)))
            pad_x = max(20, int(30 * min(1.0, cw / 60.0)))
            crop_up = cv2.copyMakeBorder(crop_up, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # Variant A: Original crop (padded/scaled)
        var_a = crop_up

        # Variant B: Grayscale + CLAHE
        gray = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY) if len(crop_up.shape) == 3 else crop_up
        clip = 4.0 if quality.get("is_faint") else 2.5
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
        var_b_gray = clahe.apply(gray)
        var_b = cv2.cvtColor(var_b_gray, cv2.COLOR_GRAY2BGR)

        # Variant C: Otsu
        _, otsu = cv2.threshold(var_b_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        var_c = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)

        # Variant D: Sharpened unsharp mask
        blurred = cv2.GaussianBlur(crop_up, (0, 0), 1.5)
        var_d = cv2.addWeighted(crop_up, 1.6, blurred, -0.6, 0)
        if len(var_d.shape) == 2:
            var_d = cv2.cvtColor(var_d, cv2.COLOR_GRAY2BGR)

        # Variant E: Direct 3x bicubic with margin padding
        var_e = cv2.resize(crop, (max(1, cw * 3), max(1, ch * 3)), interpolation=cv2.INTER_CUBIC)
        var_e = cv2.copyMakeBorder(var_e, 20, 20, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        return [
            ("original", var_a),
            ("clahe", var_b),
            ("otsu", var_c),
            ("sharpened", var_d),
            ("upscaled", var_e)
        ]

    # =========================================================================
    # 6. Dot-Matrix Character Disambiguation & Type Validation
    # =========================================================================
    def disambiguate_numeric(self, text: str, expected_type: str, col_name: str = "") -> str:
        """
        Disambiguates common dot-matrix font confusions:
        0 <-> 6, 0 <-> 8, 1 <-> I/l, 5 <-> S, 3 <-> 8, 10.9 <-> 10.S, 1GRAN <-> 1GRAM.
        """
        t = (text or "").strip()
        is_num = expected_type in [
            CellType.INTEGER_QTY, CellType.DECIMAL_VALUE, CellType.RATE,
            CellType.AMOUNT, CellType.PERCENTAGE, "numeric", "decimal", "integer", "percentage"
        ]
        is_pack = (expected_type in [CellType.PACKING, "packing"] or any(k in col_name.lower() for k in ["pack", "pkg", "unit"])) and not is_num
        is_sn = (expected_type in [CellType.SERIAL, "serial"] or any(k in col_name.lower() for k in ["sn", "sl", "sr", "s.no"])) and not is_num

        if is_num:
            # Check for dot-matrix multi-zero representations
            if t in ["0000", "000", "000o", "000O", "0000E", "0000e", "0010", "000:0", "000.0", "000.00", "OOO0", "OOOPE"] or re.match(r'^[0oO:.\s]{2,6}$', t) or re.match(r'^(?:0{2,5}|000[oOeE]|0010|000:0)$', t, re.IGNORECASE):
                is_decimal_col = expected_type in [CellType.DECIMAL_VALUE, "decimal"] or any(k in col_name.lower() for k in ["open", "in", "out", "bal", "clos", "rec", "iss", "amt", "rate"])
                return "0.000" if is_decimal_col else "0"

            # Single letter to digit
            t = re.sub(r'^[oO]$', '0', t)
            t = re.sub(r'^[lI|]$', '1', t)
            t = re.sub(r'^[sS]$', '5', t)
            t = re.sub(r'^[bB]$', '8', t)
            t = re.sub(r'^[θ]$', '0', t)
            # Dot-matrix digit confusions inside numbers (leading, middle, trailing, and adjacent to decimals)
            t = re.sub(r'(?<=[0-9\.])[oO](?=[0-9\.])', '0', t)
            t = re.sub(r'^[oO](?=[0-9\.])', '0', t)
            t = re.sub(r'(?<=[0-9\.])[oO]$', '0', t)

            t = re.sub(r'(?<=[0-9\.])[lI|](?=[0-9\.])', '1', t)
            t = re.sub(r'^[lI|](?=[0-9\.])', '1', t)
            t = re.sub(r'(?<=[0-9\.])[lI|]$', '1', t)

            t = re.sub(r'(?<=[0-9\.])[sS](?=[0-9\.])', '5', t)
            t = re.sub(r'^[sS](?=[0-9\.])', '5', t)
            t = re.sub(r'(?<=[0-9\.])[sS]$', '5', t)

            t = re.sub(r'(?<=[0-9\.])[bB](?=[0-9\.])', '8', t)
            t = re.sub(r'^[bB](?=[0-9\.])', '8', t)
            t = re.sub(r'(?<=[0-9\.])[bB]$', '8', t)

            # Normalize decimal commas and middle dots
            t = t.replace("·", ".").replace("，", ".")
            # Clean spaces between digits (e.g. '10 0.00' or '25 0')
            if re.search(r'\d\s+\d', t):
                # Only collapse spaces if surrounded by digits on both sides
                t = re.sub(r'(?<=\d)\s+(?=\d)', '', t)

        elif is_pack:
            # Packaging fixes in dot-matrix
            # 10.9, 10.8, 10:8, 10.5, 6.8, 6.9 -> 10.S, 6.S
            t = re.sub(r'^(\d+)[\.:]([985])\.?$', r"\1.S", t)
            t = re.sub(r'^(\d+)\.S\.?$', r"\1.S", t, flags=re.IGNORECASE)
            t = re.sub(r'1GRAN$', '1GRAM', t, flags=re.IGNORECASE)
            t = re.sub(r'(\d+)TAB$', r"\1TAB", t, flags=re.IGNORECASE)
            # Collapse spacing around the pack separator ("1 x 10" -> "1x10"),
            # but do NOT rewrite the separator itself. Documents genuinely
            # differ - some print "1X10", others "1*10" (both confirmed against
            # the source images) - and PACKING_REGEX accepts either, so forcing
            # one spelling only destroys a real distinction between documents.
            t = re.sub(r'(\d+)\s*([xX*])\s*(\d+)', r'\1\2\3', t)

        elif is_sn:
            # Serial numbers are discrete 1, 2, 3, ...
            t = re.sub(r'^[oO]$', '0', t)
            t = re.sub(r'^[lI|]$', '1', t)
            t = re.sub(r'^[sS]$', '5', t)
            t = re.sub(r'^[bB]$', '8', t)

        elif expected_type == CellType.DESCRIPTION or any(k in col_name.lower() for k in ["desc", "particular", "product", "item"]):
            # Normalize letter O before decimal in numbers (e.g. O.058 -> 0.058)
            t = re.sub(r'\b[oO]\.(\d+)', r'0.\1', t)
            # Separate dosage forms and formulation indicators fused by tight OCR / dot-matrix print
            t = re.sub(r'([A-Za-z]{3,})(\d+(?:\.\d+)?\s*(?:mg|ml|gm|mcg|kg|g|%))\b', r'\1 \2', t, flags=re.IGNORECASE)
            t = re.sub(r'([A-Za-z]{3,})(TAB|CAP|CAPS|TABS|SYP|INJ|OINT|GEL|DROPS|SUSP|SOLUTION|SOLN|LOTION)\b', r'\1 \2', t, flags=re.IGNORECASE)
            t = re.sub(r'([A-Za-z]{3,})(DS|DSR|CV|SB|MR|OZ|SR|XL|OD|PLUS|FORTE)(?=\d|\b)', r'\1 \2 ', t, flags=re.IGNORECASE)
            # Dot-matrix percentage confusions in product names (e.g. 0.058CREAM -> 0.05% CREAM, 108 GEL -> 10% GEL)
            t = re.sub(r'(\d+(?:\.\d+)?)(?:8|[oO])\s*(CREAM|GEL|LOTION|OINT|OINTMENT|SOLUTION)\b', r'\1% \2', t, flags=re.IGNORECASE)
            # Hyphen before formulation: MINOSTRONG-FSOLUTION -> MINOSTRONG-F SOLUTION
            t = re.sub(r'([A-Za-z]+)-(F|H|D|T)(SOLUTION|TAB|CAP|GEL)', r'\1-\2 \3', t, flags=re.IGNORECASE)
            # Fused dosage forms: TOFUSTAB -> TOFUS TAB, ADABORGEL -> ADABOR GEL
            t = re.sub(r'([A-Za-z]{3,})(TAB|TABS|CAP|CAPS|GEL|OINT|OINTMENT|CREAM|LOTION|SOLUTION|SUSP|INJ|DROPS|SYRUP)\b', r'\1 \2', t, flags=re.IGNORECASE)
            # Fused sub-brand codes: BORITSB65 -> BORIT SB 65
            t = re.sub(r'\b([A-Za-z]+)(SB)(\d+)\b', r'\1 \2 \3', t, flags=re.IGNORECASE)
            t = re.sub(r'([A-Za-z]+)(\d+)\b', r'\1 \2', t)
            t = re.sub(r'\s+', ' ', t).strip()

        return t

    def _is_valid_num(self, val: str) -> bool:
        """Checks if string represents a valid float/integer/dash."""
        clean = val.replace(",", "").strip()
        if clean in ["-", "--", "---", "一", "—", "0", "0.0", "0.00"]:
            return True
        try:
            float(clean)
            return True
        except ValueError:
            return False

    def _normalize_val(self, val: str, expected_type: str) -> Any:
        """Normalizes extracted cell value according to type."""
        v = (val or "").strip()
        if expected_type in ["numeric", "decimal", "percentage"]:
            clean = v.replace(",", "").replace("$", "").replace("₹", "").replace("%", "").strip()
            if clean in ["-", "--", "---", "一", "—", "NA", "N/A", ""]:
                return None
            try:
                return float(clean)
            except ValueError:
                return v
        elif expected_type == "integer":
            clean = v.replace(",", "").strip()
            try:
                return int(float(clean))
            except (ValueError, TypeError):
                return v
        return v

    # =========================================================================
    # 7. Candidate Ranking Engine (Multi-Variant Agreement & 6 Signals)
    # =========================================================================
    def _bbox_xc_ratio(self, box: Any, img_width: int) -> Optional[float]:
        """
        x-center of an OCR result's bounding box as a fraction of the crop/variant
        image width (0.0 = left edge, 0.5 = centered, 1.0 = right edge). Used by
        rank_candidates as real geometric evidence: a candidate whose bbox sits
        near the horizontal center of its own cell crop is a clean, well-isolated
        read, while one pushed to an edge is more likely contaminated by bleed
        from a neighboring column.
        """
        try:
            if not box or img_width <= 0:
                return None
            xs = [p[0] for p in box]
            xc = (min(xs) + max(xs)) / 2.0
            return float(xc) / float(img_width)
        except Exception:
            return None

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        expected_type: str,
        quality: Dict[str, Any],
        neighbor_info: Optional[Dict[str, Any]] = None,
        col_name: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Ranks candidates using deterministic weighted 6-factor scoring:
        - OCR confidence (20%)
        - Numeric / type format validity (20%)
        - Candidate agreement across variants (20%)
        - Visual stroke quality (15%)
        - Bbox centering within its own cell crop (15%) - a real geometric
          measurement (see _bbox_xc_ratio), not a flat bonus: a candidate whose
          recognized text sits near the horizontal center of the cell crop scores
          higher than one pushed toward an edge, which is more likely bleed from
          a neighboring column rather than this cell's own content.
        - Neighbor consistency & length prior (10%)
        """
        if not candidates:
            return None

        is_num = expected_type in [
            CellType.INTEGER_QTY, CellType.DECIMAL_VALUE, CellType.RATE,
            CellType.AMOUNT, CellType.PERCENTAGE, "numeric", "decimal", "integer", "percentage"
        ]
        is_pack = expected_type in [CellType.PACKING, "packing"] or any(k in col_name.lower() for k in ["pack", "pkg", "unit"])
        is_sn = expected_type in [CellType.SERIAL, "serial"] or any(k in col_name.lower() for k in ["sn", "sl", "sr", "s.no"])

        # Count variant agreement across candidates
        text_counts: Dict[str, int] = {}
        for c in candidates:
            k = c["text"].strip().lower()
            text_counts[k] = text_counts.get(k, 0) + 1

        scored_cands = []
        for cand in candidates:
            txt = cand["text"].strip()
            conf = cand["confidence"]

            # 1. OCR confidence (0 to 0.20)
            c_score = conf * 0.20

            # 2. Numeric / Type format validity (0 to 0.20)
            if is_num:
                if re.match(r'^-?\d+(?:\.\d{1,3})?$', txt):
                    c_score += 0.20
                elif txt in ["-", "--", "一", "—", "0", "0.0", "0.00"]:
                    c_score += 0.18
                elif self._is_valid_num(txt):
                    c_score += 0.15
                elif any(c.isdigit() for c in txt):
                    c_score += 0.08
            elif is_pack:
                if PACKING_REGEX.search(txt) or re.match(r'^\d+\s*[\*xX]\s*\d+', txt) or re.match(r'^\d+[\.·]?[sS]$', txt):
                    c_score += 0.22
                elif any(k in txt.upper() for k in ["TAB", "CAP", "ML", "GM", "KG", "LTR", "VIAL", "AMP"]):
                    c_score += 0.18
                elif txt in ["-", "--"]:
                    c_score += 0.15
                else:
                    c_score += 0.08
            elif is_sn:
                if txt.isdigit() and 1 <= int(txt) <= 999:
                    c_score += 0.22
                elif any(c.isalpha() for c in txt):
                    c_score -= 0.10
                else:
                    c_score += 0.05
            else:
                # Text column: prefer alphabetic string
                if len(txt) >= 2:
                    c_score += 0.20
                else:
                    c_score += 0.08

            # 3. Candidate agreement across variants (0 to 0.20)
            cnt = text_counts.get(txt.lower(), 1)
            if cnt >= 3:
                c_score += 0.20
            elif cnt == 2:
                c_score += 0.15
            else:
                c_score += 0.05

            # 4. Visual stroke quality (0 to 0.15)
            if quality.get("contrast", 0.0) >= 45.0 and not quality.get("is_glare"):
                c_score += 0.15
            else:
                c_score += 0.08

            # 5. Bbox centering within its own cell crop (0 to 0.15) - real geometry,
            # not a flat bonus. Falls back to a neutral half-credit when no bbox is
            # available (e.g. a variant crop where per-token position wasn't captured).
            xc_ratio = cand.get("xc_ratio")
            if xc_ratio is None:
                c_score += 0.075
            else:
                c_score += 0.15 * max(0.0, 1.0 - abs(xc_ratio - 0.5) * 2.0)

            # 6. Neighbor consistency & length prior (0 to 0.10)
            if neighbor_info and is_num:
                if "." in txt and neighbor_info.get("has_decimals", False):
                    c_score += 0.05
                elif "." not in txt and not neighbor_info.get("has_decimals", False):
                    c_score += 0.05
            if is_num and 1 <= len(txt) <= 9:
                c_score += 0.05
            elif not is_num and 2 <= len(txt) <= 40:
                c_score += 0.05

            scored_cands.append({
                **cand,
                "score": round(c_score, 4)
            })

        scored_cands.sort(key=lambda c: c["score"], reverse=True)
        return scored_cands[0]

    # =========================================================================
    # 8. Level 3: Selective Qwen2.5-VL Cell Crop Repair
    # =========================================================================
    def targeted_qwen_cell_repair(
        self,
        image: np.ndarray,
        cell_bbox: Tuple[int, int, int, int],
        col_name: str,
        current_candidate: str
    ) -> Optional[str]:
        """
        Sends ONLY the problematic cell crop to Qwen2.5-VL with strict prompt:
        'Read only the visible text inside this exact cell. Return the value exactly as visible.
        Do not infer from formulas. Do not calculate missing values. Do not use neighboring values
        to invent text. If the cell is unreadable, return REVIEW.'
        Returns parsed value or None.
        """
        x0, y0, x1, y1 = [int(v) for v in cell_bbox]
        h_img, w_img = image.shape[:2]
        pad = 6
        crop = image[max(0, y0-pad):min(h_img, y1+pad), max(0, x0-pad):min(w_img, x1+pad)]

        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return None

        # Mild upscale for vision model clarity
        crop_large = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        temp_path = settings.OUTPUT_DIR / f"temp_qwen_cell_{int(time.time()*1000)%100000}.png"
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(temp_path), crop_large)

        prompt = (
            f"Read only the visible text inside this exact cell (Column: '{col_name}').\n"
            "Return the value exactly as visible.\n"
            "Do not infer from formulas. Do not calculate missing values. Do not use neighboring values to invent text.\n"
            "If the cell is unreadable, return REVIEW.\n"
            "Respond ONLY with a JSON object:\n"
            '{"value": "<exact visible text>"}'
        )

        try:
            resp = model_engine.predict(temp_path, prompt)
            if temp_path.exists():
                temp_path.unlink()

            # Parse JSON
            m = re.search(r'\{.*\}', resp, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                val = str(data.get("value", "")).strip()
                header_words = ["packing", "qty", "rate", "amount", "total", "subtotal", "balance", "opening", "closing", "issue", "receipt", "unit", "description"]
                # A crop that accidentally spans more than one row makes the model
                # read several cells and return them together ("10 TAB\n10TAB").
                # One cell holds one value, so a line break is proof the crop was
                # not a single cell - reject rather than write the concatenation
                # into it.
                if "\n" in val or "\r" in val:
                    logger.debug(
                        f"Qwen cell repair rejected for '{col_name}': multi-line response "
                        f"implies the crop spans more than one row ({val!r})"
                    )
                    return None
                if val and val.upper() != "REVIEW" and len(val) < 40:
                    if val.lower() != col_name.lower() and val.lower() not in header_words:
                        logger.info(f"Targeted Qwen cell repair for '{col_name}': '{current_candidate}' -> '{val}'")
                        return val
        except Exception as e:
            logger.debug(f"Targeted Qwen cell repair skipped: {e}")
            if temp_path.exists():
                temp_path.unlink()

        return None

    # =========================================================================
    # 9. Master 3-Level Cell OCR Cascade Interface
    # =========================================================================
    def extract_cell_value(
        self,
        image: np.ndarray,
        cell_bbox: Tuple[int, int, int, int],
        col_name: str,
        expected_type: str = "text",
        existing_candidate: Optional[str] = None,
        existing_conf: float = 0.0,
        neighbor_info: Optional[Dict[str, Any]] = None,
        row_arithmetic_mismatch: bool = False,
        allow_qwen: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the 3-level OCR cascade for a specific cell:
        LEVEL 1: Fast Accept (high confidence, properly formatted existing candidate)
        LEVEL 2: Small-Cell Upscale + Image Quality Tailored 5-Variant Re-OCR
        LEVEL 3: Selective Qwen2.5-VL Crop Repair
        """
        x0, y0, x1, y1 = [int(v) for v in cell_bbox]
        c_val = (existing_candidate or "").strip()
        cell_type = self.classify_cell_type(col_name, c_val)
        is_num = cell_type in [
            CellType.INTEGER_QTY, CellType.DECIMAL_VALUE, CellType.RATE,
            CellType.AMOUNT, CellType.PERCENTAGE
        ] or expected_type in ["numeric", "decimal", "integer", "percentage"]
        is_pack = (cell_type == CellType.PACKING) or expected_type == "packing" or any(k in col_name.lower() for k in ["pack", "pkg", "unit"])
        is_sn = (cell_type == CellType.SERIAL) or expected_type == "serial" or any(k in col_name.lower() for k in ["sn", "sl", "sr", "s.no"])

        # ---------------------------------------------------------------------
        # LEVEL 1: FAST ACCEPT (LESS AGGRESSIVE)
        # ---------------------------------------------------------------------
        suspicious_fast_accept = False
        if is_num and c_val:
            if not self._is_valid_num(c_val):
                suspicious_fast_accept = True
            clean_c = c_val.replace(",", "").strip()
            if len(clean_c) > 10 or ("." in clean_c and len(clean_c.split(".")[1]) > 3):
                suspicious_fast_accept = True
            if bool(re.search(r'[θbBsSlIOo一—]', c_val)):
                suspicious_fast_accept = True
            if bool(re.match(r'^000+\d+$', clean_c)):
                suspicious_fast_accept = True
            if row_arithmetic_mismatch:
                suspicious_fast_accept = True
            if neighbor_info and is_num:
                has_dec = "." in clean_c
                nbr_all_int = neighbor_info.get("all_integers", False)
                if nbr_all_int and has_dec and not clean_c.endswith(".00"):
                    suspicious_fast_accept = True
        elif is_pack and c_val:
            # Packaging strings with valid format fast-accept immediately
            if not (PACKING_REGEX.search(c_val) or re.match(r'^\d+\s*[\*xX]\s*\d+', c_val) or re.match(r'^\d+\.?S?$', c_val, re.IGNORECASE)):
                suspicious_fast_accept = True

        if c_val and existing_conf >= 0.88 and not suspicious_fast_accept and not row_arithmetic_mismatch:
            return {
                "raw_value": c_val,
                "normalized_value": self._normalize_val(c_val, expected_type),
                "confidence": existing_conf,
                "engine": "rapidocr_level1",
                "variants_used": ["level1_fast_accept"],
                "review_required": False
            }

        # Crop cell region
        h_img, w_img = image.shape[:2]
        pad = 4
        crop = image[max(0, y0-pad):min(h_img, y1+pad), max(0, x0-pad):min(w_img, x1+pad)]

        # ---------------------------------------------------------------------
        # LEVEL 2: MULTI-VARIANT RE-OCR & DASH DETECTION
        # ---------------------------------------------------------------------
        # Case A: Empty cell in numeric column -> check for printed dash '-' or blank preservation
        if not c_val and is_num and crop.size > 0:
            dash_val, sem, d_conf = self.detect_cell_dash_vs_empty(image, cell_bbox)
            if dash_val:
                return {
                    "raw_value": dash_val,
                    "normalized_value": "-",
                    "confidence": d_conf,
                    "engine": "level2_dash_detector",
                    "variants_used": ["morphology_dash"],
                    "review_required": False
                }

        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return {
                "raw_value": c_val,
                "normalized_value": self._normalize_val(c_val, expected_type),
                "confidence": existing_conf,
                "engine": "boundary_limit",
                "variants_used": [],
                "review_required": bool(c_val)
            }

        # Quality scoring
        quality = self.compute_cell_quality(crop)

        variants = self.generate_cell_variants(crop, quality)

        candidates: List[Dict[str, Any]] = []
        variants_used: List[str] = []
        if self.engine:
            # Stage 1: Fast variants (Original and CLAHE)
            stage1_variants = variants[:2]
            for v_name, v_img in stage1_variants:
                variants_used.append(v_name)
                try:
                    res, _ = self.engine(v_img)
                    if res:
                        for item in res:
                            txt = item[1].strip()
                            conf = float(item[2])
                            if txt:
                                txt_dis = self.disambiguate_numeric(txt, expected_type, col_name)
                                candidates.append({
                                    "raw": txt,
                                    "text": txt_dis,
                                    "confidence": conf,
                                    "variant": v_name,
                                    "xc_ratio": self._bbox_xc_ratio(item[0], v_img.shape[1])
                                })
                except Exception:
                    continue

            # Stage 2: Empty cell check after Stage 1
            if not c_val and not candidates:
                is_empty = self.is_cell_visually_empty(
                    crop, col_name, expected_type, candidates, neighbor_info
                )
                if is_empty:
                    return {
                        "raw_value": "",
                        "normalized_value": "",
                        "confidence": 1.0,
                        "engine": "blank_preservation",
                        "variants_used": variants_used,
                        "review_required": False
                    }

            # Stage 3: Deep variants (sharpened, otsu, upscaled) if needed
            best_conf_stage1 = max((c["confidence"] for c in candidates), default=0.0)
            if len(candidates) == 0 or best_conf_stage1 < 0.85:
                for v_name, v_img in variants[2:]:
                    variants_used.append(v_name)
                    try:
                        res, _ = self.engine(v_img)
                        if res:
                            for item in res:
                                txt = item[1].strip()
                                conf = float(item[2])
                                if txt:
                                    txt_dis = self.disambiguate_numeric(txt, expected_type, col_name)
                                    candidates.append({
                                        "raw": txt,
                                        "text": txt_dis,
                                        "confidence": conf,
                                        "variant": v_name,
                                        "xc_ratio": self._bbox_xc_ratio(item[0], v_img.shape[1])
                                    })
                    except Exception:
                        continue

        # If originally empty cell: verify visual presence with all candidate evidence
        if not c_val:
            is_empty = self.is_cell_visually_empty(
                crop, col_name, expected_type, candidates, neighbor_info
            )
            if is_empty:
                return {
                    "raw_value": "",
                    "normalized_value": "",
                    "confidence": 1.0,
                    "engine": "blank_preservation",
                    "variants_used": variants_used,
                    "review_required": False
                }

        best_cand = self.rank_candidates(candidates, expected_type, quality, neighbor_info, col_name)

        # For originally empty cells, require higher confidence / variant agreement before accepting
        min_accept_score = 0.70 if not c_val else 0.65
        if best_cand and best_cand["score"] >= min_accept_score:
            return {
                "raw_value": best_cand["text"],
                "normalized_value": self._normalize_val(best_cand["text"], expected_type),
                "confidence": best_cand["confidence"],
                "engine": f"level2_{best_cand['variant']}",
                "variants_used": [c["variant"] for c in candidates],
                "review_required": best_cand["score"] < 0.80
            }

        # ---------------------------------------------------------------------
        # LEVEL 3: SELECTIVE QWEN CELL REPAIR
        # ---------------------------------------------------------------------
        qwen_attempted = False
        # Qwen should ONLY be called on cells that have visual ink or a corrupted candidate
        # NEVER call Qwen on an empty cell where candidates is empty!
        has_visual_content = bool(c_val) or (candidates and any(c["confidence"] >= 0.40 for c in candidates))
        if allow_qwen and has_visual_content and (row_arithmetic_mismatch or (best_cand and best_cand["score"] < 0.65) or not c_val):
            # Check ink stroke presence
            if not self.is_cell_visually_empty(crop, col_name, expected_type, candidates, neighbor_info):
                qwen_attempted = True
                qwen_val = self.targeted_qwen_cell_repair(image, cell_bbox, col_name, c_val)
                if qwen_val:
                    qwen_clean = self.disambiguate_numeric(qwen_val, expected_type, col_name)
                    # For numeric column, strictly enforce numeric or dash formatting
                    if not is_num or self._is_valid_num(qwen_clean) or qwen_clean == "-":
                        return {
                            "raw_value": qwen_clean,
                            "normalized_value": self._normalize_val(qwen_clean, expected_type),
                            "confidence": 0.95,
                            "engine": "level3_qwen",
                            "variants_used": ["qwen2.5vl_crop"],
                            "review_required": False
                        }

        # Fallback strictly to original candidate if no candidate met the 0.65 score threshold
        fallback_val = c_val
        fallback_conf = existing_conf
        all_variants = [c["variant"] for c in candidates] if candidates else []
        if qwen_attempted:
            all_variants.append("qwen2.5vl_crop")
        return {
            "raw_value": fallback_val,
            "normalized_value": self._normalize_val(fallback_val, expected_type),
            "confidence": fallback_conf,
            "engine": "level2_fallback",
            "variants_used": all_variants,
            "review_required": bool(c_val)
        }

    # =========================================================================
    # 10. Table Grid Refinement Orchestrator
    # =========================================================================
    def refine_table_cells(
        self,
        image: np.ndarray,
        col_bounds: List[Tuple[str, float, float]],
        logical_row_bboxes: List[Tuple[float, float]],
        grid_rows: List[List[str]],
        grid_meta: List[List[Dict[str, Any]]],
        doc_name: str = "doc"
    ) -> Tuple[List[List[str]], List[List[Dict[str, Any]]]]:
        """
        Orchestrates full cell cascade refinement across all rows and columns:
        - Fused packing extraction from descriptions
        - Specialized narrow column cell extraction (Sn., Unit, Balance)
        - Printed dash vs empty detection
        - Small-cell upscaling & multi-variant candidate ranking
        """
        if not grid_rows or not col_bounds or image is None:
            return grid_rows, grid_meta

        num_cols = len(col_bounds)
        qwen_calls_count = 0

        # Adaptive Qwen budget calculation (Task 8)
        # 0-5 suspicious cells: repair all
        # 6-20 suspicious cells: repair top high-risk cells (budget = 6)
        # >20 suspicious cells: systemic failure, budget = 0, mark review required
        suspicious_cells = []
        for r_idx, row in enumerate(grid_rows):
            for c_idx in range(min(num_cols, len(row))):
                c_val = row[c_idx].strip()
                c_name = col_bounds[c_idx][0]
                cell_type = self.classify_cell_type(c_name, c_val)
                is_num_col = cell_type in [
                    CellType.INTEGER_QTY, CellType.DECIMAL_VALUE,
                    CellType.RATE, CellType.AMOUNT, CellType.PERCENTAGE
                ]
                if is_num_col and c_val:
                    if not self._is_valid_num(c_val) or bool(re.search(r'[θbBsSlI]', c_val)):
                        suspicious_cells.append((r_idx, c_idx))

        num_suspicious = len(suspicious_cells)
        if num_suspicious <= 5:
            max_qwen_per_doc = num_suspicious
        elif num_suspicious <= 20:
            max_qwen_per_doc = 6
        else:
            max_qwen_per_doc = 0  # >20 suspicious cells: mark review required

        # Locate key column indices
        desc_idx = next((i for i, c in enumerate(col_bounds) if any(k in c[0].lower() for k in ["desc", "product", "item"])), None)
        pack_idx = next((i for i, c in enumerate(col_bounds) if any(k in c[0].lower() for k in ["pack", "unit"])), None)
        sn_idx = next((i for i, c in enumerate(col_bounds) if any(k in c[0].lower() for k in ["sn", "sl", "sr"])), None)

        for r_idx, row in enumerate(grid_rows):
            if r_idx < len(logical_row_bboxes):
                r_y0, r_y1 = logical_row_bboxes[r_idx]
            else:
                continue

            # Count data cells filled outside description
            row_data_count = sum(1 for c_i, v in enumerate(row) if v.strip() and c_i != desc_idx)

            # Pass 2: Cell-by-Cell Cascade Refinement
            for c_idx in range(min(num_cols, len(row))):
                c_name, c_x0, c_x1 = col_bounds[c_idx]
                c_val = row[c_idx].strip()
                c_meta = grid_meta[r_idx][c_idx] if r_idx < len(grid_meta) and c_idx < len(grid_meta[r_idx]) else {}

                # Cell crops use the row's full axis-aligned band. Shearing each
                # crop along the page's tilt (so a cell is cut from where its own
                # column actually sits) was implemented and measured WORSE on the
                # only tilted document - 86.7% -> 81.0% tight, 83.8% with a 4px
                # margin - while leaving every untilted document untouched. The
                # geometry was verified correct against real token coordinates,
                # so the loss is not misplacement: correcting the band also
                # tightens it onto the glyphs, and the cell cascade evidently
                # depends on the generous vertical margin the uncorrected (and
                # tilt-inflated) band happens to provide. Left uncorrected until
                # the cascade's sensitivity to crop margin is addressed.
                cy0, cy1 = r_y0, r_y1
                cell_bbox = (int(c_x0), int(cy0), int(c_x1), int(cy1))
                cell_type = self.classify_cell_type(c_name, c_val)
                is_num_col = cell_type in [
                    CellType.INTEGER_QTY, CellType.DECIMAL_VALUE,
                    CellType.RATE, CellType.AMOUNT, CellType.PERCENTAGE
                ]
                is_pack_col = (cell_type == CellType.PACKING)
                is_sn_col = (cell_type == CellType.SERIAL or c_idx == sn_idx)
                is_unit_col = (cell_type == CellType.UNIT)

                exp_type = cell_type

                # Narrow Sn., Unit, or Packing column recovery: if empty in active transaction row, crop cell and read
                is_subtotal_or_total_row = any(k in str(v).lower() for v in row for k in ["s-total", "subtotal", "sub-total", "total", "grand total"])
                is_total_row = any(k in str(v).lower() for v in row for k in ["total:", "grand total", "net total"])

                # Product row Unit / Serial guard: on a product row, Unit or Serial can NEVER be 'S-Total' or 'Total'
                if (is_unit_col or is_pack_col or is_sn_col) and not is_subtotal_or_total_row:
                    if c_val.lower() in ["s-total", "subtotal", "sub-total", "total"]:
                        c_val = ""
                        row[c_idx] = ""
                        if c_meta:
                            c_meta["raw"] = ""
                            c_meta["status"] = "EMPTY"

                if (is_sn_col or is_unit_col or is_pack_col) and not c_val and (r_y1 - r_y0) > 8 and row_data_count > 0:
                    if (is_sn_col and is_subtotal_or_total_row) or (is_total_row and (is_unit_col or is_pack_col)):
                        pass
                    else:
                        narrow_exp = CellType.SERIAL if is_sn_col else (CellType.UNIT if is_unit_col else CellType.PACKING)
                        # Focus cell_bbox tightly on column interval so it doesn't bleed into Description
                        cell_bbox_focused = cell_bbox
                        if is_sn_col and desc_idx is not None:
                            desc_x0 = col_bounds[desc_idx][1]
                            sn_x0 = max(c_x0, c_x1 - 50.0)
                            sn_x1 = min(c_x1, desc_x0 - 5.0)
                            if sn_x1 > sn_x0:
                                cell_bbox_focused = (int(sn_x0), int(cy0), int(sn_x1), int(cy1))

                        narrow_res = self.extract_cell_value(
                            image=image,
                            cell_bbox=cell_bbox_focused,
                            col_name=c_name,
                            expected_type=narrow_exp,
                            existing_candidate=c_val,
                            existing_conf=0.0,
                            allow_qwen=False
                        )
                        raw_ext = narrow_res.get("raw_value", "").strip()
                        # Reject 'S-Total', 'ST' on product rows
                        if not is_subtotal_or_total_row and raw_ext.lower() in ["s-total", "subtotal", "sub-total", "total", "st", "s.t", "s-t", "s"]:
                            raw_ext = ""

                        if raw_ext:
                            row[c_idx] = raw_ext
                            c_val = raw_ext
                            if c_meta:
                                c_meta["raw"] = c_val
                                c_meta["repaired"] = True
                                c_meta["status"] = "VALUE"

                # Unit / Packing on product rows: a bare "1" is very likely a narrow-column
                # OCR truncation of a real token like "10TAB"/"1BOT", and a blank cell may
                # simply not have been read yet. Neither case gives license to guess a
                # specific packing value from the product DESCRIPTION text or from other
                # rows in the grid - that is fabrication, not OCR evidence, and a
                # description mentioning "ml" or "tab" says nothing about what THIS cell's
                # own packing token visually says. Attempt a targeted re-OCR of the cell
                # itself first; only accept a value the re-OCR actually reads and that is
                # packing-format-valid. Otherwise resolve to EMPTY/UNKNOWN.
                if (is_unit_col or is_pack_col) and not is_subtotal_or_total_row and (c_val == "1" or not c_val):
                    reocr_allow_qwen = qwen_calls_count < max_qwen_per_doc
                    reocr_res = self.extract_cell_value(
                        image=image,
                        cell_bbox=cell_bbox,
                        col_name=c_name,
                        expected_type=CellType.PACKING,
                        existing_candidate="",
                        existing_conf=0.0,
                        allow_qwen=reocr_allow_qwen
                    )
                    if "qwen2.5vl_crop" in reocr_res.get("variants_used", []) or reocr_res.get("engine") == "level3_qwen":
                        qwen_calls_count += 1
                    reocr_val = reocr_res.get("raw_value", "").strip()
                    if reocr_val and reocr_val != c_val and PACKING_REGEX.search(reocr_val):
                        row[c_idx] = reocr_val
                        c_val = reocr_val
                        if c_meta:
                            c_meta["raw"] = reocr_val
                            c_meta["repaired"] = True
                            c_meta["repair_source"] = "narrow_column_reocr"
                            c_meta["status"] = "VALUE"
                    elif not c_val and c_meta:
                        # No independent visual evidence for a specific value - let the
                        # visual-ink check decide EMPTY vs UNKNOWN, never a guessed value.
                        pad = 3
                        crop = image[max(0, int(cy0)-pad):min(image.shape[0], int(cy1)+pad), max(0, int(c_x0)-pad):min(image.shape[1], int(c_x1)+pad)]
                        c_meta["status"] = "EMPTY" if self.is_cell_visually_empty(crop, c_name, CellType.PACKING, [], row_type="product") else "UNKNOWN"
                    elif c_val == "1" and c_meta:
                        # Raw OCR result preserved as-is (possibly a truncated read) -
                        # flagged for review rather than overwritten with a guess.
                        c_meta["status"] = "UNKNOWN"

                # Disambiguate packing: e.g. 10.9 or 10:8 -> 10.S in packing column
                if is_pack_col and c_val:
                    c_val_fixed = self.disambiguate_numeric(c_val, CellType.PACKING, c_name)
                    if c_val_fixed != c_val:
                        row[c_idx] = c_val_fixed
                        c_val = c_val_fixed
                        if c_meta:
                            c_meta["raw"] = c_val_fixed
                            c_meta["repaired"] = True

                # Disambiguate description: e.g. TOFUSTAB -> TOFUS TAB, BORITSB65 -> BORIT SB 65
                is_desc_col = (c_idx == desc_idx)
                if is_desc_col and c_val:
                    c_val_fixed = self.disambiguate_numeric(c_val, CellType.DESCRIPTION, c_name)
                    if c_val_fixed != c_val:
                        row[c_idx] = c_val_fixed
                        c_val = c_val_fixed
                        if c_meta:
                            c_meta["raw"] = c_val_fixed
                            c_meta["repaired"] = True

                # Trigger cascade on suspicious chars, low confidence, or corrupted numbers
                toks = c_meta.get("tokens", [])
                min_conf = min((t.get("confidence", 1.0) for t in toks), default=1.0)

                # For originally empty cells: check printed dash vs ink presence vs genuine empty
                if not c_val:
                    # On subtotal lines, empty numeric/transaction columns represent zero ("0.000")
                    is_subtotal_row = any(k in str(v).lower() for v in row for k in ["s-total", "subtotal", "sub-total"]) and not is_total_row
                    if is_subtotal_row and is_num_col:
                        row[c_idx] = "0.000"
                        if c_meta:
                            c_meta["raw"] = "0.000"
                            c_meta["normalized"] = 0.0
                            c_meta["semantic"] = "ZERO"
                            c_meta["status"] = "VALUE"
                        continue

                    if is_num_col and (r_y1 - r_y0) >= 6 and row_data_count > 0:
                        dash_val, sem, d_conf = self.detect_cell_dash_vs_empty(image, cell_bbox)
                        if dash_val:
                            row[c_idx] = dash_val
                            if c_meta:
                                c_meta["raw"] = dash_val
                                c_meta["repaired"] = True
                                c_meta["status"] = "DASH"
                        elif not is_subtotal_or_total_row:
                            # If not a dash, check if meaningful ink is present (small digit or dot-matrix zero)
                            pad = 3
                            crop = image[max(0, int(cy0)-pad):min(image.shape[0], int(cy1)+pad), max(0, int(c_x0)-pad):min(image.shape[1], int(c_x1)+pad)]
                            if not self.is_cell_visually_empty(crop, c_name, exp_type, [], row_type="product"):
                                res = self.extract_cell_value(
                                    image=image,
                                    cell_bbox=cell_bbox,
                                    col_name=c_name,
                                    expected_type=exp_type,
                                    existing_candidate="",
                                    existing_conf=0.0,
                                    allow_qwen=False
                                )
                                if res.get("raw_value"):
                                    row[c_idx] = res["raw_value"]
                                    c_val = res["raw_value"]
                                    if c_meta:
                                        c_meta["raw"] = res["raw_value"]
                                        c_meta["repaired"] = True
                                        c_meta["status"] = "VALUE"
                                        c_meta["engine"] = res.get("engine", "micro_cell")
                                else:
                                    if c_meta:
                                        c_meta["status"] = "UNKNOWN"
                    continue

                needs_repair = (
                    (is_num_col and min_conf < 0.80) or
                    (is_num_col and not self._is_valid_num(c_val)) or
                    (is_num_col and bool(re.search(r'[θoOlIsSbB]', c_val))) or
                    (is_num_col and bool(re.match(r'^000\d+$', c_val))) or
                    (is_pack_col and min_conf < 0.70 and not PACKING_REGEX.search(c_val)) or
                    (is_sn_col and min_conf < 0.70 and not c_val.isdigit())
                )

                if needs_repair:
                    # Never call Qwen on genuinely empty cells without candidates (Task 8)
                    allow_qwen = bool(c_val) and (qwen_calls_count < max_qwen_per_doc)
                    res = self.extract_cell_value(
                        image=image,
                        cell_bbox=cell_bbox,
                        col_name=c_name,
                        expected_type=exp_type,
                        existing_candidate=c_val,
                        existing_conf=min_conf,
                        allow_qwen=allow_qwen
                    )
                    if "qwen2.5vl_crop" in res.get("variants_used", []) or res.get("engine") == "level3_qwen":
                        qwen_calls_count += 1

                    if res.get("raw_value") and res["raw_value"] != c_val:
                        row[c_idx] = res["raw_value"]
                        if c_meta:
                            c_meta["raw"] = res["raw_value"]
                            c_meta["repaired"] = True
                            c_meta["engine"] = res["engine"]

                # A lone "1" in a numeric column is geometrically ambiguous with a
                # short printed dash - both are a single narrow stroke, and re-OCR
                # (needs_repair above) only retries the same text-recognition path
                # that produced "1" in the first place, so it can't self-correct
                # this. Checked after repair (not before) since repair itself is
                # often what settles a cell on "1". The stroke's own shape
                # resolves it: a genuine "1" is tall and narrow, a dash is short
                # and wide - the same connected-component geometry already used
                # to detect dashes on empty cells (detect_cell_dash_vs_empty)
                # applies here too.
                if is_num_col and row[c_idx].strip() == "1" and (r_y1 - r_y0) >= 6:
                    dash_val, _dash_sem, _dash_conf = self.detect_cell_dash_vs_empty(image, cell_bbox)
                    if dash_val == "-":
                        row[c_idx] = "-"
                        if c_meta:
                            c_meta["raw"] = "-"
                            c_meta["repaired"] = True
                            c_meta["repair_source"] = "dash_vs_one_geometry"
                            c_meta["semantic"] = "DASH"

                # A single-stroke glyph can also come back as a CJK/lookalike
                # character (e.g. "一", the Chinese numeral "one" - itself
                # literally a single horizontal stroke) rather than "1" or "-".
                # _is_valid_num/_normalize_val already treat these as dash-
                # equivalent for validation, but the raw cell value was never
                # canonicalized to "-" for output, so it still fails an exact
                # comparison against ground truth despite being semantically
                # identical.
                elif is_num_col and row[c_idx].strip() in ("一", "—", "--", "---"):
                    row[c_idx] = "-"
                    if c_meta:
                        c_meta["raw"] = "-"
                        c_meta["repaired"] = True
                        c_meta["repair_source"] = "dash_glyph_canonicalized"
                        c_meta["semantic"] = "DASH"

            # Fallback: Fused Packing Extraction if Packing/Unit is still empty after visual cell cascade.
            # This relocates a packing token that is genuinely present in THIS row's own
            # description text (the column model glued it onto the description column) -
            # evidence-preserving, not fabrication. It must never be replaced with a
            # different, hardcoded literal (e.g. collapsing any "<N> ML" match into a fixed
            # "1BOT") - that would discard the actual extracted characters in favor of a guess.
            if desc_idx is not None and pack_idx is not None:
                cur_desc = row[desc_idx].strip()
                cur_pack = row[pack_idx].strip()
                if cur_desc and not cur_pack:
                    clean_d, ext_p = self.extract_fused_packing(cur_desc)
                    if ext_p:
                        row[pack_idx] = ext_p
                        if grid_meta and r_idx < len(grid_meta) and pack_idx < len(grid_meta[r_idx]):
                            grid_meta[r_idx][pack_idx]["raw"] = ext_p
                            grid_meta[r_idx][pack_idx]["repaired"] = True
                            grid_meta[r_idx][pack_idx]["repair_source"] = "split_from_description_same_row"

        # Pass 3: Reconcile single-transaction product rows with their subtotal rows
        num_indices = [
            i for i, (c_name, _, _) in enumerate(col_bounds)
            if self.classify_cell_type(c_name) in [
                CellType.INTEGER_QTY, CellType.DECIMAL_VALUE, CellType.RATE, CellType.AMOUNT
            ]
        ]
        if desc_idx is not None and len(num_indices) >= 2:
            for r_idx in range(len(grid_rows) - 1):
                cur_r = grid_rows[r_idx]
                next_r = grid_rows[r_idx + 1]
                cur_is_sub = any(k in str(v).lower() for v in cur_r for k in ["s-total", "subtotal", "sub-total"])
                next_is_sub = any(k in str(v).lower() for v in next_r for k in ["s-total", "subtotal", "sub-total"])
                if not cur_is_sub and next_is_sub:
                    cur_desc = re.sub(r'[^a-zA-Z0-9]', '', cur_r[desc_idx]).lower()
                    next_desc = re.sub(r'[^a-zA-Z0-9]', '', next_r[desc_idx]).lower()
                    # Containment (not the stricter length-ratio similarity used for
                    # general row-merging elsewhere) is appropriate here specifically
                    # because next_is_sub already confirms next_r is a subtotal-marker
                    # row, and this domain's convention is the subtotal line repeating
                    # the exact product name plus a suffix ("DRUG X" -> "DRUG X S-Total"),
                    # which a symmetric-similarity check would wrongly penalize for length.
                    desc_overlap = bool(cur_desc and next_desc and (cur_desc in next_desc or next_desc in cur_desc))
                    # A subtotal row commonly carries no repeated product description at
                    # all - that alone is not evidence it belongs to THIS product row's
                    # numbers (an intervening skipped row could just as easily produce the
                    # same blank). Require genuine numeric complementarity (one side filled,
                    # the other blank, for at least one transaction column) before
                    # reconciling on empty-description grounds, instead of merging
                    # unconditionally whenever the subtotal's description happens to be blank.
                    has_complementary_numeric = any(
                        bool(cur_r[ni].strip()) != bool(next_r[ni].strip()) for ni in num_indices
                    )
                    if desc_overlap or (not next_desc and has_complementary_numeric):
                        for ni in num_indices:
                            if not cur_r[ni].strip() and next_r[ni].strip():
                                cur_r[ni] = next_r[ni]
                                if grid_meta and r_idx < len(grid_meta) and ni < len(grid_meta[r_idx]):
                                    grid_meta[r_idx][ni] = grid_meta[r_idx + 1][ni]
                            elif not next_r[ni].strip() and cur_r[ni].strip():
                                next_r[ni] = cur_r[ni]
                                if grid_meta and r_idx + 1 < len(grid_meta) and ni < len(grid_meta[r_idx + 1]):
                                    grid_meta[r_idx + 1][ni] = grid_meta[r_idx][ni]

                        # Arithmetic check on Opening + In - Out = Balance
                        if len(num_indices) >= 4:
                            op_i, in_i, out_i, bal_i = num_indices[:4]
                            try:
                                op_v = float(cur_r[op_i].replace(",", "")) if cur_r[op_i].strip() else None
                                in_v = float(cur_r[in_i].replace(",", "")) if cur_r[in_i].strip() else None
                                out_v = float(cur_r[out_i].replace(",", "")) if cur_r[out_i].strip() else None
                                bal_v = float(cur_r[bal_i].replace(",", "")) if cur_r[bal_i].strip() else None

                                # If subtotal Out satisfies arithmetic but product Out has smudged digits,
                                # this is a CONFLICT signal, not license to overwrite the cell with the
                                # neighbor's string. Trigger a targeted re-OCR of THIS row's own Out cell
                                # and only accept a repair when that independent evidence itself supports
                                # the subtotal-implied value; otherwise keep the original raw read and
                                # flag it for review.
                                next_out = float(next_r[out_i].replace(",", "")) if next_r[out_i].strip() else None
                                if op_v is not None and in_v is not None and bal_v is not None and next_out is not None:
                                    if abs(op_v + in_v - next_out - bal_v) <= 0.05 and (out_v is None or abs(op_v + in_v - out_v - bal_v) > 0.05):
                                        reocr_confirmed = False
                                        if r_idx < len(logical_row_bboxes) and out_i < len(col_bounds):
                                            out_c_name, out_x0, out_x1 = col_bounds[out_i]
                                            out_ry0, out_ry1 = logical_row_bboxes[r_idx]
                                            out_bbox = (int(out_x0), int(out_ry0), int(out_x1), int(out_ry1))
                                            reocr = self.extract_cell_value(
                                                image=image,
                                                cell_bbox=out_bbox,
                                                col_name=out_c_name,
                                                expected_type=CellType.DECIMAL_VALUE,
                                                existing_candidate=cur_r[out_i].strip(),
                                                existing_conf=0.0,
                                                allow_qwen=False
                                            )
                                            reocr_val = reocr.get("raw_value", "").strip()
                                            try:
                                                reocr_num = float(reocr_val.replace(",", "")) if reocr_val else None
                                            except ValueError:
                                                reocr_num = None
                                            if reocr_num is not None and abs(reocr_num - next_out) <= 0.05:
                                                cur_r[out_i] = reocr_val
                                                out_v = reocr_num
                                                reocr_confirmed = True
                                                if grid_meta and r_idx < len(grid_meta) and out_i < len(grid_meta[r_idx]):
                                                    grid_meta[r_idx][out_i]["raw"] = reocr_val
                                                    grid_meta[r_idx][out_i]["repaired"] = True
                                                    grid_meta[r_idx][out_i]["repair_source"] = "arithmetic_conflict_reocr_confirmed"
                                        if not reocr_confirmed and grid_meta and r_idx < len(grid_meta) and out_i < len(grid_meta[r_idx]):
                                            grid_meta[r_idx][out_i]["arithmetic_status"] = "CONFLICT"
                                            grid_meta[r_idx][out_i]["needs_review"] = True

                                # Opening=0, Out=0, Balance>0 implying In=Balance is a pure computation
                                # with zero visual evidence for "In" - it must not be written as a value.
                                # Flag both rows' In cell for review instead of fabricating a number.
                                if op_v == 0.0 and out_v == 0.0 and bal_v is not None and bal_v > 0.0 and (in_v == 0.0 or in_v is None):
                                    for target_r, target_ridx in ((cur_r, r_idx), (next_r, r_idx + 1)):
                                        if not target_r[in_i].strip() and grid_meta and target_ridx < len(grid_meta) and in_i < len(grid_meta[target_ridx]):
                                            grid_meta[target_ridx][in_i]["status"] = "UNKNOWN"
                                            grid_meta[target_ridx][in_i]["arithmetic_status"] = "SUSPICIOUS"
                                            grid_meta[target_ridx][in_i]["needs_review"] = True
                            except Exception:
                                pass

        # Pass 4: Row-level arithmetic reconciliation
        self._reconcile_row_arithmetic(
            image, col_bounds, logical_row_bboxes, grid_rows, grid_meta
        )

        return grid_rows, grid_meta

    # =========================================================================
    # 10b. Row-level arithmetic reconciliation
    # =========================================================================
    STOCK_ROLE_KEYWORDS = {
        "opening": ("opening", "op.bal", "opbal", "op bal", "o.bal"),
        "inflow": ("receipt", "purchase", "inward", "in"),
        "outflow": ("issue", "sale", "dispatch", "outward", "out"),
        "closing": ("closing", "balance", "bal", "clos"),
    }

    def _map_stock_roles(self, col_bounds: List[Tuple[str, float, float]]) -> Dict[str, int]:
        """
        Locates the four stock-movement columns by header text. Value/amount
        columns are excluded - only quantity columns participate in the
        movement equation.
        """
        roles: Dict[str, int] = {}
        for idx, (name, _x0, _x1) in enumerate(col_bounds):
            n = (name or "").strip().lower()
            if not n or any(k in n for k in ("amount", "amt", "value", "val", "rate")):
                continue
            for role, keys in self.STOCK_ROLE_KEYWORDS.items():
                if role in roles:
                    continue
                if any(n == k or n.startswith(k) or k in n.split() or k in n for k in keys):
                    roles[role] = idx
                    break
        return roles

    def _reconcile_row_arithmetic(
        self,
        image: np.ndarray,
        col_bounds: List[Tuple[str, float, float]],
        logical_row_bboxes: List[Tuple[float, float]],
        grid_rows: List[List[str]],
        grid_meta: List[List[Dict[str, Any]]]
    ) -> None:
        """
        Uses the table's own arithmetic to find and repair misread numbers.

        A stock row must satisfy Closing = Opening + In - Out. When it doesn't,
        one of those four cells was misread, and the equation states exactly what
        the offending cell would have to be. That is a far more specific signal
        than confidence alone - but it is NOT evidence, so the implied number is
        never written directly (that would be fabricating a value no one read).
        Instead it is used to aim a re-OCR: the cell is read again, and the new
        reading is accepted ONLY if it independently agrees with what the
        arithmetic requires. Two independent sources agreeing is evidence; either
        one alone is not.

        When nothing confirms, the original raw values are left untouched and the
        row is flagged, so an unreconciled row surfaces for review rather than
        being silently "corrected".
        """
        if image is None or not grid_rows or not col_bounds:
            return

        roles = self._map_stock_roles(col_bounds)
        if len({"opening", "inflow", "outflow", "closing"} & set(roles)) < 4:
            return
        op_i, in_i, out_i, bal_i = (
            roles["opening"], roles["inflow"], roles["outflow"], roles["closing"]
        )

        def as_num(v: str) -> Optional[float]:
            s = (v or "").strip().replace(",", "")
            if not s:
                return None
            if s in ("-", "--", "---"):
                return 0.0
            try:
                return float(s)
            except ValueError:
                return None

        # Capped so a badly-read page cannot turn into an unbounded number of
        # model calls. Held in a list so the nested helper can decrement it.
        qwen_budget = [8]

        for r_idx, row in enumerate(grid_rows):
            if r_idx >= len(logical_row_bboxes) or max(op_i, in_i, out_i, bal_i) >= len(row):
                continue
            # Subtotal/total rows aggregate other rows and are handled elsewhere.
            if any(k in " ".join(row).lower() for k in ("s-total", "subtotal", "sub-total", "total")):
                continue

            vals = {i: as_num(row[i]) for i in (op_i, in_i, out_i, bal_i)}
            if any(v is None for v in vals.values()):
                continue
            op_v, in_v, out_v, bal_v = vals[op_i], vals[in_i], vals[out_i], vals[bal_i]
            if op_v == in_v == out_v == bal_v == 0.0:
                continue
            if abs((op_v + in_v - out_v) - bal_v) <= 0.05:
                continue  # already consistent

            # What each cell would have to read for the row to balance.
            implied = {
                op_i: bal_v - in_v + out_v,
                in_i: bal_v - op_v + out_v,
                out_i: op_v + in_v - bal_v,
                bal_i: op_v + in_v - out_v,
            }

            # Try the cells most likely to be misread first: a value whose text
            # carries digit-confusable characters or a malformed number is a
            # better suspect than a clean read.
            def suspicion(i: int) -> float:
                raw = (row[i] or "").strip()
                meta = grid_meta[r_idx][i] if (r_idx < len(grid_meta) and i < len(grid_meta[r_idx])) else {}
                score = 0.0
                if not self._is_valid_num(raw):
                    score += 3.0
                if re.search(r"[θoOlIsSbB]", raw):
                    score += 2.0
                if re.match(r"^0{2,}\d+$", raw.replace(".", "")):
                    score += 2.0
                toks = meta.get("tokens") or []
                if toks:
                    score += (1.0 - min(t.get("confidence", 1.0) for t in toks)) * 2.0
                if meta.get("repaired"):
                    score += 0.5
                return score

            # DETECTION ONLY - auto-repair was implemented here and removed after
            # it made things worse. Two reasons, both fundamental:
            #
            # 1. A failed equation says the ROW is wrong, not WHICH cell is wrong.
            #    Any of the four could be the culprit, and "solving" for the wrong
            #    one produces a row that balances with TWO wrong values instead of
            #    one. Observed: a row whose Balance was misread (21 for 30) was
            #    "repaired" by changing its Out from 0 to 9.
            # 2. The confirming re-read is not independent. In this domain the
            #    subtotal row immediately below repeats the product row's own
            #    figures, so a crop that bleeds even slightly into it will confirm
            #    exactly the value the arithmetic predicted - manufacturing
            #    agreement between two sources that are really one.
            #
            # The equation is still an excellent ERROR DETECTOR, and that is kept:
            # an unbalanced row is flagged for review, values untouched. That is
            # what the review workflow needs, and it cannot corrupt data.
            candidates: List[int] = []

            def attempt(c_i: int, use_qwen: bool) -> bool:
                """Re-read one cell and accept only if it independently agrees with the equation."""
                target = implied[c_i]
                c_name, c_x0, c_x1 = col_bounds[c_i]
                r_y0, r_y1 = logical_row_bboxes[r_idx]
                bbox = (int(c_x0), int(r_y0), int(c_x1), int(r_y1))
                current = (row[c_i] or "").strip()

                if use_qwen:
                    # Asked directly rather than through extract_cell_value: that
                    # cascade accepts its own Level-2 candidate at score >= 0.65
                    # and returns before ever reaching the model - which here just
                    # re-confirms the very value the arithmetic says is wrong.
                    raw_val = self.targeted_qwen_cell_repair(image, bbox, c_name, current)
                    new_val = self.disambiguate_numeric(raw_val, CellType.DECIMAL_VALUE, c_name) if raw_val else ""
                else:
                    res = self.extract_cell_value(
                        image=image,
                        cell_bbox=bbox,
                        col_name=c_name,
                        expected_type=CellType.DECIMAL_VALUE,
                        existing_candidate=current,
                        existing_conf=0.0,
                        row_arithmetic_mismatch=True,
                        allow_qwen=False
                    )
                    new_val = (res.get("raw_value") or "").strip()

                confirmed = as_num(new_val)
                if confirmed is None or abs(confirmed - target) > 0.05:
                    return False
                row[c_i] = new_val.strip()
                if r_idx < len(grid_meta) and c_i < len(grid_meta[r_idx]):
                    meta = grid_meta[r_idx][c_i]
                    meta["raw"] = row[c_i]
                    meta["repaired"] = True
                    meta["repair_source"] = (
                        "row_arithmetic_qwen_confirmed" if use_qwen else "row_arithmetic_reocr_confirmed"
                    )
                    meta["arithmetic_status"] = "RECONCILED"
                    meta["status"] = "VALUE"
                logger.info(
                    f"Row {r_idx} arithmetic reconciled via '{c_name}': independent re-read "
                    f"confirms {row[c_i]!r} (equation implied {target:g}, qwen={use_qwen})"
                )
                return True

            # A cheap re-read first, across every cell the equation could blame.
            repaired = any(attempt(c_i, False) for c_i in candidates)

            # The cheap pass re-runs the same recognizer that produced the wrong
            # value, so it often just repeats it. Escalating to the vision model
            # is safe here precisely BECAUSE the answer is checked against the
            # arithmetic: a hallucinated number that doesn't satisfy the equation
            # is rejected, so the model can only help, never invent.
            if not repaired:
                for c_i in candidates[:2]:
                    if qwen_budget[0] <= 0:
                        break
                    qwen_budget[0] -= 1
                    if attempt(c_i, True):
                        repaired = True
                        break

            if not repaired:
                for c_i in (op_i, in_i, out_i, bal_i):
                    if r_idx < len(grid_meta) and c_i < len(grid_meta[r_idx]):
                        meta = grid_meta[r_idx][c_i]
                        # Never downgrade a more specific diagnosis already made
                        # upstream (e.g. CONFLICT, where a subtotal row pinpoints
                        # which cell disagrees) to this generic row-level one.
                        meta.setdefault("arithmetic_status", "UNBALANCED")
                        meta["needs_review"] = True

    # =========================================================================
    # 11. Visual Cell Debug Card Export
    # =========================================================================
    def export_cell_debug_image(
        self,
        image: np.ndarray,
        cell_bbox: Tuple[int, int, int, int],
        doc_name: str,
        row_idx: int,
        col_idx: int,
        col_name: str,
        gt_val: str = "",
        ocr_val: str = "",
        out_dir: str = "outputs"
    ) -> Optional[str]:
        """
        Exports a multi-panel visual debug card showing:
        - Panel 1: Original cell crop
        - Panel 2: CLAHE enhanced crop
        - Panel 3: Otsu binarization
        - Panel 4: Sharpened edge crop
        With annotations for Row, Col, GT value, and OCR value.
        """
        try:
            x0, y0, x1, y1 = [int(v) for v in cell_bbox]
            h_img, w_img = image.shape[:2]
            pad = 4
            crop = image[max(0, y0-pad):min(h_img, y1+pad), max(0, x0-pad):min(w_img, x1+pad)]
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                return None

            ch, cw = crop.shape[:2]
            scale = 2.0 if ch < 40 else 1.0
            if scale > 1.0:
                crop_up = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
            else:
                crop_up = crop

            gray = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY) if len(crop_up.shape) == 3 else crop_up
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            var_b_gray = clahe.apply(gray)
            var_b = cv2.cvtColor(var_b_gray, cv2.COLOR_GRAY2BGR)

            _, otsu = cv2.threshold(var_b_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            var_c = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)

            blurred = cv2.GaussianBlur(crop_up, (0, 0), 1.5)
            var_d = cv2.addWeighted(crop_up, 1.6, blurred, -0.6, 0)
            if len(var_d.shape) == 2:
                var_d = cv2.cvtColor(var_d, cv2.COLOR_GRAY2BGR)
            if len(crop_up.shape) == 2:
                crop_up = cv2.cvtColor(crop_up, cv2.COLOR_GRAY2BGR)

            target_h = 70
            panels = [("Orig", crop_up), ("CLAHE", var_b), ("Otsu", var_c), ("Sharp", var_d)]
            vis_variants = []
            for v_name, v_img in panels:
                s = target_h / max(1, v_img.shape[0])
                target_w = max(40, int(v_img.shape[1] * s))
                v_resized = cv2.resize(v_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                cv2.putText(v_resized, v_name, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                vis_variants.append(v_resized)

            row_vis = np.hstack(vis_variants)
            banner_h = 28
            banner = np.zeros((banner_h, row_vis.shape[1], 3), dtype=np.uint8)
            banner[:] = (30, 30, 30)
            text_line = f"R{row_idx+1} C{col_idx+1} ({col_name}) | GT:'{gt_val}' | OCR:'{ocr_val}'"
            cv2.putText(banner, text_line[:55], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            card = np.vstack([banner, row_vis])
            out_p = Path(out_dir) / f"debug_cell_{Path(doc_name).stem}_r{row_idx+1}_c{col_idx+1}.png"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), card)
            return str(out_p)
        except Exception as e:
            logger.debug(f"Failed to export cell debug image: {e}")
            return None


cell_ocr_service = CellOCRService()
