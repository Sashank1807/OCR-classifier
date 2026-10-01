"""
Coordinate-Aware Table OCR Service
Preserves OCR bounding boxes, executes dynamic baseline clustering for physical lines,
constructs a global column model via multi-evidence fusion (ruling lines, headers, numeric alignment),
and groups physical lines into logical rows.
Guarantees 1-to-1 row-cell alignment and distinguishes blanks, zeros, and dashes.
"""

import re
import time
import unicodedata
import cv2
import numpy as np
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from app.core.config import settings
from app.core.logger import logger
from app.services.cell_ocr_service import PACKING_REGEX
from app.utils.row_types import PARTY_OR_HEADER


def _trailing_digit_run(s: str) -> Optional[str]:
    m = re.search(r'(\d+)$', s)
    return m.group(1) if m else None


def _digit_runs(s: str) -> List[str]:
    """All digit sequences in order - a product's distinguishing numbers, wherever they sit."""
    return re.findall(r'\d+', s)


def _description_similarity(norm_a: str, norm_b: str) -> float:
    """
    Length-ratio-gated similarity for deciding whether two normalized row
    descriptions are the same product, rather than pure substring containment
    (which wrongly treats e.g. "PARA500" as the same product as "PARA500XR"
    since one is a substring of the other despite being different SKUs).

    Plain character-overlap similarity (SequenceMatcher) is NOT sufficient on
    its own: two pharma SKUs that differ only by a dosage number ("MINOSTRONG
    1.25 TAB" vs "MINOSTRONG 2.5 TAB"; "PARA500" vs "PARA500XR") still score
    above 0.85 on pure character overlap despite being different products - the
    shared prefix dominates the ratio.

    So the dosage numbers must match exactly, compared ACROSS THE WHOLE string
    rather than only at the end. An earlier version compared only the trailing
    digit run, which silently merged (and therefore DROPPED) every SKU pair
    whose distinguishing number is interior and which ends in letters:
    "MINOSTRONG 1.25 TAB"/"MINOSTRONG 2.5 TAB" and "TREBOR 0.025% CREAM"/
    "TREBOR 0.05% CREAM" both end in "tab"/"cream", so both sides yielded "no
    trailing digits", the guard passed, and one of the two rows vanished.

    Erring toward "different product" is the safe direction here: a missed
    merge leaves a duplicate-looking line, while a wrong merge destroys a row.
    """
    if not norm_a or not norm_b:
        return 0.0
    shorter, longer = sorted((norm_a, norm_b), key=len)
    if len(shorter) / max(1, len(longer)) < 0.7:
        return 0.0
    if _digit_runs(norm_a) != _digit_runs(norm_b):
        return 0.0
    # Both guards are needed and catch different pairs: digit runs separate
    # "1.25 TAB"/"2.5 TAB" (interior number differs), while the trailing run
    # separates "PARA500"/"PARA500XR" (same numbers, one carries a suffix).
    if _trailing_digit_run(norm_a) != _trailing_digit_run(norm_b):
        return 0.0
    return SequenceMatcher(None, norm_a, norm_b).ratio()


class TableOCRService:
    """
    Layout-aware and coordinate-aware table extraction service.
    Reconstructs tables from exact OCR token geometries rather than string heuristics.
    """

    COMPOUND_HEADER_SUBWORDS = {
        "name", "qty", "val", "value", "price", "rate", "no", "code", "date",
        "amt", "amount", "loss", "dump", "issue", "receipt", "closing", "opening",
        "op", "cl", "bal", "pur", "sale", "exp", "stat", "statement"
    }

    # FALLBACK vocabulary only.
    #
    # Header detection is structural (see _detect_header_line): a header is
    # found by where it sits and what lines up beneath it, so a document using
    # labels nobody has ever listed is read correctly. This list is consulted
    # only when a page yields too little geometry to measure - too few numeric
    # rows to form column bands, or nothing above the body that aligns with
    # them. Adding a word here therefore no longer fixes a document; if an
    # unseen layout extracts badly, the structural signals are what to look at.
    #
    # Kept deliberately generic. Vendor names, city names and product words do
    # NOT belong here: they cannot generalise to the next document by
    # construction, and every one previously added had to be removed again.
    HEADER_KEYWORDS = [
        "product", "item", "description", "particulars", "code", "pack", "packing",
        "op", "bal", "balance", "opening", "receipt", "receipts", "issue", "issues",
        "total", "closing", "stock", "sale", "sales", "purchase", "rate", "amount",
        "value", "val", "qty", "quantity", "expiry", "exp", "batch", "unit", "sn",
        "sr", "mrp", "disc", "tax", "free", "dump", "near", "dty", "oty", "aty",
        "age", "ageing", "aging", "days", "qoh", "stk", "o.stk", "purc", "tot",
    ]

    # Keywords that also occur inside ordinary words, so they only count when
    # they ARE the token: "age" in "Page"/"package", "tot" in "total".
    WHOLE_TOKEN_KEYWORDS = {"age", "tot", "days", "qoh", "purc"}

    def __init__(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.engine = RapidOCR(
                det_box_thresh=settings.RAPIDOCR_DET_BOX_THRESH,
                det_unclip_ratio=settings.RAPIDOCR_DET_UNCLIP_RATIO,
                text_score=settings.RAPIDOCR_TEXT_SCORE,
            )
            self.engine_name = "RapidOCR"
        except Exception as e:
            self.engine = None
            self.engine_name = "None"
            logger.warning(f"RapidOCR engine initialization notice: {e}")

    def extract_table(self, image_path: Path, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Runs coordinate OCR on image and reconstructs full document layout & tables.
        Returns:
            {
                "markdown": str,
                "tables": List[Dict[str, Any]],
                "tokens": List[Dict[str, Any]],
                "raw_text": str,
                "confidence": float,
                "ocr_engine": str,
                "elapsed": float
            }
        """
        t0 = time.time()
        image_path = Path(image_path)

        if not self.engine or not image_path.exists():
            return {
                "markdown": "",
                "tables": [],
                "tokens": [],
                "raw_text": "",
                "confidence": 0.0,
                "ocr_engine": "None",
                "elapsed": 0.0
            }

        img = cv2.imread(str(image_path))
        if img is None:
            return {
                "markdown": "",
                "tables": [],
                "tokens": [],
                "raw_text": "",
                "confidence": 0.0,
                "ocr_engine": "None",
                "elapsed": 0.0
            }

        try:
            # Pass image array directly to ensure exact OpenCV EXIF orientation
            result, elapse = self.engine(img)
            t1 = time.time()
            elapsed = round(t1 - t0, 3)

            if not result:
                return {
                    "markdown": "No text has been found.",
                    "tables": [],
                    "tokens": [],
                    "raw_text": "",
                    "confidence": 0.0,
                    "ocr_engine": self.engine_name,
                    "elapsed": elapsed
                }

            # 1. Parse and preserve raw OCR tokens
            tokens: List[Dict[str, Any]] = []
            confidences: List[float] = []

            for item in result:
                box, text, score = item[0], item[1].strip(), float(item[2])
                if not text:
                    continue

                # The detector is CJK-trained and returns fullwidth forms for
                # characters the page prints as plain ASCII - "TOTAL ：" instead
                # of "TOTAL :", fullwidth digits, "％", "＊". Downstream keyword
                # matching, numeric parsing and comparison all assume ASCII, so
                # every one of those silently fails to match. NFKC maps the
                # compatibility forms onto their ASCII equivalents and leaves
                # genuine CJK characters (e.g. "一", handled separately as a
                # dash lookalike) and ordinary text untouched.
                text = unicodedata.normalize("NFKC", text)

                pts = np.array(box, dtype="float32")
                x0 = float(np.min(pts[:, 0]))
                y0 = float(np.min(pts[:, 1]))
                x1 = float(np.max(pts[:, 0]))
                y1 = float(np.max(pts[:, 1]))
                y_center = (y0 + y1) / 2.0
                h = max(1.0, y1 - y0)
                w = max(1.0, x1 - x0)

                # Tilt of this token's own detection quad (top edge, TL->TR). A
                # photographed page is rarely axis-aligned, and the detector's
                # quad follows the text baseline, so this is a direct read of
                # the local text skew - aggregated into a page slope in
                # _cluster_lines_by_y. Only wide tokens give a stable angle.
                edge_dx = float(pts[1][0] - pts[0][0])
                edge_dy = float(pts[1][1] - pts[0][1])
                quad_slope = (edge_dy / edge_dx) if abs(edge_dx) > 25.0 else None

                confidences.append(score)
                tokens.append({
                    "text": text,
                    "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "xc": (x0 + x1) / 2.0,
                    "yc": y_center,
                    "y_center": y_center,
                    "height": h,
                    "width": w,
                    "quad_slope": quad_slope,
                    "confidence": round(score, 4)
                })

            avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0

            # 2. Detect OpenCV Ruling Lines
            rulings = self._detect_ruling_lines(img)

            # 3. Filter external noise and detect table boundaries
            table_bbox, filtered_tokens = self._detect_table_bounds(tokens, img)

            # 4. Dynamic Y-Coordinate Clustering into horizontal physical lines
            lines = self._cluster_lines_by_y(filtered_tokens)

            # 5. Detect and reconstruct Table Grid using global column model & logical rows
            table_dict, non_table_lines, col_bounds, logical_row_bboxes, hdr_bbox = self._reconstruct_table_grid(
                lines, rulings, img.shape[1], img=img
            )

            # 6. Export Geometry Debug Images
            self._export_geometry_debug_images(
                img, table_bbox, hdr_bbox, logical_row_bboxes, col_bounds, tokens, image_path.stem
            )

            # 7. Assemble Final Markdown & Text
            markdown_parts = []
            for pre_text in non_table_lines.get("header_lines", []):
                markdown_parts.append(pre_text)

            tables_list = []
            if table_dict and table_dict.get("rows"):
                markdown_parts.append("")
                markdown_parts.append(table_dict["markdown"])
                markdown_parts.append("")
                tables_list.append(table_dict)

            # Footer lines are the application's own chrome - status bars, the
            # dealer's contact block, the monitor's brand - not document content.
            # They are still returned in `non_table` for auditing, but rendering
            # them into the deliverable makes the extracted document look like it
            # contains data it does not. Header lines are kept: those carry the
            # real report metadata (company, statement period).
            non_table_lines["footer_lines"] = [
                t for t in non_table_lines.get("footer_lines", []) if t and t.strip()
            ]

            combined_markdown = "\n".join(markdown_parts).strip()
            raw_text = "\n".join(t["text"] for t in tokens)

            return {
                "markdown": combined_markdown,
                "tables": tables_list,
                "tokens": tokens,
                "raw_text": raw_text,
                "confidence": round(avg_conf, 4),
                "ocr_engine": self.engine_name,
                "elapsed": elapsed
            }

        except Exception as e:
            logger.error(f"TableOCRService extraction failed: {str(e)}", exc_info=True)
            return {
                "markdown": "",
                "tables": [],
                "tokens": [],
                "raw_text": "",
                "confidence": 0.0,
                "ocr_engine": self.engine_name,
                "elapsed": 0.0
            }

    def _detect_ruling_lines(self, img: Optional[np.ndarray]) -> Dict[str, List[Tuple[float, float, float, float]]]:
        """
        Detects horizontal and vertical ruling lines in the document.
        Returns:
            {
                "horizontal": [(y0, y1, x0, x1), ...],
                "vertical": [(x0, x1, y0, y1), ...]
            }
        """
        if img is None:
            return {"horizontal": [], "vertical": []}

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        # Horizontal lines
        h_len = max(20, int(w / 35))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
        h_cnts, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_boxes = []
        for c in h_cnts:
            bx, by, bw_w, bw_h = cv2.boundingRect(c)
            if bw_w >= w * 0.15:
                h_boxes.append((float(by), float(by + bw_h), float(bx), float(bx + bw_w)))
        h_boxes.sort(key=lambda b: b[0])

        # Vertical lines
        v_len = max(20, int(h / 35))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
        v_cnts, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_boxes = []
        for c in v_cnts:
            bx, by, bw_w, bw_h = cv2.boundingRect(c)
            if bw_h >= h * 0.08:
                v_boxes.append((float(bx), float(bx + bw_w), float(by), float(by + bw_h)))
        v_boxes.sort(key=lambda b: b[0])

        return {"horizontal": h_boxes, "vertical": v_boxes}

    def _detect_table_bounds(
        self,
        tokens: List[Dict[str, Any]],
        img: np.ndarray
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Determines [table_x1, table_y1, table_x2, table_y2] and filters out external
        noise (monitor bezel text, camera watermarks, and separate side panels).
        """
        h, w = img.shape[:2]
        if not tokens:
            return [0.0, 0.0, float(w), float(h)], tokens

        # Detect screen photo conditions (e.g. Compaq bezel, OnePlus watermark)
        has_screen_bezel = any(
            any(k in t["text"].lower() for k in ["compaq", "18.5", "resolution", "1366x768", "vga", "dynamic contrast"])
            for t in tokens if t["y0"] < h * 0.25
        )

        filtered = []
        for t in tokens:
            txt_lower = t["text"].lower()
            # Filter monitor bezel - relative to image height so this doesn't depend on
            # a specific captured resolution.
            if has_screen_bezel and t["y0"] < h * 0.20:
                continue
            # Filter camera watermark at bottom
            if any(w_k in txt_lower for w_k in ["shot on", "oneplus", "chandan4u", "redmi note"]):
                continue
            # Filter side panel at right - relative to image width, not an absolute pixel
            # cutoff. Calibrated against real captured documents (not just a synthetic
            # test image): on a 1600px-wide screen photo, legitimate rightmost table
            # column data (e.g. a CLOSING column) sits up to x-ratio ~0.77, while a
            # genuinely separate side panel starts at ~0.80 - an inter-column gap in the
            # table itself can be larger in absolute pixels than the true table/panel
            # boundary, so gap-detection alone is not reliable here; 0.78 sits in the
            # narrow band between the two clusters with margin on both sides.
            if has_screen_bezel and t["x0"] > w * 0.78:
                continue
            filtered.append(t)

        if not filtered:
            filtered = tokens

        t_x1 = min(t["x0"] for t in filtered)
        t_y1 = min(t["y0"] for t in filtered)
        t_x2 = max(t["x1"] for t in filtered)
        t_y2 = max(t["y1"] for t in filtered)

        return [t_x1, t_y1, t_x2, t_y2], filtered

    def _estimate_page_slope(self, tokens: List[Dict[str, Any]]) -> float:
        """
        Estimates the page's text-baseline slope (dy/dx) from the OCR detection
        quads themselves.

        A photograph of a page or screen is almost never axis-aligned, and even
        a small tilt breaks row grouping: at 2 degrees, text 1000px to the right
        sits ~35px lower than text on the left - more than a row height, so a
        row's own right-hand cells land in the NEXT row's band. Deriving the
        angle from the text quads (rather than Hough lines over the whole image)
        deliberately ignores monitor bezels, table rules and page edges, which
        can be level while the text is not.
        """
        slopes = [t["quad_slope"] for t in tokens if t.get("quad_slope") is not None]
        if len(slopes) < 5:
            return 0.0
        page_slope = float(np.median(slopes))
        # Beyond ~8 degrees this is not a page tilt (rotated/upside-down capture,
        # or a bad quad); shearing on it would do more harm than leaving it.
        if not np.isfinite(page_slope) or abs(page_slope) > 0.15:
            return 0.0
        return page_slope

    def _cluster_lines_by_y(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clusters tokens into horizontal text lines by Y-coordinate using dynamic
        font-height tolerance and horizontal collision awareness.

        Comparisons run on a tilt-corrected Y (the token's Y shifted back along
        the page's own text baseline) so a skewed capture still groups into the
        rows a reader sees. Reported line geometry stays in true image
        coordinates - only the grouping decision is corrected.
        """
        if not tokens:
            return []

        sorted_tokens = sorted(tokens, key=lambda t: (t["y_center"], t["x0"]))
        median_h = float(np.median([t["height"] for t in sorted_tokens]))
        tol = max(7.0, median_h * 0.55)

        page_slope = self._estimate_page_slope(sorted_tokens)

        # Only shear when the tilt can actually break row grouping. Across the
        # token span, tilt displaces a row's far end by slope*span; if that stays
        # within the row tolerance, every token still lands in its own row band
        # unaided, and at that scale the slope estimate is itself mostly noise -
        # so correcting would perturb borderline decisions for no gain. The
        # margin above tol is that noise allowance.
        if page_slope:
            x_span = max(t["x1"] for t in sorted_tokens) - min(t["x0"] for t in sorted_tokens)
            if abs(page_slope) * x_span <= tol * 1.5:
                page_slope = 0.0

        def _y_adj(t: Dict[str, Any]) -> float:
            xc = t.get("xc", (t["x0"] + t["x1"]) / 2.0)
            return t["y_center"] - page_slope * xc

        if page_slope:
            sorted_tokens.sort(key=lambda t: (_y_adj(t), t["x0"]))
            logger.debug(
                f"Row clustering tilt-corrected: slope={page_slope:.5f} "
                f"({np.degrees(np.arctan(page_slope)):.2f} deg, "
                f"{page_slope * 1000:.0f}px drift per 1000px)"
            )

        lines: List[Dict[str, Any]] = []
        for t in sorted_tokens:
            t_y = _y_adj(t)
            best_line = None
            best_dist = 999.0
            for l in lines:
                # Anchored on the row's mean Y (after the page-tilt shear above).
                # Two ways of following physically CURVED paper were tried here
                # and both measured worse, so the simple anchor stands: a
                # nearest-token-in-X anchor let a row walk into its neighbour one
                # hop at a time, and a per-row least-squares baseline fit did not
                # help the curved document at all while breaking a flat one
                # (1000411295 row accuracy 94% -> 59% in both cases). Curvature
                # appears to need real image dewarping before OCR, not a smarter
                # y-banding rule.
                dist = abs(l["y_adj"] - t_y)
                if dist <= tol:
                    has_collision = False
                    for exist_tok in l["tokens"]:
                        x_overlap = min(exist_tok["x1"], t["x1"]) - max(exist_tok["x0"], t["x0"])
                        min_w = min(exist_tok["x1"] - exist_tok["x0"], t["x1"] - t["x0"])
                        if x_overlap > max(20.0, 0.45 * min_w):
                            has_collision = True
                            break
                    if not has_collision and dist < best_dist:
                        best_dist = dist
                        best_line = l

            if best_line is not None:
                best_line["tokens"].append(t)
                best_line["y_adj"] = sum(_y_adj(x) for x in best_line["tokens"]) / len(best_line["tokens"])
                best_line["y_avg"] = sum(x["y_center"] for x in best_line["tokens"]) / len(best_line["tokens"])
            else:
                lines.append({
                    "y_adj": t_y,
                    "y_avg": t["y_center"],
                    "tokens": [t]
                })

        lines.sort(key=lambda l: l["y_adj"])
        for line in lines:
            line["tokens"].sort(key=lambda item: item["x0"])
            line["text"] = " ".join(item["text"] for item in line["tokens"])
            line["x0"] = min(item["x0"] for item in line["tokens"])
            line["x1"] = max(item["x1"] for item in line["tokens"])
            line["y0"] = min(item.get("y0", item.get("y_center", 0.0) - item.get("height", 10.0) / 2.0) for item in line["tokens"])
            line["y1"] = max(item.get("y1", item.get("y_center", 0.0) + item.get("height", 10.0) / 2.0) for item in line["tokens"])

        return lines

    # ------------------------------------------------------------------ #
    # Structural table signals
    #
    # These describe a table's SHAPE, not its wording, and that is what lets
    # an unseen document be read. A column-label row is recognisable because
    # it is the last predominantly-textual line sitting directly above a run
    # of numeric-dense lines, and because its tokens line up with the vertical
    # bands those figures form - regardless of whether it reads "Closing",
    # "Cl.Bal", "Qoh" or a word no fixture has ever contained.
    # ------------------------------------------------------------------ #

    _NUMERIC_TOKEN_RE = re.compile(r'^[-+(]?[\d,]*\.?\d+\)?%?$')
    _DASH_TOKENS = {"-", "--", "---", "一", "—", "–"}

    @classmethod
    def _is_numeric_token(cls, txt: str) -> bool:
        s = (txt or "").strip()
        if not s:
            return False
        return bool(cls._NUMERIC_TOKEN_RE.match(s)) or s in cls._DASH_TOKENS

    @classmethod
    def _numeric_profile(cls, line: Dict[str, Any]) -> Tuple[int, int]:
        toks = line.get("tokens") or []
        return sum(1 for t in toks if cls._is_numeric_token(t.get("text", ""))), len(toks)

    @classmethod
    def _is_data_like(cls, line: Dict[str, Any]) -> bool:
        """A line whose token mix reads as table figures rather than prose."""
        n, total = cls._numeric_profile(line)
        return total >= 2 and n >= 2 and (n / total) >= 0.30

    @staticmethod
    def _token_xc(t: Dict[str, Any]) -> float:
        return float(t.get("xc", (t["x0"] + t["x1"]) / 2.0))

    @staticmethod
    def _looks_like_report_metadata(text: str) -> bool:
        """
        A report-period or identifier line, recognised by shape not wording.

        "From 01/05/2026 To 29/05/2026", a GSTIN, a phone number: these sit in
        the header band's x-range and would otherwise anchor the header row on
        report metadata, shifting every column right.
        """
        t = (text or "").lower()
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}.*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', t):
            return True
        if re.search(r'\bfrom\b.*\bto\b.*\d', t):
            return True
        # A 10+ character alphanumeric run beginning with a digit is an
        # identifier (GSTIN, phone, licence no), never a column label.
        return bool(re.search(r'\d[\dA-Za-z]{9,}', t))

    def _find_body_start(self, lines: List[Dict[str, Any]], search_limit: int = 25) -> Optional[int]:
        """
        Index of the first line of the numeric-dense table body.

        Corroboration is required so that a stray figure in the letterhead - a
        date, a phone number - cannot be mistaken for the body. But the
        corroborating lines must NOT have to be consecutive: these ERP layouts
        interleave customer-name and division-banner lines between the figure
        rows, and demanding an unbroken run made the first real data row look
        like more preamble, so it was swallowed into the header band.
        """
        window = 4
        first_any = None
        for i, l in enumerate(lines[:search_limit]):
            if not self._is_data_like(l):
                continue
            if first_any is None:
                first_any = i
            if sum(1 for w in lines[i:i + window] if self._is_data_like(w)) >= 2:
                return i
        return first_any

    def _column_anchors(self, lines: List[Dict[str, Any]]) -> List[float]:
        """
        The x-centres of the vertical bands the body's figures line up in.

        These ARE the columns, measured off the data itself with no labels
        involved. A band only counts when several rows agree on it, so a
        single stray number cannot invent one.
        """
        xs = sorted(
            self._token_xc(t)
            for l in lines for t in (l.get("tokens") or [])
            if self._is_numeric_token(t.get("text", ""))
        )
        if not xs:
            return []
        clusters, cur = [], [xs[0]]
        for x in xs[1:]:
            if x - float(np.mean(cur)) <= 35.0:
                cur.append(x)
            else:
                clusters.append(cur)
                cur = [x]
        clusters.append(cur)
        min_support = max(2, int(len(lines) * 0.25))
        return [float(np.mean(c)) for c in clusters if len(c) >= min_support]

    def _alignment_score(self, line: Dict[str, Any], anchors: List[float], tol: float = 45.0) -> float:
        """
        What fraction of the body's column bands this line puts a token above.

        The label row scores high because it has one token per band; a banner,
        an address block or a division title does not. The tolerance is
        generous on purpose - a label is often printed left of the
        right-aligned figures it names, so overlap counts, not coincidence.
        """
        toks = line.get("tokens") or []
        if not toks or not anchors:
            return 0.0
        covered = set()
        for t in toks:
            for k, a in enumerate(anchors):
                if t["x0"] - tol <= a <= t["x1"] + tol:
                    covered.add(k)
                    break
        return len(covered) / len(anchors)

    def _detect_header_line(
        self,
        lines: List[Dict[str, Any]],
        body_start: Optional[int],
        look_back: int = 6
    ) -> Optional[int]:
        """
        Locate the column-label row structurally, with no vocabulary at all.

        Of the textual lines immediately above the numeric body, the label row
        is the one whose tokens best line up with the bands the body's own
        figures form. Ties go to the lowest line: where a header is stacked in
        tiers, the leaf tier - the one actually naming each column - is the
        one nearest the data.
        """
        if body_start is None or body_start <= 0:
            return None
        anchors = self._column_anchors(lines[body_start:])
        if len(anchors) < 2:
            return None

        window_start = max(0, body_start - look_back)
        best, best_score = None, 0.0
        for i in range(window_start, body_start):
            line = lines[i]
            if len(line.get("tokens") or []) < 2:
                continue
            n_num, n_tot = self._numeric_profile(line)
            if n_tot and (n_num / n_tot) > 0.5:
                continue
            if self._looks_like_report_metadata(line.get("text", "")):
                continue
            score = self._alignment_score(line, anchors) + 0.02 * (i - window_start)
            if score > best_score:
                best, best_score = i, score

        # Below roughly half the bands labelled there is no real evidence this
        # is a header rather than a stray line of prose; the caller falls back.
        return best if best_score >= 0.4 else None

    def _name_from_header(
        self,
        hdr_lines: List[Dict[str, Any]],
        xc: float,
        tol: float = 60.0
    ) -> Optional[str]:
        """The label actually printed above a column band, if the page has one."""
        best, best_d = None, None
        for l in hdr_lines:
            for t in l.get("tokens", []):
                name = (t.get("text") or "").strip()
                if not name or self._is_numeric_token(name):
                    continue
                if t["x0"] - tol <= xc <= t["x1"] + tol:
                    d = abs(self._token_xc(t) - xc)
                    if best_d is None or d < best_d:
                        best, best_d = name, d
        return best

    def _split_compound_header_token(
        self,
        token: Dict[str, Any],
        anchors: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split a header label that tight spacing or a dot-matrix printer fused
        with its neighbour ("Op.BalReceipt", "Qty.Balance").

        Which labels those are is not known in advance, so nothing here matches
        on words. The token must physically straddle two of the bands the data
        forms, and it is cut at a boundary its own casing, punctuation or
        spacing provides, at whichever such boundary falls nearest the gap
        between those two bands.
        """
        txt = token["text"].strip()
        x0, x1 = float(token["x0"]), float(token["x1"])
        if not anchors or len(txt) < 6 or x1 <= x0:
            return [token]

        inside = sorted(a for a in anchors if x0 - 5.0 <= a <= x1 + 5.0)
        if len(inside) < 2:
            return [token]

        cuts = sorted({m.start() + 1 for m in re.finditer(r'[a-z.)\s][A-Z]', txt)}
                      | {m.start() for m in re.finditer(r'(?<=\S)\s(?=\S)', txt)})
        cuts = [c for c in cuts if 2 <= c <= len(txt) - 2]
        if not cuts:
            return [token]

        mid_x = (inside[0] + inside[1]) / 2.0
        target = ((mid_x - x0) / (x1 - x0)) * len(txt)
        cut = min(cuts, key=lambda c: abs(c - target))
        left, right = txt[:cut].strip(), txt[cut:].strip()
        if not left or not right:
            return [token]

        split_x = x0 + (cut / len(txt)) * (x1 - x0)
        return [
            {**token, "text": left, "x0": x0, "x1": split_x},
            {**token, "text": right, "x0": split_x, "x1": x1},
        ]

    def _synthesize_columns_from_band(
        self,
        hdr_lines: List[Dict[str, Any]],
        data_lines: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes a unified, non-overlapping list of columns from 1, 2, or multi-line header bands.
        """
        if not hdr_lines:
            return []

        body_anchors = self._column_anchors(data_lines)

        all_hdr_tokens: List[Dict[str, Any]] = []
        for l in hdr_lines:
            for t in l["tokens"]:
                all_hdr_tokens.extend(self._split_compound_header_token(t, body_anchors))

        # Sort tokens bottom-to-top so leaf sub-headers form clusters first
        all_hdr_tokens.sort(key=lambda t: (-t.get("y0", 0.0), t["x0"]))

        col_clusters: List[Dict[str, Any]] = []
        for t in all_hdr_tokens:
            t_w = max(1.0, t["x1"] - t["x0"])
            matching_indices = []
            for idx, cl in enumerate(col_clusters):
                x_overlap = min(cl["x1"], t["x1"]) - max(cl["x0"], t["x0"])
                min_w = min(cl["x1"] - cl["x0"], t_w)
                ratio = x_overlap / min_w if min_w > 0 else 0
                if ratio > 0.25:
                    matching_indices.append((idx, ratio))

            # Case 1: Spans multiple existing child clusters -> Super-header! Qualifies both without merging
            if len(matching_indices) > 1:
                for m_idx, _ in matching_indices:
                    col_clusters[m_idx]["tokens"].append(t)
            # Case 2: Overlaps with exactly one cluster -> Add and adjust bounds
            elif len(matching_indices) == 1:
                m_idx = matching_indices[0][0]
                col_clusters[m_idx]["tokens"].append(t)
                col_clusters[m_idx]["x0"] = min(col_clusters[m_idx]["x0"], t["x0"])
                col_clusters[m_idx]["x1"] = max(col_clusters[m_idx]["x1"], t["x1"])
            # Case 3: New column cluster
            else:
                col_clusters.append({
                    "x0": t["x0"],
                    "x1": t["x1"],
                    "tokens": [t]
                })

        col_clusters.sort(key=lambda cl: cl["x0"])

        def _is_known_label_word(txt: str) -> bool:
            clean = re.sub(r'[^a-z]', '', (txt or "").lower())
            if not clean:
                return False
            return any(
                clean == k if (len(k) <= 2 or k in self.WHOLE_TOKEN_KEYWORDS) else k in clean
                for k in self.HEADER_KEYWORDS
            )

        # A banner or division title printed close to the header row gets merged
        # into it by y-clustering and ends up inside a column's NAME - "Product
        # Name HETERODERMA GLOW", "HETERO DRUGS LTD : Company DESCRIPTION". The
        # stray text can land on either side of the real label, so position
        # within the cluster is not enough to tell them apart.
        #
        # They are separable by BASELINE instead, which needs no word list: a
        # row of column labels runs across the table, so its baseline is shared
        # by several different columns, whereas a banner title sits on a
        # baseline of its own touching just one. Group the header band's tokens
        # into baselines, and treat a baseline that spans two or more columns as
        # a label tier - multi-tier headers and super-headers qualify, one-off
        # banner text does not.
        heights = [
            float(t.get("y1", 0.0)) - float(t.get("y0", 0.0))
            for t in all_hdr_tokens
            if float(t.get("y1", 0.0)) > float(t.get("y0", 0.0))
        ]
        base_tol = (float(np.median(heights)) * 0.6) if heights else 6.0

        def _yc(t: Dict[str, Any]) -> float:
            return (float(t.get("y0", 0.0)) + float(t.get("y1", 0.0))) / 2.0

        # Group by proximity rather than by rounding into fixed-width buckets:
        # a bucket boundary can fall between two tokens that plainly share a
        # baseline (y=100 and y=106 land either side of a 12px bucket edge),
        # which would split one label row in two and defeat the spread test.
        ys = sorted(_yc(t) for t in all_hdr_tokens)
        baseline_bands: List[Tuple[float, float]] = []
        if ys:
            run = [ys[0]]
            for y in ys[1:]:
                if y - run[-1] <= base_tol:
                    run.append(y)
                else:
                    baseline_bands.append((run[0], run[-1]))
                    run = [y]
            baseline_bands.append((run[0], run[-1]))

        def _baseline_key(t: Dict[str, Any]) -> int:
            yc = _yc(t)
            for k, (lo, hi) in enumerate(baseline_bands):
                if lo - 0.01 <= yc <= hi + 0.01:
                    return k
            return -1

        baseline_spread: Dict[int, set] = {}
        for c_i, cl in enumerate(col_clusters):
            for t in cl["tokens"]:
                baseline_spread.setdefault(_baseline_key(t), set()).add(c_i)

        cols: List[Dict[str, Any]] = []
        for cl in col_clusters:
            cl["tokens"].sort(key=lambda t: (t.get("y0", 0.0), t["x0"]))

            tier_tokens = [
                t for t in cl["tokens"]
                if len(baseline_spread.get(_baseline_key(t), ())) >= 2
            ]
            # If nothing in this cluster sits on a shared baseline, keep what is
            # there: a garbled or lonely label still beats an empty one.
            cluster_tokens = tier_tokens if tier_tokens else cl["tokens"]

            # COSMETIC ONLY - cleaning up the label STRING, never deciding which
            # columns exist or where they sit.
            #
            # Baseline geometry separates a banner on its own line from the
            # label row, but it cannot separate OCR garble sitting on the label
            # row itself: in 'ksue Closing' both words share a baseline and only
            # one of them is a word. Telling those apart is a recognition
            # problem, and recognition is the one place a vocabulary genuinely
            # helps - so when a cluster holds BOTH recognised and unrecognised
            # words, the recognised ones are the name and the rest is noise.
            #
            # A cluster with no recognised word is left exactly as printed. That
            # is the important half: a column labelled with something this list
            # has never heard of keeps its real name instead of being blanked.
            known = [t for t in cluster_tokens if _is_known_label_word(t["text"])]
            if known and len(known) < len(cluster_tokens):
                cluster_tokens = known

            words = []
            for t in cluster_tokens:
                w = t["text"].strip()
                w_norm = re.sub(r'^(aty\.?|oty\.?|dty\.?)$', 'Qty.', w, flags=re.IGNORECASE)
                w_norm = re.sub(r'^nane$', 'Name', w_norm, flags=re.IGNORECASE)
                if not words or words[-1].lower() != w_norm.lower():
                    words.append(w_norm)
            col_name = " ".join(words).strip()
            cols.append({
                "name": col_name,
                "x0": cl["x0"],
                "x1": cl["x1"]
            })

        return cols

    def _build_global_column_model(
        self,
        hdr_lines: List[Dict[str, Any]],
        data_lines: List[Dict[str, Any]],
        rulings: Dict[str, List[Tuple[float, float, float, float]]],
        img_width: int
    ) -> Tuple[List[Tuple[str, float, float]], bool]:
        """
        Builds an immutable global column model for the table using:
        1. Multi-tier synthesized header intervals.
        2. Repeated numeric alignment clusters.
        3. Inferred Packing / Unit column.
        4. Gap recovery for missing numeric columns (e.g. Opening/Receipt in screen photos),
           corroborated by a detected vertical ruling line OR by numeric alignment
           across at least 3 data lines with a plausible gap width - a gap with only
           weak/uncorroborated numeric evidence is absorbed into the neighboring
           column instead of spawning a possibly-fabricated column.
        5. Trailing column recovery (e.g. (%) or Net Amount), gated the same way.

        Returns (col_bounds, column_model_uncertain) - the latter is True whenever
        at least one candidate gap/trailing column was found but rejected for lack
        of corroborating evidence, signalling the model may be missing a column
        rather than having fabricated one.
        """
        cols = self._synthesize_columns_from_band(hdr_lines, data_lines)
        if not cols:
            return [], False

        column_model_uncertain = False

        # Vertical ruling x-centers whose vertical span actually overlaps the table
        # body (data rows), used as direct physical evidence for a column boundary -
        # much stronger evidence than numeric clustering alone.
        vertical_ruling_xs: List[float] = []
        data_y_vals = [t.get("y0", 0.0) for dl in data_lines for t in dl.get("tokens", [])] + \
                      [t.get("y1", 0.0) for dl in data_lines for t in dl.get("tokens", [])]
        if data_y_vals and rulings:
            data_y0, data_y1 = min(data_y_vals), max(data_y_vals)
            for (rx0, rx1, ry0, ry1) in rulings.get("vertical", []):
                if ry1 >= data_y0 and ry0 <= data_y1:
                    vertical_ruling_xs.append((rx0 + rx1) / 2.0)

        def _ruling_supports_gap(gap_left: float, gap_right: float) -> bool:
            return any(gap_left <= rx <= gap_right for rx in vertical_ruling_xs)

        # A. Detect numeric alignment clusters across all data lines
        num_xc_list = []
        for dl in data_lines:
            for dt in dl.get("tokens", []):
                txt = dt["text"].strip()
                if re.match(r'^-?\d+(\.\d+)?$', txt) or txt in ['-', '--', '---', '一']:
                    num_xc_list.append(dt["xc"])

        # Each cluster is (mean_x, member_count) so downstream gating can require
        # stronger numeric corroboration (>=3 lines) before fabricating a column.
        num_clusters: List[Tuple[float, int]] = []
        if num_xc_list:
            sorted_num_xc = sorted(num_xc_list)
            cur_cl = [sorted_num_xc[0]]
            for x in sorted_num_xc[1:]:
                if x - np.mean(cur_cl) <= 35.0:
                    cur_cl.append(x)
                else:
                    if len(cur_cl) >= 2:
                        num_clusters.append((float(np.mean(cur_cl)), len(cur_cl)))
                    cur_cl = [x]
            if len(cur_cl) >= 2:
                num_clusters.append((float(np.mean(cur_cl)), len(cur_cl)))

        # B. Detect packing / unit tokens between description and first numeric column
        has_packing_col = any("pack" in c["name"].lower() or "unit" in c["name"].lower() for c in cols)
        first_num_x = num_clusters[0][0] if num_clusters else float(img_width) * 0.5
        desc_col = next((c for c in cols if any(k in c["name"].lower() for k in ["desc", "particular", "product", "item", "name"])), None)

        # Recover a missing Description/Product-name column. Every other column
        # recovered above (packing, numeric gaps, trailing (%)) has a fallback
        # for when its header wasn't detected - the description column, despite
        # being the single most important one, does not: if its header text was
        # blank, faint, or simply missed by OCR, no cluster in `cols` matches the
        # description keywords, and the whole column silently vanishes with its
        # text absorbed into whatever column starts next (e.g. "XTANZ-TAB 10.S"
        # swallowed whole into a "PACKING" column). Corroborate with real data
        # evidence rather than assuming: only synthesize a leading column when
        # >=3 data lines have their own leftmost token (real alphabetic text,
        # not a stray digit) starting well to the left of the current first
        # column - the same "requires convergent evidence" pattern used above.
        if desc_col is None and cols:
            leftmost_col = cols[0]
            desc_left_edges = []
            for dl in data_lines:
                toks = sorted(dl.get("tokens", []), key=lambda t: t["x0"])
                if not toks:
                    continue
                first_tok = toks[0]
                if first_tok["x0"] < leftmost_col["x0"] - 20.0 and re.search(r'[A-Za-z]{2,}', first_tok["text"].strip()):
                    desc_left_edges.append(first_tok["x0"])
            if len(desc_left_edges) >= 3:
                new_x0 = max(0.0, min(desc_left_edges) - 10.0)
                new_x1 = leftmost_col["x0"] - 5.0
                if new_x1 > new_x0 + 20.0:
                    cols.insert(0, {"name": "Description", "x0": new_x0, "x1": new_x1})
                    desc_col = cols[0]
                else:
                    column_model_uncertain = True
            else:
                column_model_uncertain = True

        desc_x1 = desc_col["x1"] if desc_col else 0.0

        packing_tokens = []
        for dl in data_lines:
            tokens_in_line = dl.get("tokens", [])
            for dt in tokens_in_line:
                txt = dt["text"].strip()
                # Must be strictly between description and first numeric column
                if dt["xc"] < first_num_x - 20.0 and dt["x0"] >= desc_x1 - 15.0:
                    # Dedicated packing/unit token (e.g. '10TAB', '1BOT', '100GM', '10*10', '1X60ML', '1X10')
                    # NOT a full product description string like 'BILASET 40 TAB'
                    m_pack = PACKING_REGEX.match(txt)
                    pure_pack_match = bool(m_pack and m_pack.end() == len(txt))
                    if pure_pack_match:
                        # Must have another token to its left or be well-separated from description
                        has_left_token = any(other_t["x1"] <= dt["x0"] + 10.0 for other_t in tokens_in_line if other_t != dt)
                        if has_left_token or (dt["x0"] >= desc_x1):
                            packing_tokens.append(dt)

        if not has_packing_col and len(packing_tokens) >= 3:
            pack_xc = float(np.mean([t["xc"] for t in packing_tokens]))
            desc_xc = desc_col["xc"] if desc_col and "xc" in desc_col else (desc_col["x0"] + desc_col["x1"]) / 2.0 if desc_col else 0.0
            if pack_xc > desc_xc + 40.0:
                desc_idx = 0
                for ci, c in enumerate(cols):
                    if any(k in c["name"].lower() for k in ["desc", "particular", "product", "item", "name"]):
                        desc_idx = ci
                        break
                cols.insert(desc_idx + 1, {"name": "Packing", "x0": pack_xc - 30.0, "x1": pack_xc + 30.0})

        # C. Insert missing numeric columns based on measured numeric clusters in gaps,
        # but only when corroborated by a ruling line or by strong (>=3 line) numeric
        # alignment with a plausible gap width - an uncorroborated cluster is left for
        # the neighboring column to absorb rather than spawning a possibly-fabricated
        # column that would silently shift real data under an invented header.
        MIN_PLAUSIBLE_GAP_WIDTH = 30.0
        MIN_CORROBORATING_LINES = 3
        inserted_num_cols = []
        for i in range(len(cols) - 1):
            c_left = cols[i]
            c_right = cols[i + 1]
            gap_left, gap_right = c_left["x1"] + 15.0, c_right["x0"] - 15.0
            gap_width = c_right["x0"] - c_left["x1"]
            gap_clusters = [nc for nc in num_clusters if gap_left <= nc[0] <= gap_right]
            if gap_clusters:
                for cl_idx, (g_nc, g_count) in enumerate(gap_clusters):
                    ruling_supported = _ruling_supports_gap(gap_left, gap_right)
                    numeric_supported = (g_count >= MIN_CORROBORATING_LINES and gap_width >= MIN_PLAUSIBLE_GAP_WIDTH)
                    if not (ruling_supported or numeric_supported):
                        column_model_uncertain = True
                        continue
                    # Name it from whatever the page actually prints above this
                    # band. Positional guesses ("the first inserted column must
                    # be Opening, the second Receipt") are right only on the
                    # layouts they were derived from and silently mislabel real
                    # data everywhere else; an honest placeholder is better,
                    # and D1b/D2 below can still reunite it with a real label.
                    col_name = self._name_from_header(hdr_lines, g_nc) or f"Col_Num_{i}_{cl_idx+1}"
                    inserted_num_cols.append((i + 1, {"name": col_name, "x0": g_nc - 25.0, "x1": g_nc + 25.0}))

        for ins_idx, new_c in reversed(inserted_num_cols):
            cols.insert(ins_idx, new_c)

        # D. Trailing numeric column (e.g. (%) or Net Amount on far right), same gating
        if num_clusters and num_clusters[-1][0] > (cols[-1]["x1"] + 25.0):
            last_nc, last_count = num_clusters[-1]
            trailing_ruling_supported = any(rx > cols[-1]["x1"] + 10.0 for rx in vertical_ruling_xs)
            if trailing_ruling_supported or last_count >= MIN_CORROBORATING_LINES:
                # Name it from a header token actually printed above this band.
                # Asserting "(%)" was a guess that happened to be right on the
                # one fixture with a percentage column and would mislabel real
                # data on every other layout, so there is no invented fallback
                # any more - an unnamed band stays honestly unnamed.
                trailing_name = self._name_from_header(hdr_lines, last_nc) or f"Col_Num_{len(cols)}_1"
                cols.append({"name": trailing_name, "x0": last_nc - 25.0, "x1": last_nc + 25.0})
            else:
                column_model_uncertain = True

        # D1b. Reunite a header label with its own data column.
        #
        # A header label is not always printed over the values it names - a label
        # can sit left of right-aligned figures, or simply be narrower than its
        # column. The header-derived column then captures almost nothing, while
        # section C sees the unclaimed numbers beside it and invents a column for
        # them. The table ends up with both: an empty named column AND an unnamed
        # twin holding its values (observed: "Opening" filled 2/16 rows sitting
        # next to Col_Num_3_1 filled 14/16). That inflates the column count and
        # leaves every real column mislabelled.
        #
        # Where an invented column carries the data and an adjacent named one is
        # effectively empty, they describe one column: keep the data, take the name.
        def _data_hits(c: Dict[str, Any]) -> int:
            return sum(
                1 for dl in data_lines
                if any(c["x0"] - 5.0 <= dt["xc"] <= c["x1"] + 5.0 for dt in dl.get("tokens", []))
            )

        if len(data_lines) >= 4:
            hits = [_data_hits(c) for c in cols]
            absorbed: set = set()
            for i, c in enumerate(cols):
                if not str(c["name"]).startswith("Col_Num_") or i in absorbed:
                    continue
                # Prefer whichever neighbour is emptier and carries a real name.
                for j in (i - 1, i + 1):
                    if not (0 <= j < len(cols)) or j in absorbed:
                        continue
                    nb = cols[j]
                    if str(nb["name"]).startswith("Col_Num_"):
                        continue
                    # Complementary fill is the real signal that these are one
                    # column split in two: a value lands in the named column OR
                    # in the invented one, but almost never in both on the same
                    # line. That happens because right-aligned figures of
                    # different widths have different centres, so a boundary
                    # taken from the header's position cuts through the column.
                    # (A "named column is nearly empty" test alone missed this -
                    # there the split was roughly even, 24 rows vs 15.)
                    both = sum(
                        1 for dl in data_lines
                        if any(nb["x0"] - 5.0 <= dt["xc"] <= nb["x1"] + 5.0 for dt in dl.get("tokens", []))
                        and any(c["x0"] - 5.0 <= dt["xc"] <= c["x1"] + 5.0 for dt in dl.get("tokens", []))
                    )
                    complementary = both <= max(1, int(len(data_lines) * 0.1))
                    if complementary and (hits[i] + hits[j]) > max(hits[i], hits[j]):
                        logger.debug(
                            f"Merged {nb['name']!r} ({hits[j]}/{len(data_lines)} lines) with invented "
                            f"{c['name']!r} ({hits[i]}/{len(data_lines)}), co-occurring in only "
                            f"{both} - one column split across two boundaries"
                        )
                        c["name"] = nb["name"]
                        c["x0"] = min(c["x0"], nb["x0"])
                        c["x1"] = max(c["x1"], nb["x1"])
                        absorbed.add(j)
                        break
            if absorbed:
                cols = [c for k, c in enumerate(cols) if k not in absorbed]

        # NOTE: adjacent columns sharing a label ("Opn!", "opn!") are NOT a
        # split column. Verified on sanjeev medical bijnor.jpeg: both hold data
        # in 11 of 12 rows, because a super-header ("Opn") spans two real
        # sub-columns (Qty and Value) and only the super-header was detected.
        # The column POSITIONS are right and only the labels are duplicated, so
        # fusing them would destroy a real column. A fusion pass was written and
        # removed after its own evidence check refused to fire on every case.

        # D2. Drop columns that are pure header-OCR noise. When a header label is
        # badly misread it can fragment into several clusters (e.g. a "(%)"
        # header coming back as "C" plus a CJK glyph), inventing a column that
        # never holds any data. That extra column shifts every column index
        # after it, so downstream consumers can no longer line the table up
        # positionally. Requires BOTH signals before dropping - an unreadable
        # name alone is not enough (a legitimately all-blank column with a
        # readable header, e.g. "Near Expiry", must survive), and neither is
        # emptiness alone.
        def _name_is_unreadable(name: str) -> bool:
            # Our own gap-inserted placeholder. It was never read off the page,
            # so if it also holds no data there is nothing to justify it.
            if re.match(r'^Col_Num_\d+_\d+$', name or ""):
                return True
            # Readability is a property of the string, not of a vocabulary: a
            # label we have never seen ("Qoh", "Liq") is perfectly readable,
            # while a lone stray glyph left behind by a misread is not.
            clean = re.sub(r'[^a-z0-9]', '', (name or "").lower())
            return len(clean) < 2

        if len(cols) > 2:
            kept = []
            for c in cols:
                if _name_is_unreadable(c["name"]):
                    # No padding for these: a neighbouring column's token drifting
                    # within 10px was enough to keep an empty invented column alive.
                    pad = 0.0 if re.match(r'^Col_Num_\d+_\d+$', c["name"] or "") else 10.0
                    has_data = any(
                        c["x0"] - pad <= dt["xc"] <= c["x1"] + pad
                        for dl in data_lines for dt in dl.get("tokens", [])
                    )
                    if not has_data:
                        logger.debug(
                            f"Dropped empty column with unreadable header {c['name']!r} "
                            f"(x {c['x0']:.0f}-{c['x1']:.0f}) - header OCR noise, no data in range"
                        )
                        column_model_uncertain = True
                        continue
                kept.append(c)
            if len(kept) >= 2:
                cols = kept

        # E. Build strict, non-overlapping column bounds
        col_bounds: List[Tuple[str, float, float]] = []
        num_c = len(cols)
        # Determine actual table right boundary: if last column ends well before img_width,
        # do not let it extend blindly to img_width (e.g. screen photo with side panels)
        table_right_limit = float(img_width)
        if hdr_lines:
            hdr_right = max(t["x1"] for l in hdr_lines for t in l.get("tokens", []))
            data_right = max((t["x1"] for l in data_lines for t in l.get("tokens", []) if t["x1"] <= hdr_right + 60.0), default=hdr_right)
            table_right_limit = min(float(img_width), max(hdr_right, data_right) + 25.0)

        for i in range(num_c):
            lb = 0.0 if i == 0 else (cols[i - 1]["x1"] + cols[i]["x0"]) / 2.0
            # If left gap between adjacent columns is large (> 100px), do not let left boundary stretch excessively into the gap
            if i > 0 and (cols[i]["x0"] - lb) > 100.0:
                col_w = cols[i]["x1"] - cols[i]["x0"]
                lb = max(lb, cols[i]["x0"] - max(60.0, col_w * 0.8))

            rb = table_right_limit if i == num_c - 1 else (cols[i]["x1"] + cols[i + 1]["x0"]) / 2.0
            col_bounds.append((cols[i]["name"], lb, rb))

        return col_bounds, column_model_uncertain

    def _assemble_logical_rows(
        self,
        lines: List[Dict[str, Any]],
        hdr_end_idx: int,
        col_bounds: List[Tuple[str, float, float]],
        rulings: Optional[Dict[str, List[Tuple[float, float, float, float]]]] = None
    ) -> Tuple[List[List[str]], List[List[Dict[str, Any]]], List[str]]:
        """
        Assembles physical lines into logical rows using a two-level model.
        Separates continuation lines from distinct transaction rows.
        """
        grid_rows: List[List[str]] = []
        grid_metadata: List[List[Dict[str, Any]]] = []
        footer_lines: List[str] = []
        # Tracks the bottom (y1) of the most recent physical line folded into each
        # logical row, so a detected horizontal ruling line between two physical
        # lines can act as a hard boundary that suppresses the row-merge
        # heuristics below (Case A/B/C, continuation) from firing across it -
        # a ruling line is strong physical evidence that two lines are genuinely
        # different rows/cells, independent of description-text similarity.
        last_line_y1: List[Optional[float]] = []
        horizontal_rulings = (rulings or {}).get("horizontal", [])

        def _ruling_between(y_top: Optional[float], y_bottom: float) -> bool:
            if y_top is None or y_bottom <= y_top:
                return False
            for (ry0, ry1, _rx0, _rx1) in horizontal_rulings:
                ry_center = (ry0 + ry1) / 2.0
                if y_top + 1.0 < ry_center < y_bottom - 1.0:
                    return True
            return False

        desc_col_idx = 0
        unit_col_idx = None
        numeric_col_indices = []
        for idx, (c_name, _, _) in enumerate(col_bounds):
            cn_lower = c_name.lower()
            if any(k in cn_lower for k in ["desc", "particular", "item", "product", "name"]):
                desc_col_idx = idx
            elif any(k in cn_lower for k in ["unit", "pack"]):
                unit_col_idx = idx
            elif any(k in cn_lower for k in ["qty", "free", "rate", "amount", "val", "bal", "open", "clos", "sale", "pur", "rec", "iss", "dump", "total", "in", "out", "cr", "dr", "(%)", "%"]):
                numeric_col_indices.append(idx)

        in_table = True
        # Set when a bare total label has already been consumed by the numeric
        # line above it (see the look-ahead below), so it is not emitted twice.
        skip_next_line = False

        for i in range(hdr_end_idx + 1, len(lines)):
            if skip_next_line:
                skip_next_line = False
                continue
            line = lines[i]
            lt_lower = line["text"].lower()

            # Ignore system/report headers that might appear after table.
            # "Page 1 of 1" is pagination, not a product - it was landing in a
            # value column as table data.
            if any(k in lt_lower for k in ["end of report", "page no >", "page no:"]) \
                    or re.search(r'page\s+\d+\s+of\s+\d+', lt_lower):
                continue

            # Detect the ERP footer / application chrome below the table.
            #
            # A hardcoded vendor vocabulary ("marg erp", a specific dealer name)
            # only recognises footers someone has already seen, and misses the
            # same screen from a different installation - so the UI status bar
            # and the dealer's contact block get parsed as product rows, putting
            # things like "JLine:31" into the Packing column. These structural
            # signals describe what a footer IS instead: application key hints,
            # contact details, and web/email addresses never appear in a stock
            # table's own rows.
            has_contact = bool(re.search(r'(?:e-?mail|mobile|phone|help\s*line|tel)\s*[:.]', lt_lower))
            has_address = bool(re.search(r'[\w.+-]+@[\w-]+\.[\w.]+|www\.|\.com\b', lt_lower))
            has_ui_keys = bool(re.search(r'\b(?:f\d{1,2}|esc|alt|ctrl)\s*[-:=]\s*\w', lt_lower))
            has_ui_words = bool(re.search(r'\b(?:enter-edit|tab-line|jline|authorised user|authonsed user)\b', lt_lower))
            # Vendor advertising strips printed under the grid, e.g.
            # "MARGERPNANORs 5550|ManageStockAccountsGSTBarcodeing|Cal 0161...".
            # A run of 10+ digits is a phone number, never a stock quantity, and
            # pipe-separated marketing copy is not a table row.
            has_phone_run = bool(re.search(r'\d{10,}', line["text"]))
            has_marketing = lt_lower.count("|") >= 2 or "margerp" in lt_lower.replace(" ", "")
            # Print stamps under the grid: "Printed By : KNM 1 - COUNTER 1,
            # 06-07-2026 15:45:59".
            #
            # A broader "two or more label: value pairs means summary" rule was
            # tried here and reverted: batch/expiry continuation lines have
            # exactly that shape ("BATCH: 25S2GCA517 EXP: 10/26") and are real
            # table data, so the rule deleted them. Kept narrow deliberately.
            has_print_stamp = bool(re.search(r'\bprint(?:ed)?\s*(?:by|on)\b', lt_lower))
            is_erp_footer = (
                any(f in lt_lower for f in ["help find", "heup find", "authorised marg", "helpline:", "masterpartner", "marg erp", "support@marg"]) or
                ("find" in lt_lower and any(h in lt_lower for h in ["help", "heup", "?", "\uff1f", "f1", "f2"])) or
                has_contact or has_address or has_ui_keys or has_ui_words
                or has_phone_run or has_marketing or has_print_stamp
            )
            if in_table and is_erp_footer:
                in_table = False
                footer_lines.append(line["text"])
                continue


            if not in_table:
                footer_lines.append(line["text"])
                continue

            row_cells, row_meta = self._project_tokens_to_columns(line["tokens"], col_bounds)
            if not any(row_cells):
                continue

            ruling_blocks_merge = bool(grid_rows) and _ruling_between(
                last_line_y1[-1] if last_line_y1 else None, line.get("y0", 0.0)
            )

            num_filled = sum(1 for ni in numeric_col_indices if row_cells[ni].strip())
            starts_index = bool(re.match(r'^\*?\d{1,3}\b', line["text"].strip()))
            is_subtotal = any(k in lt_lower for k in ["s-total", "sub-total", "subtotal"]) and not any(k in lt_lower for k in ["grand total", "net total", "total:"])
            is_total = any(k in lt_lower for k in ["grand total", "net total", "totals:", "total"]) and not is_subtotal
            has_desc = bool(row_cells[desc_col_idx].strip())
            is_batch_or_exp = bool(re.search(r'\b(batch|exp|expiry|mrp|within\s+six|quantity)\b', lt_lower))

            # If preceding product row had 0 numbers, this line completes the product row, not a subtotal
            if is_subtotal and grid_rows:
                prev_row = grid_rows[-1]
                prev_num_filled = sum(1 for ni in numeric_col_indices if prev_row[ni].strip())
                if prev_num_filled == 0:
                    is_subtotal = False

            # Division/category header banner line printed between the column header and
            # the first product: a structural signal (position before any product row +
            # zero numeric fill), not a company/product name list - a hardcoded vocabulary
            # like "hetero"/"derma"/"glow" would misfire (drop a real row for that specific
            # distributor, or fail to generalize to any other). Preserved as its own
            # PARTY_OR_HEADER row instead of silently dropped.
            if not grid_rows and num_filled == 0 and not starts_index:
                if any(k in lt_lower for k in ["division", "statement", "analysis", "stock & sales"]):
                    row_cells[desc_col_idx] = line["text"].strip()
                    row_meta[desc_col_idx]["raw"] = row_cells[desc_col_idx]
                    row_meta[desc_col_idx]["row_type"] = PARTY_OR_HEADER
                    grid_rows.append(row_cells)
                    grid_metadata.append(row_meta)
                    last_line_y1.append(line.get("y1"))
                    continue

            # Align total row label to description column if placed in adjacent column (e.g. PACK or SL NO)
            if is_total:
                found_label = False
                for c_i, c_val in enumerate(row_cells):
                    if any(t_w in c_val.lower() for t_w in ["total:", "grand total", "total"]):
                        row_cells[desc_col_idx] = c_val.strip()
                        if c_i != desc_col_idx:
                            row_cells[c_i] = ""
                        has_desc = True
                        found_label = True
                        break
                if not found_label:
                    if "grand total:" in lt_lower:
                        row_cells[desc_col_idx] = "Grand Total:"
                    elif "grand total" in lt_lower:
                        row_cells[desc_col_idx] = "Grand Total"
                    else:
                        row_cells[desc_col_idx] = "TOTAL"
                    has_desc = True

            # Subtotal row should describe the preceding product being totaled
            elif is_subtotal and grid_rows:
                prev_row = grid_rows[-1]
                prev_desc = prev_row[desc_col_idx].strip()
                cur_desc = row_cells[desc_col_idx].strip()
                if prev_desc and (not cur_desc or cur_desc.lower() != prev_desc.lower()):
                    row_cells[desc_col_idx] = prev_desc
                    has_desc = True

            # Tilt-fusion recovery: if previous row is a subtotal
            # and current line starts a new product description on the left with blank Opening,
            # but has transaction numbers (In, Out, Balance) on the right:
            # transfer those transaction numbers to complete the preceding subtotal.
            if grid_rows:
                prev_row = grid_rows[-1]
                prev_is_subtotal = any(k in prev_row[desc_col_idx].lower() or k in " ".join(prev_row).lower() for k in ["s-total", "subtotal", "total"])
                if prev_is_subtotal and len(numeric_col_indices) >= 2:
                    op_col = numeric_col_indices[0]
                    trans_cols = numeric_col_indices[1:]
                    cur_desc_lower = row_cells[desc_col_idx].lower().strip()
                    cur_is_subtotal = any(k in cur_desc_lower or k in lt_lower for k in ["s-total", "subtotal", "total"])
                    cur_has_trans = (
                        any(row_cells[tc].strip() for tc in trans_cols) and
                        not row_cells[op_col].strip() and
                        has_desc and not cur_is_subtotal
                    )
                    if cur_has_trans:
                        for tc in trans_cols:
                            c_val_cur = row_cells[tc].strip()
                            if c_val_cur:
                                prev_row[tc] = c_val_cur
                                grid_metadata[-1][tc] = row_meta[tc]
                                row_cells[tc] = ""
                                row_meta[tc] = {"raw": "", "normalized": None, "semantic": "EMPTY", "tokens": []}
                        if not prev_row[op_col].strip():
                            prev_row[op_col] = "0.000"
                        num_filled = sum(1 for ni in numeric_col_indices if row_cells[ni].strip())

            # A total's figures can be printed on the line ABOVE the word TOTAL
            # (the label sits lower than its own numbers on some ERP screens).
            # Read in order, those figures look like a continuation of the last
            # product row - which both corrupts that product and leaves the total
            # empty. Look ahead one line: if the next line is nothing but a total
            # label, these numbers belong to it.
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            next_is_bare_total_label = False
            if next_line is not None and not has_desc and num_filled > 0:
                nl = next_line["text"].strip().lower()
                next_is_bare_total_label = (
                    len(nl) <= 16
                    and not re.search(r'\d', nl)
                    and any(k in nl for k in ("grand total", "net total", "total"))
                )
            if next_is_bare_total_label:
                row_cells[desc_col_idx] = next_line["text"].strip()
                row_meta[desc_col_idx]["raw"] = row_cells[desc_col_idx]
                grid_rows.append(row_cells)
                grid_metadata.append(row_meta)
                last_line_y1.append(line.get("y1"))
                skip_next_line = True
                continue

            # Handle lines with no description but filled numeric columns
            if grid_rows and not has_desc and num_filled > 0 and not is_subtotal:
                prev_row = grid_rows[-1]
                prev_num_filled = sum(1 for ni in numeric_col_indices if prev_row[ni].strip())
                prev_desc = prev_row[desc_col_idx].strip()
                prev_is_total = any(k in prev_row[desc_col_idx].lower() for k in ["grand total", "net total", "totals:", "total"])
                prev_is_subtotal = any(k in prev_row[desc_col_idx].lower() or k in " ".join(prev_row).lower() for k in ["s-total", "subtotal", "total"])

                # Case 1: Preceding row was a Total row split across multiple lines (e.g. Grand Total)
                if prev_is_total:
                    for c_i in range(len(row_cells)):
                        if not prev_row[c_i].strip() and row_cells[c_i].strip():
                            prev_row[c_i] = row_cells[c_i]
                            grid_metadata[-1][c_i] = row_meta[c_i]
                    in_table = False
                    continue

                # Case 2: Preceding row was missing numeric cells that current line supplies
                can_fill_prev = any(not prev_row[ni].strip() and row_cells[ni].strip() for ni in numeric_col_indices)
                if can_fill_prev:
                    for c_i in range(len(row_cells)):
                        if not prev_row[c_i].strip() and row_cells[c_i].strip():
                            prev_row[c_i] = row_cells[c_i]
                            grid_metadata[-1][c_i] = row_meta[c_i]
                    continue

                # Case 3: Preceding row was a product row, and current line is the subtotal line
                if not prev_is_subtotal and prev_num_filled > 0:
                    row_cells[desc_col_idx] = prev_desc
                    if unit_col_idx is not None and not row_cells[unit_col_idx].strip():
                        row_cells[unit_col_idx] = "S-Total"
                    has_desc = True
                    is_subtotal = True
                else:
                    # Duplicate or ghost line
                    continue

            # Check if this line is continuation: description wrapping, batch/expiry annotations, or short broken text
            is_continuation = False
            # `is_total` matters as much as `is_subtotal` here: a bare "TOTAL"
            # label is short and carries no digits, so without this guard it
            # satisfies the continuation test and gets glued onto the last
            # product row ("VETORY P TAB TOTAL"), destroying both that product
            # and the total row.
            if grid_rows and not is_subtotal and not is_total and not starts_index and not ruling_blocks_merge:
                is_pack_suffix = bool(re.match(r'^\s*\d+\s*(?:s|tab|cap|ml|gm)s?\s*$', lt_lower))
                is_batch_word = bool(re.search(r'\b(batch|exp|expiry)\b', lt_lower))
                if is_batch_or_exp or (num_filled == 0 and (len(line["text"].strip()) < 8 or is_pack_suffix or is_batch_word)):
                    is_continuation = True

            if is_continuation and grid_rows:
                # Merge continuation into previous logical row
                if has_desc:
                    grid_rows[-1][desc_col_idx] = (grid_rows[-1][desc_col_idx] + " " + row_cells[desc_col_idx]).strip()
                    grid_metadata[-1][desc_col_idx]["raw"] = grid_rows[-1][desc_col_idx]
                    grid_metadata[-1][desc_col_idx]["tokens"].extend(row_meta[desc_col_idx].get("tokens", []))
                if last_line_y1:
                    last_line_y1[-1] = line.get("y1", last_line_y1[-1])
                continue

            # Check for duplicate title, pending title completion, or multi-line product merging
            if grid_rows:
                prev_row = grid_rows[-1]
                prev_num_filled = sum(1 for ni in numeric_col_indices if prev_row[ni].strip())
                prev_desc = prev_row[desc_col_idx].strip()
                cur_desc = row_cells[desc_col_idx].strip()
                prev_is_subtotal = any(k in prev_row[desc_col_idx].lower() or k in " ".join(prev_row).lower() for k in ["s-total", "subtotal", "total"])

                norm_prev = re.sub(r'[^a-zA-Z0-9]', '', prev_desc).lower()
                norm_cur = re.sub(r'[^a-zA-Z0-9]', '', cur_desc).lower()
                # Length-ratio-gated similarity, not pure substring containment - a plain
                # "norm_prev in norm_cur" check would wrongly merge genuinely different
                # products that share a prefix, e.g. "PARA500" vs "PARA500XR" or
                # "BORIT SB 65" vs "BORIT SB 130" (the latter pair already fails
                # containment, but many real product-code families do not).
                is_same_product = bool(
                    _description_similarity(norm_prev, norm_cur) >= 0.85
                    and not ruling_blocks_merge
                )

                # Case A: Duplicate title line with 0 numbers - merge non-empty fields (e.g. Sn.) into prev_row
                if prev_num_filled == 0 and num_filled == 0 and not is_subtotal and not prev_is_subtotal and is_same_product:
                    for c_i in range(len(row_cells)):
                        if not prev_row[c_i].strip() and row_cells[c_i].strip():
                            prev_row[c_i] = row_cells[c_i]
                            grid_metadata[-1][c_i] = row_meta[c_i]
                        elif row_meta[c_i].get("tokens"):
                            grid_metadata[-1][c_i]["tokens"].extend(row_meta[c_i]["tokens"])
                    if last_line_y1:
                        last_line_y1[-1] = line.get("y1", last_line_y1[-1])
                    continue

                # Case B: Preceding row had title with 0 numbers, current line brings the numbers
                if prev_num_filled == 0 and num_filled > 0 and not prev_is_subtotal and is_same_product:
                    for c_i in range(len(row_cells)):
                        if not row_cells[c_i].strip() and prev_row[c_i].strip():
                            row_cells[c_i] = prev_row[c_i]
                            row_meta[c_i] = grid_metadata[-1][c_i]
                        elif c_i < len(grid_metadata[-1]):
                            prev_toks = grid_metadata[-1][c_i].get("tokens", [])
                            if prev_toks:
                                row_meta[c_i]["tokens"] = prev_toks + row_meta[c_i].get("tokens", [])
                    if unit_col_idx is not None and row_cells[unit_col_idx].strip().lower() == "s-total" and not is_subtotal:
                        row_cells[unit_col_idx] = ""
                        row_meta[unit_col_idx] = {"raw": "", "normalized": None, "semantic": "EMPTY", "tokens": []}
                    grid_rows[-1] = row_cells
                    grid_metadata[-1] = row_meta
                    if last_line_y1:
                        last_line_y1[-1] = line.get("y1", last_line_y1[-1])
                    continue

                # Case C: Same product split across 2 print lines (Line 1 had Op/In, Line 2 had Out/Balance)
                if prev_num_filled > 0 and num_filled > 0 and not is_subtotal and not prev_is_subtotal and is_same_product and not starts_index:
                    for c_i in range(len(row_cells)):
                        if not prev_row[c_i].strip() and row_cells[c_i].strip():
                            prev_row[c_i] = row_cells[c_i]
                            grid_metadata[-1][c_i] = row_meta[c_i]
                        elif row_meta[c_i].get("tokens"):
                            grid_metadata[-1][c_i]["tokens"].extend(row_meta[c_i]["tokens"])
                    if last_line_y1:
                        last_line_y1[-1] = line.get("y1", last_line_y1[-1])
                    continue

            grid_rows.append(row_cells)
            grid_metadata.append(row_meta)
            last_line_y1.append(line.get("y1"))

        return grid_rows, grid_metadata, footer_lines

    def _reconstruct_table_grid(
        self,
        lines: List[Dict[str, Any]],
        rulings: Dict[str, List[Tuple[float, float, float, float]]],
        img_width: int,
        img: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, List[str]], List[Tuple[str, float, float]], List[List[float]], List[float]]:
        """
        Reconstructs multi-tier stacked headers, defines strict column intervals,
        and projects tokens into logical table rows.
        """
        non_table = {"header_lines": [], "footer_lines": []}
        if len(lines) < 2:
            return None, {"header_lines": [l["text"] for l in lines], "footer_lines": []}, [], [], []

        def is_hdr_tok(txt):
            """
            Legacy vocabulary test. Used ONLY as a fallback when the structural
            detector below cannot find the header - see the note on
            HEADER_KEYWORDS. Kept because a known word is still evidence when
            geometry has none, not because the pipeline depends on it.
            """
            clean = re.sub(r'[^a-z]', '', txt.lower())
            if not clean:
                return False
            # Some keywords are themselves common substrings of ordinary words
            # and must match a whole token, or they fire everywhere: "op" inside
            # "OPOX" promotes a product row into the header band, and "age"
            # inside "Page" turns a page number into a column label.
            return any(
                clean == k if (len(k) <= 2 or k in self.WHOLE_TOKEN_KEYWORDS) else k in clean
                for k in self.HEADER_KEYWORDS
            )

        # 1. Identify start of table header band.
        #
        # Structurally first, and on its own terms: find where the numeric body
        # begins, measure the vertical bands the body's figures form, and take
        # the line above it whose tokens line up with those bands. This reads a
        # header whose labels no fixture has ever contained - "Qoh", "O.Stk",
        # "Liq Days" - and equally a header the OCR has garbled, because it
        # never asks what the labels SAY, only where they sit.
        body_start = self._find_body_start(lines)
        body_anchors = self._column_anchors(lines[body_start:]) if body_start is not None else []
        hdr_start = self._detect_header_line(lines, body_start)
        if hdr_start is not None:
            # ascii() not !r: a garbled header can carry CJK glyphs (the CJK-
            # trained recogniser's guesses), and those cannot be encoded to the
            # cp1252 console this runs on - the log record then dies instead of
            # the message being written.
            logger.debug(
                f"Header row located structurally at line {hdr_start}: "
                f"{ascii(lines[hdr_start]['text'])} (body starts at line {body_start})"
            )

        # Fallback: no usable geometry (too few numeric rows to form bands, or
        # nothing above the body aligns). Only here does wording get a vote.
        if hdr_start is None:
            for idx, line in enumerate(lines[:15]):
                lt = line["text"].lower()
                # A statement-period line is report metadata, never the column
                # header row - even when it carries a couple of right-hand
                # column labels ("From 01/05/2026 To 29/05/2026  Value  Age").
                # Anchoring the band on it turns the date text into a column
                # name and shifts every column right. Those stray labels are
                # still recovered below, beyond the real header's right edge.
                if self._looks_like_report_metadata(lt):
                    continue
                hdr_count = sum(1 for t in line["tokens"] if is_hdr_tok(t["text"]))
                if hdr_count >= 2:
                    hdr_start = idx
                    break

        if hdr_start is None:
            multi_token_lines = [i for i, l in enumerate(lines) if len(l["tokens"]) >= 4]
            if multi_token_lines:
                hdr_start = multi_token_lines[0]

        if hdr_start is None:
            return None, {"header_lines": [l["text"] for l in lines], "footer_lines": []}, [], [], []

        # 1b. Extend the band UPWARD onto super-header tiers.
        #
        # A stacked header names each column across two printed rows:
        #
        #        Op.   Pur   Pur     Sale  Sale  ...  Bal.
        #   SlNo  Qty   Qty   F.Qty   Qty   F.Qty ...  Qty
        #
        # The detector anchors on the LEAF tier (it is the one that lines up
        # with every band), and the band only ever grew downward - so the tier
        # above was never read. Six different columns then come back named just
        # "Qty", which is worse than useless as a key: nothing downstream can
        # tell opening from purchase from sale, and role mapping gives up and
        # emits nulls for every quantity.
        #
        # A super tier qualifies on the same evidence as a lower one, one notch
        # stricter: no figures at all, several tokens, and labels standing over
        # at least TWO bands. Letterhead lines above the header are one-token
        # address or title lines and fail that, which is the point of the
        # stricter bar on this side.
        hdr_top = hdr_start
        for k in range(hdr_start - 1, max(-1, hdr_start - 3), -1):
            l = lines[k]
            toks = l.get("tokens") or []
            if len(toks) < 2 or not body_anchors:
                break
            if any(self._is_numeric_token(t["text"]) for t in toks):
                break
            if self._looks_like_report_metadata(l.get("text", "")):
                break
            covered = sum(
                1 for a in body_anchors
                if any(t["x0"] - 45.0 <= a <= t["x1"] + 45.0 for t in toks)
            )
            if covered < 2:
                break
            hdr_top = k
            logger.debug(
                f"Absorbed super-header tier at line {k}: {ascii(l.get('text', ''))}"
            )

        # Collect pre-table banner lines
        for i in range(hdr_top):
            line_text = lines[i]["text"]
            if i == 0:
                non_table["header_lines"].append(f"# {line_text}")
            elif i == 1:
                non_table["header_lines"].append(f"### {line_text}")
            else:
                non_table["header_lines"].append(line_text)

        # 2. Identify end of multi-tier header band.
        #
        # The band stops where the data begins, and the data announces itself:
        # the first line of the numeric body ends it, full stop. That single
        # structural bound replaces the old keyword-count rules, which had to
        # guess whether one fuzzy keyword hit meant "second header tier" or
        # "first product row" - and when it guessed wrong the product row was
        # deleted and a column's name corrupted with that product's text.
        hdr_end = hdr_start
        for j in range(hdr_start + 1, min(len(lines), hdr_start + 4)):
            if body_start is not None and j >= body_start:
                break
            l = lines[j]
            num_numeric = sum(1 for t in l["tokens"] if self._is_numeric_token(t["text"]))
            # A further header tier carries labels and NOTHING else - no
            # figures, not even a dash placeholder - and at least one of those
            # labels stands over a band the data forms. Both halves are needed,
            # and each was learned from a document the other half got wrong:
            #
            #   "DESCRIPTION QTY."   0 figures, 1 band  -> a real second tier
            #   "HETERO-DERMA GLOW"  0 figures, 0 bands -> a division banner row
            #   "ADABOR GEL  -"      1 figure           -> a real product row
            #
            # Absorbing either of the last two deletes a row from the table and
            # drags its words into a column name; rejecting the first loses the
            # tier that names half the columns.
            if body_anchors:
                if num_numeric > 0:
                    break
                covered = sum(
                    1 for a in body_anchors
                    if any(t["x0"] - 45.0 <= a <= t["x1"] + 45.0 for t in l["tokens"])
                )
                if covered < 1:
                    break
            elif num_numeric >= 2:
                break
            if body_start is None:
                # No measured body to stop against: fall back to the old
                # evidence test rather than absorbing lines blindly.
                if sum(1 for t in l["tokens"] if is_hdr_tok(t["text"])) < 1 and len(l["tokens"]) < 3:
                    break
            if not l["tokens"]:
                break
            hdr_end = j

        hdr_lines = lines[hdr_top:hdr_end + 1]
        data_lines = lines[hdr_end + 1:]

        # The band only ever grows DOWNWARD from the row it anchored on, but the
        # right-hand columns are often labelled on a line ABOVE it, sharing that
        # line with the report metadata:
        #
        #     From 01/05/2026 To 29/05/2026                Value      Age
        #     Product Name   Packing  O.Stk  Purc  Tot  Sale  Qoh
        #
        # Those labels were therefore never seen, so no column was built for
        # them: their figures fused into the last real column ("26 1146.18" in
        # Qoh) and the final column's data was discarded entirely.
        #
        # A token qualifies by evidence, not by wording: it must be textual, sit
        # beyond the main header's right edge, and stand above a band the data
        # itself demonstrably forms out there. That is why the date text sharing
        # the line stays behind - nothing is stacked under it - and it is also
        # why labels nobody listed ("Value", "Age", "Liq") are now recovered on
        # their own merits instead of needing to be added to a keyword list
        # after each new document goes wrong.
        main_right = max((t["x1"] for l in hdr_lines for t in l["tokens"]), default=0.0)
        unclaimed_bands = [a for a in self._column_anchors(data_lines) if a > main_right - 25.0]
        trailing_hdr_tokens = [
            t
            for k in range(max(0, hdr_top - 3), hdr_top)
            for t in lines[k]["tokens"]
            if t["x0"] >= main_right - 10.0
            and not self._is_numeric_token(t["text"])
            and any(t["x0"] - 60.0 <= a <= t["x1"] + 60.0 for a in unclaimed_bands)
        ]
        if trailing_hdr_tokens:
            logger.debug(
                "Recovered right-hand header labels printed above the header row: "
                f"{[t['text'] for t in trailing_hdr_tokens]}"
            )
            hdr_lines = hdr_lines + [{
                "tokens": trailing_hdr_tokens,
                "text": " ".join(t["text"] for t in trailing_hdr_tokens),
                "x0": min(t["x0"] for t in trailing_hdr_tokens),
                "x1": max(t["x1"] for t in trailing_hdr_tokens),
                "y0": min(t.get("y0", 0.0) for t in trailing_hdr_tokens),
                "y1": max(t.get("y1", 0.0) for t in trailing_hdr_tokens),
            }]

        hdr_bbox = [
            min(l["x0"] for l in hdr_lines),
            min(l["y0"] for l in hdr_lines),
            max(l["x1"] for l in hdr_lines),
            max(l["y1"] for l in hdr_lines)
        ]

        # 3. Build Global Column Model via Multi-Evidence Fusion
        col_bounds, column_model_uncertain = self._build_global_column_model(hdr_lines, data_lines, rulings, img_width)
        if len(col_bounds) < 2:
            return None, {"header_lines": [l["text"] for l in lines], "footer_lines": []}, [], [], hdr_bbox

        # 4. Assemble logical rows and consolidate continuation lines
        grid_rows, grid_metadata, extra_footers = self._assemble_logical_rows(
            lines, hdr_end, col_bounds, rulings
        )
        non_table["footer_lines"].extend(extra_footers)

        logical_row_bboxes = []
        for r_meta in grid_metadata:
            r_tokens = [tok for cell in r_meta for tok in cell.get("tokens", [])]
            if r_tokens:
                logical_row_bboxes.append([
                    min(t["x0"] for t in r_tokens),
                    min(t["y0"] for t in r_tokens),
                    max(t["x1"] for t in r_tokens),
                    max(t["y1"] for t in r_tokens)
                ])
            else:
                logical_row_bboxes.append([0.0, 0.0, float(img_width), 0.0])

        # Enforce non-overlapping vertical intervals between adjacent logical rows
        for r_i in range(len(logical_row_bboxes) - 1):
            if logical_row_bboxes[r_i][3] > logical_row_bboxes[r_i + 1][1]:
                split_y = (logical_row_bboxes[r_i][3] + logical_row_bboxes[r_i + 1][1]) / 2.0
                logical_row_bboxes[r_i][3] = split_y
                logical_row_bboxes[r_i + 1][1] = split_y

        # 4b. Fine-grained cell refinement: dash vs empty, multi-variant OCR, numeric disambiguation
        if img is not None:
            from app.services.cell_ocr_service import cell_ocr_service
            grid_rows, grid_metadata = cell_ocr_service.refine_table_cells(
                image=img,
                col_bounds=col_bounds,
                logical_row_bboxes=[(bb[1], bb[3]) for bb in logical_row_bboxes],
                grid_rows=grid_rows,
                grid_meta=grid_metadata
            )

        # Invented placeholder columns that ended up holding nothing.
        #
        # These are also dropped earlier on geometry, but a token can sit inside
        # a placeholder's x-range while projection assigns it to the neighbouring
        # column, and refinement can blank a cell that looked filled at assembly
        # time. Only the finished grid is definitive: a column with no value in
        # any row is not a column, and leaving it in shifts every column index
        # for whoever consumes the JSON.
        empty_placeholders = {
            ci for ci, (cname, _x0, _x1) in enumerate(col_bounds)
            if re.match(r'^Col_Num_\d+_\d+$', cname or "")
            and not any(ci < len(r) and (r[ci] or "").strip() for r in grid_rows)
        }
        if empty_placeholders and len(col_bounds) - len(empty_placeholders) >= 2:
            logger.debug(f"Dropped {len(empty_placeholders)} invented column(s) holding no data in any row")
            col_bounds = [c for ci, c in enumerate(col_bounds) if ci not in empty_placeholders]
            grid_rows = [[v for ci, v in enumerate(r) if ci not in empty_placeholders] for r in grid_rows]
            grid_metadata = [[m for ci, m in enumerate(rm) if ci not in empty_placeholders] for rm in grid_metadata]

        # 5. Build Markdown Representation
        col_names = [c[0] for c in col_bounds]
        md_lines = []
        md_lines.append("| " + " | ".join(col_names) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(col_names)) + " |")

        for r in grid_rows:
            escaped_r = [c.replace("|", "/") for c in r]
            md_lines.append("| " + " | ".join(escaped_r) + " |")

        table_dict: Dict[str, Any] = {
            "columns": col_names,
            "rows": grid_rows,
            "cells_metadata": grid_metadata,
            "markdown": "\n".join(md_lines),
            "column_model_uncertain": column_model_uncertain
        }

        return table_dict, non_table, col_bounds, logical_row_bboxes, hdr_bbox

    def _project_tokens_to_columns(
        self,
        tokens: List[Dict[str, Any]],
        col_bounds: List[Tuple[str, float, float]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Projects tokens into column boundaries.
        Distinguishes blank cells (''), explicit zeros ('0'), and dashes ('-').
        """
        num_cols = len(col_bounds)
        cells = [""] * num_cols
        cells_meta: List[Dict[str, Any]] = [
            {"raw": "", "normalized": None, "semantic": "EMPTY", "tokens": []} for _ in range(num_cols)
        ]

        from app.services.cell_ocr_service import cell_ocr_service
        tokens = cell_ocr_service.split_spanning_tokens(tokens, col_bounds)

        for tok in tokens:
            x_mid = (tok["x0"] + tok["x1"]) / 2.0
            matched_col = None

            for c_idx, (c_name, b_left, b_right) in enumerate(col_bounds):
                if b_left <= x_mid < b_right:
                    matched_col = c_idx
                    break

            if matched_col is not None:
                existing = cells[matched_col]
                tok_text = tok["text"].strip()
                cells[matched_col] = (existing + " " + tok_text).strip() if existing else tok_text
                cells_meta[matched_col]["tokens"].append(tok)

        # Normalize cell semantics
        for c_idx in range(num_cols):
            raw_val = cells[c_idx].strip()
            cells_meta[c_idx]["raw"] = raw_val
            c_name = col_bounds[c_idx][0]
            is_num_col = any(k in c_name.lower() for k in ["qty", "free", "rate", "price", "amount", "amt", "val", "bal", "open", "clos", "sale", "pur", "rec", "iss", "dump", "total", "in", "out", "cr", "dr", "(%)", "%"]) or "decimal" in c_name.lower()

            # Dot-matrix multi-zero representations (e.g. '0000', '000o', '0000E', '000:0', '0010', '0.00', '000')
            is_dot_matrix_zero = (
                is_num_col and (
                    raw_val in ["0", "0.0", "0.00", "0.000", "0000", "000", "000o", "000O", "0000E", "0000e", "0010", "000:0", "000.0", "000.00", "OOO0", "OOOPE"] or
                    bool(re.match(r'^[0oO:.\s]{2,6}$', raw_val)) or
                    bool(re.match(r'^(?:0{2,5}|000[oOeE]|0010|000:0)$', raw_val, re.IGNORECASE))
                )
            )

            if not raw_val:
                cells_meta[c_idx]["semantic"] = "EMPTY"
                cells_meta[c_idx]["normalized"] = None
                cells_meta[c_idx]["status"] = "EMPTY"
                cells[c_idx] = ""
            elif raw_val in ["-", "--", "---", "NA", "N/A", "一", "—", "–"]:
                cells_meta[c_idx]["semantic"] = "NOT_REPORTED"
                cells_meta[c_idx]["normalized"] = None
                cells_meta[c_idx]["status"] = "DASH"
                cells[c_idx] = "-"
            elif is_dot_matrix_zero:
                cells_meta[c_idx]["semantic"] = "ZERO"
                cells_meta[c_idx]["normalized"] = 0.0
                cells_meta[c_idx]["status"] = "VALUE"
                is_decimal_col = any(k in c_name.lower() for k in ["open", "in", "out", "bal", "clos", "rec", "iss", "amount", "amt", "val", "rate"])
                cells[c_idx] = "0.000" if is_decimal_col else "0"
            else:
                try:
                    num = float(raw_val.replace(",", ""))
                    cells_meta[c_idx]["semantic"] = "NUMERIC"
                    cells_meta[c_idx]["normalized"] = num
                    cells_meta[c_idx]["status"] = "VALUE"
                except ValueError:
                    cells_meta[c_idx]["semantic"] = "TEXT"
                    cells_meta[c_idx]["normalized"] = raw_val
                    cells_meta[c_idx]["status"] = "VALUE"

        return cells, cells_meta

    def _export_geometry_debug_images(
        self,
        img: np.ndarray,
        table_bbox: List[float],
        hdr_bbox: List[float],
        row_bboxes: List[List[float]],
        col_bounds: List[Tuple[str, float, float]],
        tokens: List[Dict[str, Any]],
        fname_prefix: str
    ) -> None:
        """
        Exports geometry debug visual inspection images to outputs/ directory.
        """
        try:
            out_dir = Path("outputs")
            out_dir.mkdir(parents=True, exist_ok=True)

            # 1. debug_table_bbox.png
            img_table = img.copy()
            if table_bbox and len(table_bbox) == 4:
                cv2.rectangle(
                    img_table,
                    (int(table_bbox[0]), int(table_bbox[1])),
                    (int(table_bbox[2]), int(table_bbox[3])),
                    (255, 0, 0), 3
                )
            cv2.imwrite(str(out_dir / f"{fname_prefix}_debug_table_bbox.png"), img_table)

            # 2. debug_header_bbox.png
            img_hdr = img.copy()
            if hdr_bbox and len(hdr_bbox) == 4:
                cv2.rectangle(
                    img_hdr,
                    (int(hdr_bbox[0]), int(hdr_bbox[1])),
                    (int(hdr_bbox[2]), int(hdr_bbox[3])),
                    (0, 255, 255), 3
                )
            cv2.imwrite(str(out_dir / f"{fname_prefix}_debug_header_bbox.png"), img_hdr)

            # 3. debug_row_lines.png
            img_rows = img.copy()
            for rb in row_bboxes:
                y_mid = int((rb[1] + rb[3]) / 2.0)
                cv2.line(img_rows, (0, y_mid), (img.shape[1], y_mid), (0, 0, 255), 1)
            cv2.imwrite(str(out_dir / f"{fname_prefix}_debug_row_lines.png"), img_rows)

            # 4. debug_column_lines.png
            img_cols = img.copy()
            for _, lb, rb in col_bounds:
                cv2.line(img_cols, (int(lb), 0), (int(lb), img.shape[0]), (0, 255, 0), 2)
            cv2.imwrite(str(out_dir / f"{fname_prefix}_debug_column_lines.png"), img_cols)

            # 5. debug_cell_assignment.png
            img_cells = img.copy()
            colors = [
                (255, 100, 100), (100, 255, 100), (100, 100, 255),
                (255, 255, 100), (255, 100, 255), (100, 255, 255),
                (200, 150, 50), (50, 200, 150), (150, 50, 200), (200, 200, 200)
            ]
            for t in tokens:
                x_mid = t["xc"] if "xc" in t else (t["x0"] + t["x1"]) / 2.0
                assigned_col = 0
                for c_idx, (_, lb, rb) in enumerate(col_bounds):
                    if lb <= x_mid < rb:
                        assigned_col = c_idx
                        break
                color = colors[assigned_col % len(colors)]
                cv2.rectangle(img_cells, (int(t["x0"]), int(t["y0"])), (int(t["x1"]), int(t["y1"])), color, 1)
            cv2.imwrite(str(out_dir / f"{fname_prefix}_debug_cell_assignment.png"), img_cells)

        except Exception as e:
            logger.warning(f"Failed to export geometry debug images: {e}")


table_ocr_service = TableOCRService()
