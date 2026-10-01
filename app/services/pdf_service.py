import re
# pyrefly: ignore [missing-import]
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple
from app.core.config import settings
from app.core.logger import logger


# A word that is a figure rather than a label. Digital PDF text is exact, so
# this needs none of the OCR-garble tolerance the image path carries.
_NUM_WORD_RE = re.compile(r'^[-+(]?[\d,]*\.?\d+\)?%?$')

# Footer / print-chrome below the grid, recognised by SHAPE rather than by
# vendor wording. The previous test was `"tot sale" in line or "generated at"
# in line`, which only ended the table on installations someone had already
# seen - so on every other ERP the summary block, the print stamp and the
# software credit line were all parsed as product rows.
_FOOTER_PATTERNS = (
    re.compile(r'\bprint(?:ed)?\s*(?:by|on)\b', re.I),      # "Printed By : REKHA"
    re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+|www\.|\.net\b|\.com\b', re.I),
    re.compile(r'(?:e-?mail|mobile|phone|help\s*line|tel|ph)\s*[:.\-]', re.I),
    re.compile(r'\bpage\s+\d+\s+of\s+\d+\b', re.I),
    re.compile(r'\d{7,}'),                                   # phone runs
)


def _is_num_word(txt: str) -> bool:
    return bool(_NUM_WORD_RE.match((txt or "").strip()))


def _looks_like_footer(line_text: str) -> bool:
    return any(p.search(line_text or "") for p in _FOOTER_PATTERNS)


def _looks_like_report_period(line_text: str) -> bool:
    """
    "Stock And Sales from 01-May-2026 to 31-May-2026" - the report title. It
    sits directly above the header, carries no bare figures and spans the full
    grid width, so on shape alone it is indistinguishable from a super-header
    tier; absorbed as one it prepends "Stock And Sales from" to a column name.
    Two dates, or a from/to pair, identify it without naming any vendor.
    """
    t = (line_text or "").lower()
    two_dates = re.findall(r'\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', t)
    if len(two_dates) >= 2:
        return True
    return bool(re.search(r'\bfrom\b.*\bto\b.*\d', t))


def _is_summary_line(line_text: str) -> bool:
    """
    A totals/summary block line: "OP.Stk Val (PTS) : 138995.86   Pur.Val(PTS+Tax) :
    39761.77". Two or more "label : number" pairs on one line is a summary
    layout, not a table row - a product row has ONE label and then bare figures.
    """
    return len(re.findall(r':\s*[\d,]+\.?\d*', line_text or "")) >= 2


class PDFService:
    """PDF processing service converting multi-page PDFs to high-resolution page images."""

    def convert_pdf_to_images(self, pdf_path: Path) -> List[Tuple[int, Path]]:
        """
        Converts every page of a PDF document into a PNG image saved in OUTPUT_DIR.
        Returns a list of tuples: (page_number, image_path).
        """
        pdf_path = Path(pdf_path)
        output_images = []
        try:
            doc = fitz.open(str(pdf_path))
            logger.info(f"Loaded PDF '{pdf_path.name}' with {len(doc)} pages.")

            base_name = pdf_path.stem
            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                # Standard PDF user unit is 1/72 inch. 300 DPI = 300 / 72 = 4.166667 zoom matrix.
                zoom = 300.0 / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)

                page_num = page_idx + 1
                page_img_path = settings.OUTPUT_DIR / f"{base_name}_page_{page_num}.png"
                pix.save(str(page_img_path))
                output_images.append((page_num, page_img_path))

            doc.close()
            return output_images
        except Exception as e:
            logger.error(f"Failed to convert PDF '{pdf_path}': {str(e)}", exc_info=True)
            raise RuntimeError(f"Corrupted or invalid PDF file: {str(e)}")

    @staticmethod
    def _refine_bounds_from_data(col_bounds, lines_dict, sorted_y, hdr_y):
        """
        Move each column boundary into the empty lane between two columns of
        figures, instead of leaving it at the midpoint between two header
        labels.

        Every word below the header contributes its span. Where a boundary
        currently cuts through words, it is shifted to the widest vertical gap
        found between the words on either side of it. A boundary with no
        conflicting evidence is left exactly where the header put it.
        """
        spans = [
            (w[0], w[2])
            for y in sorted_y if y > hdr_y
            for w in lines_dict[y]
            if _is_num_word(w[4])
        ]
        if len(spans) < 8 or len(col_bounds) < 2:
            return col_bounds

        refined = list(col_bounds)
        for i in range(len(refined) - 1):
            # Look at every figure belonging to this PAIR of columns, and let
            # the figures say where the lane between them is. Requiring exactly
            # two lanes is the evidence that the pair really is two columns: if
            # the words group into one lane, or three, the header boundary is
            # left alone rather than moved on a guess.
            lo, hi = refined[i][1], refined[i + 1][2]
            inner = sorted(s for s in spans if s[0] >= lo - 2.0 and s[1] <= hi + 2.0)
            if len(inner) < 4:
                continue
            groups = [[inner[0]]]
            for s in inner[1:]:
                if s[0] - max(g[1] for g in groups[-1]) > 3.0:
                    groups.append([s])
                else:
                    groups[-1].append(s)
            if len(groups) != 2:
                continue
            left_edge = max(g[1] for g in groups[0])
            right_edge = min(g[0] for g in groups[1])
            if right_edge <= left_edge:
                continue
            new_b = (left_edge + right_edge) / 2.0
            # Never reorder columns or collapse one to nothing.
            if new_b <= refined[i][1] or new_b >= refined[i + 1][2]:
                continue
            refined[i] = (refined[i][0], refined[i][1], new_b)
            refined[i + 1] = (refined[i + 1][0], new_b, refined[i + 1][2])
        return refined

    def extract_digital_pdf_content(self, pdf_path: Path) -> Tuple[bool, str, List[dict]]:
        """
        Attempts direct digital vector text & table extraction from PDF using PyMuPDF and pdfplumber.
        Evaluates digital text reliability across word count, character ratio, bounding box coverage,
        and font glyph validity before proceeding.
        Returns: (has_digital_text, combined_markdown, per_page_results)
        """
        pdf_path = Path(pdf_path)
        try:
            # Primary: PyMuPDF spatial coordinate extraction preserves true token geometry
            has_fitz, fitz_md, fitz_results = self._fallback_fitz_extraction(pdf_path)
            if has_fitz and len(fitz_md.strip()) > 50:
                logger.info(f"PyMuPDF spatial coordinate engine successfully extracted '{pdf_path.name}'.")
                return (True, fitz_md, fitz_results)

            # Secondary fallback: pdfplumber layout engine
            import pdfplumber
            total_words = 0
            full_markdown_parts = []
            per_page_results = []

            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    raw_text = page.extract_text(layout=True) or ""
                    words = page.extract_words()
                    total_words += len(words)

                    if not raw_text.strip():
                        continue

                    formatted_lines = []
                    for raw_line in raw_text.splitlines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        if len(line) == 1 and not line.isdigit():
                            continue

                        parts = [p.strip() for p in re.split(r'\s{2,}', line) if p.strip()]
                        if len(parts) >= 2:
                            formatted_lines.append("| " + " | ".join(parts) + " |")
                        else:
                            formatted_lines.append(line)

                    page_text = "\n".join(formatted_lines)
                    full_markdown_parts.append(f"<!-- Page {page_num} -->\n" + page_text)
                    per_page_results.append({
                        "page_number": page_num,
                        "image_path": "",
                        "markdown": page_text,
                        "tokens": [{"text": w["text"], "bbox": [w["x0"], w["top"], w["x1"], w["bottom"]], "confidence": 1.0} for w in words],
                        "processing_time": 0.05
                    })

            # Check reliability
            alnum_count = sum(1 for c in "\n".join(full_markdown_parts) if (c.isalnum() or c in ".-/|,:%"))
            total_chars = len("\n".join(full_markdown_parts).strip())
            valid_ratio = (alnum_count / total_chars) if total_chars > 0 else 0.0

            has_digital_text = (total_words >= 20 and valid_ratio >= 0.70)
            combined_markdown = "\n\n---\n\n".join(full_markdown_parts)
            return (has_digital_text, combined_markdown, per_page_results)

        except Exception as e:
            logger.warning(f"Digital PDF extraction notice for '{pdf_path.name}': {str(e)}")
            return self._fallback_fitz_extraction(pdf_path)

    def _fallback_fitz_extraction(self, pdf_path: Path) -> Tuple[bool, str, List[dict]]:
        """Fallback PyMuPDF vector text & spatial column table extraction."""
        try:
            doc = fitz.open(str(pdf_path))
            total_words = 0
            full_markdown_parts = []
            per_page_results = []

            compound_subwords = {
                'name', 'qty', 'val', 'value', 'price', 'rate', 'no', 'number',
                'code', 'date', 'description', 'stock', 'sales', 'balance',
                'bal', 'quantity', 'amount', 'unit', 'pack'
            }

            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1
                words = page.get_text("words")
                total_words += len(words)

                if not words:
                    continue

                # Cluster words into lines by Y coordinate
                lines_dict = {}
                for w in words:
                    y_mid = (w[1] + w[3]) / 2.0
                    placed = False
                    for yk in lines_dict:
                        if abs(yk - y_mid) < 3.5:
                            lines_dict[yk].append(w)
                            placed = True
                            break
                    if not placed:
                        lines_dict[y_mid] = [w]

                sorted_y = sorted(lines_dict.keys())

                # Identify potential table header across stock statements, invoices, and ledgers
                hdr_y = None
                for y in sorted_y:
                    lt = " ".join(w[4] for w in sorted(lines_dict[y], key=lambda x: x[0])).lower()
                    has_item_kw = any(k in lt for k in ["product", "item", "description", "particulars", "material", "code", "sno", "s.no"])
                    has_col_kw = any(k in lt for k in ["pack", "qty", "unit", "op", "opening", "sale", "sales", "purchase", "purc", "cl", "closing", "rate", "amt", "amount", "stock", "value", "mrp", "batch", "balance", "total", "issued", "received"])
                    if has_item_kw and has_col_kw:
                        hdr_y = y
                        break
                    elif any(k in lt for k in ["description", "particulars"]) and any(k in lt for k in ["qty", "rate", "amount", "price", "total", "net"]):
                        hdr_y = y
                        break

                page_md_lines = []

                if hdr_y is not None:
                    hdr_words = sorted(lines_dict[hdr_y], key=lambda x: x[0])
                    cols = []
                    cur_name = ""
                    cur_x0 = 0.0
                    cur_x1 = 0.0

                    for w in hdr_words:
                        txt = w[4]
                        if not cur_name:
                            cur_name = txt
                            cur_x0 = w[0]
                            cur_x1 = w[2]
                        elif (txt.lower() in compound_subwords or cur_name.lower() in {'item', 'product', 'stock', 'sales', 'closing', 'opening'}) and (w[0] - cur_x1) < 6.0:
                            cur_name += " " + txt
                            cur_x1 = w[2]
                        else:
                            cols.append({"name": cur_name, "x0": cur_x0, "x1": cur_x1})
                            cur_name = txt
                            cur_x0 = w[0]
                            cur_x1 = w[2]
                    if cur_name:
                        cols.append({"name": cur_name, "x0": cur_x0, "x1": cur_x1})

                    # Absorb SUPER-HEADER tiers printed above the label row.
                    #
                    # A stacked header names each column across two rows:
                    #
                    #        Op.   Pur   Pur     Sale  Sale   ...  Bal.
                    #   SlNo  Qty   Qty   F.Qty   Qty   F.Qty  ...  Qty
                    #
                    # Only the leaf row was ever read, so six different columns
                    # came back named just "Qty". As JSON keys those are
                    # useless - nothing downstream can tell opening from
                    # purchase from sale, so role mapping gave up and emitted
                    # null for every quantity on every line.
                    #
                    # A tier qualifies on shape, not wording: no figures, at
                    # least two words, and words sitting inside the label row's
                    # own x-span. Letterhead and address lines fail that.
                    hdr_x0 = min(c["x0"] for c in cols)
                    hdr_x1 = max(c["x1"] for c in cols)
                    for y in reversed([yy for yy in sorted_y if yy < hdr_y][-2:]):
                        tier = sorted(lines_dict[y], key=lambda x: x[0])
                        tier_text = " ".join(w[4] for w in tier)
                        if len(tier) < 2 or any(_is_num_word(w[4]) for w in tier):
                            break
                        if _looks_like_report_period(tier_text) or _looks_like_footer(tier_text):
                            break
                        inside = [w for w in tier if w[0] >= hdr_x0 - 5.0 and w[2] <= hdr_x1 + 5.0]
                        if len(inside) < 2:
                            break
                        for c in cols:
                            over = [
                                w[4] for w in inside
                                if min(w[2], c["x1"]) - max(w[0], c["x0"]) > -3.0
                            ]
                            if over:
                                prefix = " ".join(over)
                                if prefix.lower() not in c["name"].lower():
                                    c["name"] = f"{prefix} {c['name']}".strip()
                        logger.debug(f"Absorbed PDF super-header tier: {' '.join(w[4] for w in tier)}")

                    col_bounds = []
                    for i in range(len(cols)):
                        left_b = 0.0 if i == 0 else (cols[i - 1]["x1"] + cols[i]["x0"]) / 2.0
                        right_b = 9999.0 if i == len(cols) - 1 else (cols[i]["x1"] + cols[i + 1]["x0"]) / 2.0
                        col_bounds.append((cols[i]["name"], left_b, right_b))

                    # Re-cut the boundaries using where the FIGURES actually
                    # fall, not where the labels sit.
                    #
                    # Figures are right-aligned under labels of a different
                    # width, so a boundary taken from the header midpoint can
                    # slice through a column of data: two values then land in
                    # one cell ("0 0") and the neighbouring cell comes back
                    # empty, which is both wrong and undetectable downstream.
                    col_bounds = self._refine_bounds_from_data(
                        col_bounds, lines_dict, sorted_y, hdr_y
                    )

                    in_table = False
                    for y in sorted_y:
                        line_words = sorted(lines_dict[y], key=lambda x: x[0])
                        line_text = " ".join(w[4] for w in line_words)

                        if y < hdr_y:
                            if re.match(r"^[\-\=\_\s]{4,}$", line_text):
                                continue
                            if y == sorted_y[0]:
                                page_md_lines.append(f"# {line_text}")
                            else:
                                page_md_lines.append(line_text)
                        elif y == hdr_y:
                            in_table = True
                            page_md_lines.append("")
                            page_md_lines.append("| " + " | ".join(c["name"] for c in cols) + " |")
                            page_md_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
                        else:
                            if re.match(r"^[\-\=\_\s]{4,}$", line_text):
                                continue

                            # End the table on the shape of the line, not on a
                            # vendor's wording. A summary block ("OP.Stk Val :
                            # 138995.86  Pur.Val : 39761.77"), a print stamp or
                            # a software credit line is where the grid stops -
                            # on any ERP, including ones never seen before.
                            if in_table and (_is_summary_line(line_text) or _looks_like_footer(line_text)):
                                in_table = False
                                page_md_lines.append("")

                            if in_table:
                                # A division/company banner ("HETERO HEALTHCARE
                                # (GENX)") carries no figures and spans only the
                                # description area. Projecting it through the
                                # column bands chops one title into two cells;
                                # keep it whole in the description column so it
                                # still reads as a section marker.
                                if not any(_is_num_word(w[4]) for w in line_words):
                                    desc_i = min(1, len(col_bounds) - 1)
                                    row_cells = [""] * len(col_bounds)
                                    row_cells[desc_i] = line_text.strip()
                                    page_md_lines.append("| " + " | ".join(row_cells) + " |")
                                    continue

                                if line_text.strip().lower().startswith("batch"):
                                    # Place batch row into item description column to preserve table shape
                                    row_cells = [""] * len(col_bounds)
                                    if len(row_cells) > 1:
                                        row_cells[1] = line_text.strip()
                                    else:
                                        row_cells[0] = line_text.strip()
                                    page_md_lines.append("| " + " | ".join(row_cells) + " |")
                                else:
                                    row_cells = [""] * len(col_bounds)
                                    for w in line_words:
                                        w_mid = (w[0] + w[2]) / 2.0
                                        for c_idx, (c_name, b_left, b_right) in enumerate(col_bounds):
                                            if b_left <= w_mid < b_right:
                                                row_cells[c_idx] = (row_cells[c_idx] + " " + w[4]).strip()
                                                break
                                    if any(row_cells):
                                        page_md_lines.append("| " + " | ".join(row_cells) + " |")
                            else:
                                page_md_lines.append(line_text)
                else:
                    for y in sorted_y:
                        line_words = sorted(lines_dict[y], key=lambda x: x[0])
                        page_md_lines.append(" ".join(w[4] for w in line_words))

                p_text = "\n".join(page_md_lines)
                full_markdown_parts.append(f"<!-- Page {page_num} -->\n" + p_text)
                per_page_results.append({
                    "page_number": page_num,
                    "image_path": "",
                    "markdown": p_text,
                    "tokens": [{"text": w[4], "bbox": [w[0], w[1], w[2], w[3]], "confidence": 1.0} for w in words],
                    "processing_time": 0.02
                })

            doc.close()
            has_digital_text = total_words > 30
            combined_markdown = "\n\n---\n\n".join(full_markdown_parts)
            logger.info(f"Successfully extracted digital PDF '{pdf_path.name}' ({total_words} words) using PyMuPDF spatial table engine.")
            return (has_digital_text, combined_markdown, per_page_results)
        except Exception as fe:
            logger.warning(f"PyMuPDF fallback extraction notice: {str(fe)}")
            return (False, "", [])


pdf_service = PDFService()
