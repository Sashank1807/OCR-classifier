import io
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
from app.core.logger import logger


class ExcelService:
    """High-speed native Excel (.xlsx, .xls) and CSV parsing engine using pandas & openpyxl."""

    HEADER_KEYWORDS = {
        "description", "product", "item", "qty", "stock", "value", "quantity",
        "rate", "amount", "price", "opening", "closing", "issue", "receive",
        "receipt", "purchase", "sale", "expiry", "batch", "date", "code",
        "particulars", "total", "mrp", "unit", "pack", "discount", "disc", "net",
        "name", "month", "id", "slno", "sno", "cost"
    }

    def extract_excel_content(self, file_path: Path) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Parses all sheets in an Excel or CSV file directly into clean Markdown tables and plain text.
        Detects actual table header row even when pre-table company banners/metadata exist.
        Sanitizes embedded cell newlines (Alt+Enter) so markdown table rows never break.
        Returns: (combined_markdown, combined_plain_text, per_page_results)
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower().lstrip(".")
        full_markdown_parts = []
        full_raw_text_parts = []
        per_page_results = []

        try:
            if ext == "csv":
                df = self._parse_csv_robust(file_path)
                sheet_dict = {"Sheet1": df}
            else:
                sheet_dict = pd.read_excel(file_path, sheet_name=None, header=None, engine="openpyxl" if ext == "xlsx" else None)

            page_num = 1
            for sheet_name, df_raw in sheet_dict.items():
                if df_raw.empty:
                    continue

                # Detect the actual table header row and pre-table banner/metadata lines
                hdr_idx, banner_lines = self._detect_header_row_and_metadata(df_raw)

                # Clean and sanitize column header names (strip internal newlines like OPENING\nSTOCK)
                raw_headers = df_raw.iloc[hdr_idx].values
                headers = []
                for j, c in enumerate(raw_headers):
                    c_clean = re.sub(r'\s+', ' ', str(c)).replace('|', '/').strip() if pd.notna(c) and str(c).strip().lower() != "nan" else ""
                    headers.append(c_clean if c_clean else f"Column {j + 1}")

                num_cols = len(headers)
                header_row = "| " + " | ".join(headers) + " |"
                separator_row = "| " + " | ".join(["---"] * num_cols) + " |"

                # Extract and sanitize data rows (rows after hdr_idx)
                data_rows = []
                plain_rows = []
                for r_idx in range(hdr_idx + 1, len(df_raw)):
                    row_vals_raw = df_raw.iloc[r_idx].values
                    # Check if row is completely empty
                    has_content = any(pd.notna(v) and str(v).strip() and str(v).strip().lower() != "nan" for v in row_vals_raw)
                    if not has_content:
                        continue

                    # Sanitize cells: collapse all internal \r\n, \n, \r into space, escape pipes
                    row_cells = []
                    for v in row_vals_raw[:num_cols]:
                        if pd.isna(v) or str(v).strip().lower() == "nan":
                            row_cells.append("")
                        else:
                            clean_cell = re.sub(r'\s+', ' ', str(v)).replace('|', '/').strip()
                            row_cells.append(clean_cell)

                    while len(row_cells) < num_cols:
                        row_cells.append("")

                    data_rows.append("| " + " | ".join(row_cells) + " |")
                    plain_rows.append("\t".join(row_cells))

                # Build sheet markdown with banner lines before the table
                md_sections = [f"### Sheet: {sheet_name}"]
                if banner_lines:
                    for b_line in banner_lines:
                        md_sections.append(f"**{b_line}**")

                table_md = "\n".join([header_row, separator_row] + data_rows)
                md_sections.append(table_md)
                sheet_md = "\n\n".join(md_sections)

                # Plain text representation
                plain_header = "\n".join(banner_lines) + ("\n" if banner_lines else "")
                sheet_plain = f"--- Sheet: {sheet_name} ---\n{plain_header}\t".join(headers) + "\n" + "\n".join(plain_rows)

                full_markdown_parts.append(f"<!-- Page {page_num} -->\n" + sheet_md)
                full_raw_text_parts.append(sheet_plain)

                per_page_results.append({
                    "page_number": page_num,
                    "image_path": "",
                    "markdown": sheet_md,
                    "processing_time": 0.01
                })
                page_num += 1

            combined_markdown = "\n\n---\n\n".join(full_markdown_parts)
            combined_plain = "\n\n".join(full_raw_text_parts)
            logger.info(f"Successfully parsed Excel/CSV file '{file_path.name}' with {len(per_page_results)} sheet(s).")
            return combined_markdown, combined_plain, per_page_results

        except Exception as e:
            logger.error(f"Failed to parse Excel file '{file_path}': {str(e)}", exc_info=True)
            raise RuntimeError(f"Invalid or corrupted Excel file: {str(e)}")

    def _detect_header_row_and_metadata(self, df_raw: pd.DataFrame) -> Tuple[int, List[str]]:
        """
        Detects the real table header row in a spreadsheet that may contain pre-table banner/title rows.
        Returns: (header_row_idx, pre_table_banner_lines)
        """
        if len(df_raw) <= 1:
            return 0, []

        best_row_idx = 0
        best_score = -1.0

        for r_idx in range(min(15, len(df_raw))):
            row = df_raw.iloc[r_idx]
            non_empty_cells = [str(c).strip() for c in row if pd.notna(c) and str(c).strip() and str(c).strip().lower() != "nan"]
            count = len(non_empty_cells)
            if count == 0:
                continue

            kw_hits = sum(1 for c in non_empty_cells if any(kw in c.lower() for kw in self.HEADER_KEYWORDS))

            num_numeric = 0
            for cell_str in non_empty_cells:
                clean_num = cell_str.replace(",", "").replace("-", "").strip()
                if clean_num and re.match(r'^\d+(\.\d+)?$', clean_num):
                    num_numeric += 1

            # Score: reward non-empty column breadth and keyword hits, penalize purely numeric data rows
            score = (count * 1.5) + (kw_hits * 3.0) - (num_numeric * 2.0)

            if score > best_score and count >= 2:
                best_score = score
                best_row_idx = r_idx

        # Collect pre-table banner lines
        banner_lines = []
        for i in range(best_row_idx):
            items = [str(c).strip() for c in df_raw.iloc[i] if pd.notna(c) and str(c).strip() and str(c).strip().lower() != "nan"]
            if items:
                line_str = " ".join(items)
                # Clean up weird non-ascii encoding artefacts (like 'ÿÿÿ')
                line_str = re.sub(r'[\x80-\xff]+', '', line_str).strip()
                if line_str:
                    banner_lines.append(line_str)

        return best_row_idx, banner_lines

    def _parse_csv_robust(self, file_path: Path) -> pd.DataFrame:
        """
        Robustly parses CSV files that may have metadata/title lines before the actual data table.
        Also gracefully handles Excel workbooks (.xlsx or .xls) misnamed with a .csv extension.
        """
        # Strategy 0: Check if this file is actually a binary Excel file misnamed as CSV
        try:
            with open(file_path, "rb") as f:
                header_bytes = f.read(8)
            if header_bytes.startswith(b"PK\x03\x04"):
                logger.info(f"File '{file_path.name}' has .csv extension but contains PK zip header. Parsing as Excel .xlsx")
                return pd.read_excel(file_path, engine="openpyxl")
            elif header_bytes.startswith(b"\xd0\xcf\x11\xe0"):
                logger.info(f"File '{file_path.name}' has .csv extension but contains OLE header. Parsing as Excel .xls")
                return pd.read_excel(file_path, engine="xlrd")
        except Exception as ex_chk:
            logger.warning(f"Magic bytes check notice: {ex_chk}")

        # Strategy 1: Standard parse
        try:
            df = pd.read_csv(file_path)
            if len(df.columns) > 1:
                return df
        except Exception:
            pass

        # Strategy 2: Read raw lines and find the row with the most delimited fields (= actual header)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_lines = f.readlines()
        except Exception:
            with open(file_path, "r", encoding="latin-1") as f:
                raw_lines = f.readlines()

        # Detect delimiter (comma, tab, pipe, semicolon)
        for delim in [",", "\t", "|", ";"]:
            field_counts = [len(line.split(delim)) for line in raw_lines[:20]]
            max_fields = max(field_counts) if field_counts else 1
            if max_fields >= 3:
                # Find the first row that has max_fields (this is likely the header)
                skip_rows = 0
                for i, count in enumerate(field_counts):
                    if count == max_fields:
                        skip_rows = i
                        break
                try:
                    df = pd.read_csv(file_path, skiprows=skip_rows, sep=delim)
                    if len(df.columns) >= 2:
                        logger.info(f"CSV parsed with delimiter='{delim}', skipped {skip_rows} metadata rows.")
                        return df
                except Exception:
                    continue

        # Strategy 3: Last resort - force read with error tolerance
        df = pd.read_csv(file_path, sep=None, engine="python", on_bad_lines="skip")
        return df


excel_service = ExcelService()
