import re
import json
import html
from pathlib import Path
from typing import Dict, Any, List, Tuple


def html_table_to_markdown(text: str) -> str:
    """
    Converts any HTML <table>...</table> blocks in text to standard Markdown pipe tables.
    Preserves all <td></td> empty cells, handles unclosed tables, strips inner tags, and normalizes column headers.
    """
    if not text or "<table" not in text.lower():
        return text

    normalized_text = text
    # Strip markdown code fences wrapping table blocks
    normalized_text = re.sub(r'```(?:html|xml)?\s*(<table.*?>.*?</table>)\s*```', r'\1', normalized_text, flags=re.DOTALL | re.I)
    
    # Auto-close unclosed <table> if truncated
    if "<table" in normalized_text.lower() and "</table" not in normalized_text.lower():
        if "<tr" in normalized_text.lower() and "</tr" not in normalized_text.lower()[-50:]:
            normalized_text = normalized_text + "\n</tr>"
        normalized_text = normalized_text + "\n</table>"

    def replacer(match):
        table_html = match.group(0)
        tr_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.I)
        rows = []
        for tr in tr_matches:
            td_matches = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', tr, re.DOTALL | re.I)
            if td_matches:
                rows.append([html.unescape(re.sub(r'<[^>]+>', '', td).strip()) for td in td_matches])
        if not rows:
            return table_html

        col_count = max(len(r) for r in rows)
        if col_count == 0:
            return table_html

        header = rows[0]
        while len(header) < col_count:
            header.append(f"Col {len(header) + 1}")

        md_lines = [
            "| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|"
        ]
        for r in rows[1:]:
            while len(r) < len(header):
                r.append("")
            md_lines.append("| " + " | ".join(r[:len(header)]) + " |")

        return "\n\n" + "\n".join(md_lines) + "\n\n"

def clean_markdown_fence(text: str) -> str:
    """Strips outermost markdown code fence ticks (```markdown ... ```) if LLM wrapped its entire output in one."""
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\r?\n?", "", cleaned)
        cleaned = re.sub(r"\r?\n?```$", "", cleaned)
    return cleaned.strip()


def reconcile_stock_ledger_row(cells: List[str], hdr_cells: List[str]) -> List[str]:
    """
    Applies fundamental accounting ledger reconciliation (Closing = Opening + Receipt - Issue)
    for stock statements when the far-right closing value is empty or was dropped by LLM column truncation.
    """
    hdr_upper = [h.upper() for h in hdr_cells]
    op_idx, rec_idx, iss_idx, cls_idx = -1, -1, -1, -1
    for i, h in enumerate(hdr_upper):
        if "OPEN" in h:
            op_idx = i
        elif any(k in h for k in ["RECEIPT", "PUR", "INQTY", "RCPT", "RECEV"]):
            rec_idx = i
        elif any(k in h for k in ["ISSUE", "SALE", "OUTQTY", "S A L E"]):
            iss_idx = i
        elif any(k in h for k in ["CLOSING QTY", "CL.STK", "BAL", "C.STK", "CLOSING"]):
            if "AMOUNT" not in h and "VAL" not in h and cls_idx == -1:
                cls_idx = i

    if op_idx != -1 and rec_idx != -1 and iss_idx != -1 and cls_idx != -1:
        val_cls = cells[cls_idx].strip() if len(cells) > cls_idx else ""
        if val_cls in ("", "-", "--", "None", "null"):
            try:
                raw_op = re.sub(r"[^\d.-]", "", cells[op_idx]) if len(cells) > op_idx and cells[op_idx].strip() else ""
                raw_rec = re.sub(r"[^\d.-]", "", cells[rec_idx]) if len(cells) > rec_idx and cells[rec_idx].strip() else ""
                raw_iss = re.sub(r"[^\d.-]", "", cells[iss_idx]) if len(cells) > iss_idx and cells[iss_idx].strip() else ""

                if raw_op or raw_rec or raw_iss:
                    val_op = float(raw_op) if raw_op else 0.0
                    val_rec = float(raw_rec) if raw_rec else 0.0
                    val_iss = float(raw_iss) if raw_iss else 0.0
                    calc_cls = val_op + val_rec - val_iss
                    if calc_cls >= 0:
                        formatted_cls = str(int(calc_cls)) if calc_cls.is_integer() else f"{calc_cls:.2f}"
                        while len(cells) <= cls_idx:
                            cells.append("")
                        cells[cls_idx] = formatted_cls
            except Exception:
                pass
    return cells


def reconcile_table_ledger_rows(rows: List[List[str]], hdr_cells: List[str]) -> List[List[str]]:
    """
    Multi-row accounting reconciliation across the entire table:
    1. Heals rows where Opening == Closing but VLM hallucinated non-zero issue/sale when Amount is 0.00.
    2. Detects and cancels adjacent-row vertical value bleeding (when a blank row duplicates the next row's sales/opening and violates Closing = Opening + Receipt - Issue).
    """
    hdr_upper = [h.upper() for h in hdr_cells]
    op_idx, rec_idx, iss_idx, amt_idx, cls_idx = -1, -1, -1, -1, -1
    for i, h in enumerate(hdr_upper):
        if "OPEN" in h:
            op_idx = i
        elif any(k in h for k in ["RECEIPT", "PUR", "INQTY", "RCPT", "RECEV"]):
            rec_idx = i
        elif any(k in h for k in ["ISSUE", "SALE", "OUTQTY", "S A L E"]):
            iss_idx = i
        elif "AMOUNT" in h and cls_idx == -1 and i > iss_idx:
            amt_idx = i
        elif any(k in h for k in ["CLOSING QTY", "CL.STK", "C.STK", "CLOSING"]):
            if "AMOUNT" not in h and "VAL" not in h and cls_idx == -1:
                cls_idx = i

    if op_idx == -1 or iss_idx == -1 or cls_idx == -1:
        return rows

    for idx, r in enumerate(rows):
        if len(r) <= max(op_idx, rec_idx, iss_idx, cls_idx):
            continue

        try:
            raw_op = re.sub(r"[^\d.-]", "", r[op_idx]) if r[op_idx].strip() else ""
            raw_rec = re.sub(r"[^\d.-]", "", r[rec_idx]) if rec_idx != -1 and r[rec_idx].strip() else ""
            raw_iss = re.sub(r"[^\d.-]", "", r[iss_idx]) if r[iss_idx].strip() else ""
            raw_cls = re.sub(r"[^\d.-]", "", r[cls_idx]) if r[cls_idx].strip() else ""

            v_op = float(raw_op) if raw_op else None
            v_rec = float(raw_rec) if raw_rec else 0.0
            v_iss = float(raw_iss) if raw_iss else 0.0
            v_cls = float(raw_cls) if raw_cls else None

            # Check if Opening == Closing and Sale Amount is 0:
            if v_op is not None and v_cls is not None and v_op == v_cls:
                if amt_idx != -1 and len(r) > amt_idx and r[amt_idx] in ("0", "0.00", "", "-"):
                    if v_iss != 0:
                        r[iss_idx] = "0"
                        if rec_idx != -1:
                            r[rec_idx] = "0"

            # Check for adjacent row bleeding (e.g. current row copied values from next row):
            if idx + 1 < len(rows):
                next_r = rows[idx + 1]
                if len(next_r) > max(op_idx, iss_idx):
                    same_op = (r[op_idx] == next_r[op_idx] and r[op_idx] not in ("", "0"))
                    same_iss = (r[iss_idx] == next_r[iss_idx] and r[iss_idx] not in ("", "0"))
                    if same_op and same_iss and v_op is not None and v_cls is not None:
                        calc = v_op + v_rec - v_iss
                        if abs(calc - v_cls) > 0.01:
                            # It is a bleed! Reset to zero sales and Opening = Closing
                            r[op_idx] = str(int(v_cls)) if v_cls.is_integer() else f"{v_cls:.2f}"
                            r[iss_idx] = "0"
                            if rec_idx != -1:
                                r[rec_idx] = "0"
                            if amt_idx != -1 and len(r) > amt_idx:
                                r[amt_idx] = "0.00"

        except Exception:
            pass

    return rows


def sanitize_extracted_markdown(md_text: str) -> str:
    """
    Sanitizes LLM markdown output:
    1. Removes dummy | Col 1 | Col 2 | ... | headers when row 1 is the real printed header.
    2. Strips accidental prepended category banner columns (e.g. | HETERO DERMA GLOW | ADABOR CREAM | 15GM | -> | ADABOR CREAM | 15GM |)
       and converts them into proper section headings (### HETERO DERMA GLOW).
    3. Truncates repetitive hallucinated rows at table ends.
    """
    if not md_text:
        return ""

    lines = md_text.splitlines()
    output_lines = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Check if line is a dummy table header like | Col 1 | Col 2 | ... |
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            dummy_patterns = [r"^col(umn)?\s*\d+$", r"^c\d+$"]
            dummy_count = sum(1 for c in cells if any(re.match(p, c, re.IGNORECASE) for p in dummy_patterns))

            # If this is dummy | Col 1 | Col 2 | ... |
            if dummy_count >= max(1, int(len(cells) * 0.6)) and i + 2 < n:
                sep_line = lines[i + 1].strip()
                next_line = lines[i + 2].strip()
                if sep_line.startswith("|") and re.match(r"^\|[\s\-:|]+\|$", sep_line) and next_line.startswith("|"):
                    next_cells = [c.strip() for c in next_line.split("|")[1:-1]]
                    text_cells = sum(1 for c in next_cells if re.search(r"[a-zA-Z]{3,}", c))
                    if text_cells >= 2:
                        # Skip dummy header and separator, promote real header followed by separator!
                        output_lines.append(lines[i + 2])
                        output_lines.append(lines[i + 1])
                        i += 3
                        continue

        output_lines.append(line)
        i += 1

    # Check for prepended category banner column across table rows
    table_blocks = []
    current_block = []
    for line in output_lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current_block.append(line)
        else:
            if current_block:
                table_blocks.append(current_block)
                current_block = []
            table_blocks.append([line])
    if current_block:
        table_blocks.append(current_block)

    final_lines = []
    for block in table_blocks:
        if len(block) >= 3 and block[0].strip().startswith("|") and not re.match(r"^\|[\s\-:|]+\|$", block[0].strip()):
            hdr_line = block[0]
            sep_line = block[1] if len(block) > 1 and re.match(r"^\|[\s\-:|]+\|$", block[1].strip()) else ""
            data_start = 2 if sep_line else 1
            data_rows = block[data_start:]
            banner_prefix = ""

            # Case A: Check if row 0 of data is a banner row (e.g. HETERO DERMA GLOW with blank or duplicate row 1 numbers)
            if len(data_rows) >= 2:
                r0_cells = [c.strip() for c in data_rows[0].strip().split("|")[1:-1]]
                r1_cells = [c.strip() for c in data_rows[1].strip().split("|")[1:-1]]
                if len(r0_cells) > 1 and len(r1_cells) > 1:
                    rest_blank = all(c == "" or c == "-" for c in r0_cells[1:])
                    rest_identical = (r0_cells[1:] == r1_cells[1:])
                    if rest_blank or rest_identical:
                        banner = r0_cells[0]
                        banner_prefix = f"\n### {banner}\n"
                        # Skip row 0 from table rows
                        data_rows = data_rows[1:]

            # Case B: Check if every row has a prepended banner in col 0
            if not banner_prefix:
                col0_vals = []
                for r in data_rows:
                    cells = [c.strip() for c in r.strip().split("|")[1:-1]]
                    if cells:
                        col0_vals.append(cells[0])

                if col0_vals and len(col0_vals) >= 3:
                    first_val = col0_vals[0]
                    same_count = sum(1 for v in col0_vals if v == first_val)
                    # If at least 70% of rows have the exact same non-numeric string in col 0
                    if same_count >= len(col0_vals) * 0.70 and len(first_val) > 3 and not re.match(r"^\d+$", first_val):
                        banner = first_val
                        banner_prefix = f"\n### {banner}\n"
                        cleaned_rows = []
                        for r in data_rows:
                            cells = [c.strip() for c in r.strip().split("|")[1:-1]]
                            if len(cells) > 1 and cells[0] == banner:
                                cleaned_rows.append("| " + " | ".join(cells[1:]) + " |")
                            else:
                                cleaned_rows.append(r)
                        data_rows = cleaned_rows

            # Case C: Check if row 0 has a company/division banner attached to its item name e.g. "HETERO HEALTH CARE LTD BORIT SB 130MG-10's"
            if not banner_prefix and len(data_rows) >= 2:
                r0_cells = [c.strip() for c in data_rows[0].strip().split("|")[1:-1]]
                r1_cells = [c.strip() for c in data_rows[1].strip().split("|")[1:-1]]
                prod_col = 1 if len(r0_cells) > 1 and re.match(r"^\d+$", r0_cells[0]) else 0
                if len(r0_cells) > prod_col and len(r1_cells) > prod_col:
                    p0 = r0_cells[prod_col]
                    p1 = r1_cells[prod_col]
                    banner_m = re.match(r"^([A-Z\s]{4,}\s+(?:LTD|PVT\.?\s*LTD|LIMITED|PHARMA|HEALTHCARE|HEALTH\s+CARE\s+LTD|DIVISION|ENTERPRISES))\s+(.+)$", p0, re.IGNORECASE)
                    if banner_m and not p1.upper().startswith(banner_m.group(1).upper()):
                        banner = banner_m.group(1).strip()
                        clean_item = banner_m.group(2).strip()
                        banner_prefix = f"\n### {banner}\n"
                        r0_cells[prod_col] = clean_item
                        data_rows[0] = "| " + " | ".join(r0_cells) + " |"

            # Case D: Check if row 0 has a company/division banner inserted as its own cell e.g. | 1 | HETERO HEALTH CARE LTD | BORIT SB 130MG-10's | ...
            if not banner_prefix and len(data_rows) >= 2:
                r0_cells = [c.strip() for c in data_rows[0].strip().split("|")[1:-1]]
                r1_cells = [c.strip() for c in data_rows[1].strip().split("|")[1:-1]]
                if len(r0_cells) >= 3 and len(r1_cells) >= 3:
                    cand_banner = r0_cells[1]
                    banner_full_m = re.match(r"^([A-Z\s]{4,}\s+(?:LTD|PVT\.?\s*LTD|LIMITED|PHARMA|HEALTHCARE|HEALTH\s+CARE\s+LTD|DIVISION|ENTERPRISES))$", cand_banner, re.IGNORECASE)
                    is_r0_c2_text = bool(re.search(r"[a-zA-Z]{3,}", r0_cells[2]))
                    is_r1_c2_num = bool(re.match(r"^[\d\s.,\-]*$", r1_cells[2]))
                    if banner_full_m and is_r0_c2_text and is_r1_c2_num:
                        banner = cand_banner.strip()
                        banner_prefix = f"\n### {banner}\n"
                        r0_cells = [r0_cells[0]] + r0_cells[2:]
                        data_rows[0] = "| " + " | ".join(r0_cells) + " |"

            hdr_cells = [c.strip() for c in hdr_line.split("|")[1:-1]]

            # Defensive Check: if OPENING column contains pack sizes (10'S, 60ML, 15GM) because ERP omitted 'PACKING'
            pack_unit_regex = re.compile(r"^\d+\s*('S|S|GM|GMS|ML|KG|CAP|TAB|NOS?|LTR)$", re.IGNORECASE)
            opening_col_idx = -1
            has_pack_hdr = any("PACK" in h.upper() for h in hdr_cells)
            for idx, h in enumerate(hdr_cells):
                if "OPENING" in h.upper():
                    opening_col_idx = idx
                    break

            if opening_col_idx != -1 and not has_pack_hdr and data_rows:
                pack_matches = 0
                sample_count = 0
                for r_str in data_rows[:10]:
                    if r_str.strip().startswith("|"):
                        c_list = [c.strip() for c in r_str.strip().split("|")[1:-1]]
                        if len(c_list) > opening_col_idx:
                            if pack_unit_regex.match(c_list[opening_col_idx]):
                                pack_matches += 1
                            sample_count += 1
                if sample_count > 0 and (pack_matches / sample_count) >= 0.4:
                    hdr_cells = hdr_cells[:opening_col_idx] + ["PACKING"] + hdr_cells[opening_col_idx:]

            # Defensive Check: if stock sheet has Bal. Qty / Closing Qty in header but is missing Bal.Val / Closing Val,
            # and rows end with quantity followed by currency value (e.g. 30 and 4819.80):
            if any("BAL" in h.upper() or "CLOSING" in h.upper() for h in hdr_cells) and "VAL" not in hdr_cells[-1].upper():
                has_shifted_val = False
                for r_str in data_rows[:10]:
                    if r_str.strip().startswith("|"):
                        c_list = [c.strip() for c in r_str.strip().split("|")[1:-1]]
                        non_empty = [c for c in c_list if c != ""]
                        if len(non_empty) >= 8 and re.match(r"^\d+\.\d{2}$", non_empty[-1]) and re.match(r"^\d+$", non_empty[-2]):
                            has_shifted_val = True
                            break
                if has_shifted_val:
                    hdr_cells.append("Bal.Val")
                    new_data_rows = []
                    for r_str in data_rows:
                        if r_str.strip().startswith("|") and re.search(r"\d+\.\d{2}", r_str):
                            c_list = [c.strip() for c in r_str.strip().split("|")[1:-1]]
                            non_empty = [c for c in c_list if c != ""]
                            if len(non_empty) >= 12 and re.match(r"^\d+\.\d{2}$", non_empty[-1]) and re.match(r"^\d+$", non_empty[-2]):
                                val_idx = -1
                                for idx, c in enumerate(non_empty):
                                    if re.match(r"^\d+\.\d{2}$", c):
                                        val_idx = idx
                                        break
                                if val_idx != -1 and val_idx >= 8 and len(non_empty) < len(hdr_cells):
                                    missing_count = len(hdr_cells) - len(non_empty)
                                    fixed_cells = non_empty[:val_idx] + ["0"] * missing_count + non_empty[val_idx:]
                                    new_data_rows.append("| " + " | ".join(fixed_cells) + " |")
                                    continue
                        new_data_rows.append(r_str)
                    data_rows = new_data_rows

            num_cols = len(hdr_cells)

            # Defensive Batch & Subline Filtering
            batch_regex = re.compile(r"(batch\s*(?:no\.?|number)|expiry\s*with\s*in|mfg\s*date|exp\s*date)", re.IGNORECASE)
            hdr_upper = [h.upper() for h in hdr_cells]
            has_sl_no = any("SL" in h or "S.NO" in h or "SR" in h for h in hdr_cells[:2])
            table_batch_notes = []
            prev_item = ""
            clean_data_rows = []

            for r_str in data_rows:
                stripped = r_str.strip()
                if not (stripped.startswith("|") and stripped.endswith("|")):
                    clean_data_rows.append(r_str)
                    continue
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if not cells:
                    continue

                has_batch_in_row = any(batch_regex.search(c) for c in cells)
                col0 = cells[0] if cells else ""
                col1 = cells[1] if len(cells) > 1 else ""

                is_subline = False
                if has_batch_in_row:
                    if has_sl_no and not re.match(r"^\d+$", col0):
                        is_subline = True
                    elif prev_item and (prev_item.upper() in col0.upper() or prev_item.upper() in col1.upper()):
                        if not (has_sl_no and re.match(r"^\d+$", col0)):
                            is_subline = True

                if is_subline:
                    for c in cells:
                        if batch_regex.search(c) or (len(c) > 10 and not re.match(r"^[\d\s.,\-]+$", c)):
                            table_batch_notes.append(c)
                    continue

                # Inline batch note cleaning
                new_cells = []
                for c in cells:
                    if batch_regex.search(c):
                        table_batch_notes.append(c)
                        m_qty = re.search(r"quantity:?\s*(\d+)", c, re.IGNORECASE)
                        if m_qty:
                            new_cells.append(m_qty.group(1))
                        elif "expiry" in c.lower() or "month" in c.lower():
                            new_cells.append("0")
                        else:
                            new_cells.append("")
                    else:
                        new_cells.append(c)
                cells = new_cells

                # Track item name
                if len(cells) > 1:
                    item_cand = cells[1] if has_sl_no and re.match(r"^\d+$", cells[0]) else cells[0]
                    if len(item_cand) > 2 and not re.match(r"^\d+$", item_cand):
                        prev_item = item_cand

                clean_data_rows.append("| " + " | ".join(cells) + " |")

            data_rows = clean_data_rows

            # Defensive Unmerging: split accidental composite quantity/date e.g. 12/6/27 -> 12, 6/27
            parsed_rows = []
            for r_str in data_rows:
                stripped = r_str.strip()
                if not (stripped.startswith("|") and stripped.endswith("|")):
                    parsed_rows.append(r_str)
                    continue
                cells = [c.strip() for c in stripped.split("|")[1:-1]]

                # If row has fewer cells than header or table contains quantity/expiry headers
                if len(cells) < num_cols or any(k in " ".join(hdr_cells).upper() for k in ["EXP", "DUMP"]):
                    new_cells = []
                    for c in cells:
                        m = re.match(r"^(\d+)/(\d{1,2}/\d{2,4})$", c)
                        if m and (len(new_cells) + len(cells) - 1 <= num_cols):
                            new_cells.extend([m.group(1), m.group(2)])
                        else:
                            new_cells.append(c)
                    cells = new_cells

                while len(cells) < num_cols:
                    cells.append("")
                cells = cells[:num_cols]
                cells = reconcile_stock_ledger_row(cells, hdr_cells)
                parsed_rows.append(cells)

            # Reconcile multi-row ledger math (drag-down / bleed healing)
            data_only_rows = [r for r in parsed_rows if isinstance(r, list)]
            data_only_rows = reconcile_table_ledger_rows(data_only_rows, hdr_cells)

            # Defensive Pruning: strip trailing columns that are 100% empty across all rows
            prune_count = 0
            if data_only_rows:
                for col_idx in range(num_cols - 1, -1, -1):
                    is_empty = all(
                        r[col_idx] in ("", "-", "--", "None", "null")
                        for r in data_only_rows
                    )
                    if is_empty:
                        prune_count += 1
                    else:
                        break

            if prune_count > 0:
                new_col_count = num_cols - prune_count
                hdr_cells = hdr_cells[:new_col_count]
                data_only_rows = [r[:new_col_count] for r in data_only_rows]
                parsed_rows = data_only_rows
            else:
                parsed_rows = data_only_rows

            # Reconstruct table block
            if banner_prefix:
                final_lines.append(banner_prefix.strip())
            final_lines.append("| " + " | ".join(hdr_cells) + " |")
            final_lines.append("| " + " | ".join(["---"] * len(hdr_cells)) + " |")
            for r in parsed_rows:
                if isinstance(r, list):
                    final_lines.append("| " + " | ".join(r) + " |")
                else:
                    final_lines.append(r)

            continue

        final_lines.extend(block)

    # Strip any hallucinated or generated ### Batch sections
    cleaned_blocks = []
    skip_batch_section = False
    for line in final_lines:
        stripped = line.strip()
        if re.match(r"^#{1,4}\s*batch", stripped, re.IGNORECASE):
            skip_batch_section = True
            continue
        if skip_batch_section:
            if stripped.startswith("-") or stripped.startswith("*") or not stripped:
                continue
            elif stripped.startswith("#") or stripped.startswith("|"):
                skip_batch_section = False
            else:
                continue
        cleaned_blocks.append(line)

    return "\n".join(cleaned_blocks)


def parse_json_from_llm_response(text: str) -> Dict[str, Any]:
    """
    Extracts and parses JSON object from LLM response text,
    stripping markdown block ticks ```json ... ``` and repairing truncated JSON arrays/objects.
    """
    if not text:
        return {}

    cleaned = text.strip()
    # Remove markdown codeblocks
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    # Direct parse attempt
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Regex find first '{' to last '}'
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # Auto-repair truncated or unclosed JSON
    start_idx = cleaned.find('{')
    if start_idx != -1:
        candidate = cleaned[start_idx:].strip()

        # Trim incomplete trailing fields/strings
        candidate = re.sub(r',?\s*"[^"]*$', '', candidate)
        candidate = re.sub(r',?\s*"[^"]*"\s*:\s*[^,}]*$', '', candidate)
        candidate = re.sub(r',\s*$', '', candidate)

        # Balance brackets and braces
        open_brackets = candidate.count('[') - candidate.count(']')
        open_braces = candidate.count('{') - candidate.count('}')
        if open_brackets > 0:
            candidate += ']' * open_brackets
        if open_braces > 0:
            candidate += '}' * open_braces

        try:
            return json.loads(candidate)
        except Exception:
            pass

        # Sub-string attempt up to last completed object
        last_obj = candidate.rfind('}')
        if last_obj != -1:
            sub = candidate[:last_obj + 1]
            b_diff = sub.count('[') - sub.count(']')
            c_diff = sub.count('{') - sub.count('}')
            if b_diff > 0:
                sub += ']' * b_diff
            if c_diff > 0:
                sub += '}' * c_diff
            try:
                return json.loads(sub)
            except Exception:
                pass

    return {"raw_response": text}


def extract_markdown_tables(markdown_text: str) -> List[List[List[str]]]:
    """
    Parses Markdown tables from extracted document markdown text into structured lists.
    Returns a list of tables, where each table is a list of rows (list of cell strings).
    Supports both standard Markdown pipe tables (| Col |), HTML tables, and space-aligned billing tables.
    """
    if not markdown_text:
        return []

    # First clean and sanitize markdown (fixing dummy headers & accidental prepended banner columns)
    markdown_text = sanitize_extracted_markdown(markdown_text)

    # Convert any HTML tables to standard Markdown pipe tables
    markdown_text = html_table_to_markdown(markdown_text)

    tables = []
    lines = markdown_text.splitlines()
    current_table = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            # Check if it's a separator line like |---|---|
            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                continue
            cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
            current_table.append(cells)
        else:
            if current_table:
                tables.append(current_table)
                current_table = []

    if current_table:
        tables.append(current_table)

    # Post-process parsed tables to promote real headers if row 0 was dummy Col 1, Col 2
    cleaned_tables = []
    for t in tables:
        if not t or len(t) < 2:
            cleaned_tables.append(t)
            continue
        hdr = t[0]
        r1 = t[1]
        dummy_patterns = [r"^col(umn)?\s*\d+$", r"^c\d+$"]
        dummy_count = sum(1 for h in hdr if any(re.match(p, h.strip(), re.IGNORECASE) for p in dummy_patterns))
        if dummy_count >= max(1, int(len(hdr) * 0.6)):
            text_cells = sum(1 for c in r1 if re.search(r"[a-zA-Z]{3,}", c))
            if text_cells >= 2:
                t = [r1] + t[2:]
        cleaned_tables.append(t)
    tables = cleaned_tables

    # Fallback: if no pipe tables found, check for space-aligned billing table lines
    if not tables:
        space_table = []
        in_table = False
        for line in lines:
            stripped = line.strip()
            # Check for header line
            if not in_table and ("Product Name" in stripped or "Product" in stripped) and ("Sales" in stripped or "InQty" in stripped or "Total" in stripped or "Cl.Value" in stripped):
                hdr_tokens = re.split(r'\s{2,}|\t', stripped)
                if len(hdr_tokens) < 4:
                    hdr_tokens = stripped.split()
                space_table.append(hdr_tokens)
                in_table = True
            elif in_table:
                if not stripped or stripped.startswith("Opening Value") or stripped.startswith("**") or stripped.startswith("---") or stripped.startswith("Total Value") or stripped.startswith("###"):
                    in_table = False
                    if len(space_table) > 1:
                        tables.append(space_table)
                    space_table = []
                else:
                    row_tokens = re.split(r'\s{2,}|\t', stripped)
                    if len(row_tokens) < 3:
                        row_tokens = stripped.split()
                    space_table.append(row_tokens)
        if len(space_table) > 1:
            tables.append(space_table)

    return tables


def convert_markdown_table_to_spreadsheet_json(markdown_text: str) -> Dict[str, Any]:
    """
    Converts parsed markdown tables into structured SPREADSHEET JSON.
    Captures primary table and all supplementary tables (e.g. Near Expiry, Batches).
    """
    tables = extract_markdown_tables(markdown_text)
    if not tables or not tables[0]:
        return {}

    all_tables = []
    primary_columns = []
    primary_rows = []

    for t_idx, table in enumerate(tables):
        if not table:
            continue
        header = list(table[0])
        num_cols = len(header)
        if num_cols == 0:
            continue

        raw_rows = [list(r) for r in table[1:]]

        # Check if first row is a duplicate category header with row 1's values
        if len(raw_rows) >= 2:
            r0 = raw_rows[0]
            r1 = raw_rows[1]
            if len(r0) > 1 and len(r1) > 1:
                rest_blank = all(c.strip() == "" or c.strip() == "-" for c in r0[1:])
                rest_identical = (r0[1:] == r1[1:])
                if rest_blank or rest_identical:
                    raw_rows = raw_rows[1:]

        columns = [h if h else f"Column {i+1}" for i, h in enumerate(header)]
        rows = []
        for raw_row in raw_rows:
            # Skip rows where all cells are empty
            if not any(c.strip() for c in raw_row):
                continue
            padded = list(raw_row)
            if len(padded) < num_cols:
                padded.extend([""] * (num_cols - len(padded)))
            elif len(padded) > num_cols:
                padded = padded[:num_cols]
            rows.append(padded)

        # Defensive Pruning: prune trailing columns that are 100% empty across all rows
        if rows and columns:
            prune_count = 0
            for col_idx in range(len(columns) - 1, -1, -1):
                is_empty = all(
                    r[col_idx].strip() in ("", "-", "--", "None", "null")
                    for r in rows
                )
                if is_empty:
                    prune_count += 1
                else:
                    break
            if prune_count > 0:
                new_col_count = len(columns) - prune_count
                columns = columns[:new_col_count]
                rows = [r[:new_col_count] for r in rows]

        all_tables.append({
            "table_index": t_idx + 1,
            "columns": columns,
            "rows": rows
        })

    # Select the dominant table with the most data cells (rows * columns) as primary
    if all_tables:
        dominant_table = max(all_tables, key=lambda t: len(t.get("columns", [])) * len(t.get("rows", [])))
        primary_columns = dominant_table.get("columns", [])
        primary_rows = dominant_table.get("rows", [])

    return {
        "sheet_title": "Sheet1",
        "columns": primary_columns,
        "rows": primary_rows,
        "all_tables": all_tables
    }


def auto_align_billing_table_columns(headers: List[str], rows: List[List[str]]) -> Tuple[List[List[str]], List[int]]:
    """
    Specifically for borderless billing invoices and stockist ledger tables whose header
    matches a known billing-column signature.

    NOTE on actual behavior: despite the name, this does NOT perform arithmetic-based
    column realignment (no stock-accounting equation is evaluated here). It only pads or
    truncates each row to the header's column count. If an upstream extraction stage
    already shifted a row by one column (e.g. a missing "Packing" cell), this function
    does not detect or correct that shift - it just forces the row to the right length,
    which can "lock in" a pre-existing misalignment rather than fixing it. Real
    arithmetic-based shift detection/correction is intentionally not implemented here;
    see OCR_ENGINEERING_CONTEXT_HANDOFF.md Phase 6 for why it's deferred (this function's
    call site runs for every native spreadsheet, not just scanned ones, so a change to
    the row *values* here needs dedicated shifted-column test fixtures first).

    Returns (aligned_rows, mismatch_row_indices) - the second list flags row indices
    (into `rows`) whose cell count did not already match the header count before
    padding/truncation, so callers can surface that as a review signal instead of it
    passing through silently.
    """
    if not headers or not rows:
        return rows, []

    header_str = " ".join([str(h) for h in headers]).upper()
    billing_signatures = [
        "OPENQ", "INQTY", "TOT.QTY", "OUTQTY", "CLS.QTY",
        "O.BAL", "RCPTS", "PUR.RET", "CL.VALUE"
    ]
    is_billing_table = any(sig in header_str for sig in billing_signatures)
    if not is_billing_table:
        return rows, []

    col_count = len(headers)
    aligned_rows = []
    mismatch_row_indices = []

    for row_idx, r in enumerate(rows):
        clean_row = [str(c).strip() if c is not None else "" for c in r]
        if len(clean_row) != col_count:
            mismatch_row_indices.append(row_idx)
        if len(clean_row) < col_count:
            clean_row.extend([""] * (col_count - len(clean_row)))
        elif len(clean_row) > col_count:
            clean_row = clean_row[:col_count]
        aligned_rows.append(clean_row)

    return aligned_rows, mismatch_row_indices


def map_invoice_column_roles(columns: List[str]) -> Dict[str, int]:
    """
    Maps invoice table columns to semantic roles (description/quantity/unit_price/amount)
    by header-keyword matching, the same pattern already used by
    validation_service._map_stock_columns for stock tables. Used to replace hardcoded
    positional indices (r[0]/r[1]/r[2]/r[-1]) when building invoice line_items - column
    order varies across real invoices (e.g. "Qty, Description, Rate, Disc%, Amount"), so
    a fixed position silently writes values under the wrong output key whenever order
    differs from the assumed description/quantity/unit_price/.../amount layout.
    """
    role_map: Dict[str, int] = {}
    for idx, col in enumerate(columns):
        c_upper = str(col).upper().strip()
        if "description" not in role_map and any(k in c_upper for k in ["DESCRIPTION", "PARTICULARS", "ITEM", "PRODUCT"]):
            role_map["description"] = idx
        elif "quantity" not in role_map and any(k in c_upper for k in ["QTY", "QUANTITY"]):
            role_map["quantity"] = idx
        elif "unit_price" not in role_map and any(k in c_upper for k in ["RATE", "PRICE", "UNIT PRICE", "MRP"]):
            role_map["unit_price"] = idx
        elif "amount" not in role_map and any(k in c_upper for k in ["AMOUNT", "TOTAL", "VALUE", "AMT"]):
            role_map["amount"] = idx
    return role_map


def sanitize_row_wise_table(table_rows: List[List[str]], target_col_count: int) -> List[List[str]]:
    """
    Enforces strict row-wise cell alignment.
    Guarantees every row in table_rows has exactly target_col_count cell elements,
    filling empty gaps with "" to prevent cell shifting.
    """
    sanitized = []
    for r in table_rows:
        row_cells = [c.strip() if c else "" for c in r]
        if len(row_cells) < target_col_count:
            row_cells.extend([""] * (target_col_count - len(row_cells)))
        elif len(row_cells) > target_col_count:
            row_cells = row_cells[:target_col_count]
        sanitized.append(row_cells)
    return sanitized


def strip_footer_metadata(text: str) -> str:
    """Strips UI footer noise, Excel status bar lines, LLM notes/remarks, and non-table metadata below the sheet data."""
    if not text:
        return ""

    footer_patterns = [
        r"^\s*Sheet\d+\s*$",
        r"^\s*Enter Accessibility:.*$",
        r"^\s*Accessibility:.*$",
        r"^\s*Good to go.*$",
        r"^\s*\d{1,2}[CF]\s*$",
        r"^\s*Ready\s*$",
        r"^\s*100%\s*$",
        r"^\s*Note\s*:.*$",
        r"^\s*Notes\s*:.*$",
        r"^\s*Translation note\s*:.*$",
        r"^\s*This table includes.*$",
        r"^\s*Here is the.*$"
    ]
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        is_footer = any(re.match(p, line.strip(), re.IGNORECASE) for p in footer_patterns)
        if not is_footer:
            cleaned.append(line)

    return "\n".join(cleaned).strip()


def strip_markdown(text: str) -> str:
    """Strips Markdown syntax, conversational LLM preamble lines, dividers, and bullet symbols to return clean plain text."""
    if not text:
        return ""

    # 1. Remove HTML comments like <!-- Page 1 -->
    cleaned = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # 2. Strip Excel UI footer noise
    cleaned = strip_footer_metadata(cleaned)

    # 3. Remove LLM conversational preamble lines & rule echoes
    cleaned = re.sub(r"Critical Orientation,?\s*Reversed Image.*?(?=1\.|Document Text|Extracted|\n\n\n)", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^\s*\d+\.\s+If the input image is upside down.*?\n+", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s+Extract ONLY visible information.*?\n+", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s+If a field is not visible.*?\n+", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s+If text is written in Hindi.*?\n+", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"^(Here'?s?\s+(is\s+)?(the\s+)?(transcription|translation|extracted|text).*?:?)\n+", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"^(Translation|Transcription|Extracted Text|Document Text):?\n+", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

    # 4. Remove horizontal dividers (---, ***, ___)
    cleaned = re.sub(r"^[-\*_]{3,}\s*$", "", cleaned, flags=re.MULTILINE)

    # 5. Remove headers (#, ##, ###)
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)

    # 6. Remove bold and italic formatting (**text**, *text*, __text__, _text_)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)

    # 7. Remove bullet prefixes (- , * , • )
    cleaned = re.sub(r"^\s*[\-\*•]\s+", "", cleaned, flags=re.MULTILINE)

    # 8. Remove codeblocks and inline code
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)

    # 9. Clean table pipes (| col1 | col2 | -> col1   col2)
    cleaned = re.sub(r"^\|[\s\-:|]+\|$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\|", " ", cleaned)

    # 10. Clean multiple blank lines
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)

    return cleaned.strip()


def normalize_no_text_response(text: str) -> str:
    """
    Checks if LLM response indicates no text or is describing an image without text.
    Returns clean 'No text has been found.' rather than long conversational explanations.
    """
    if not text or not text.strip():
        return "No text has been found."

    t_clean = text.strip()
    no_text_patterns = [
        r"^(the|this)\s+(image|photograph|picture|photo)\s+(shows|depicts|contains|is of|features|does not contain|appears to be|seems to be).*",
        r"^i (can|cannot|can't) (see|find|identify|detect|recognize) (any|readable|visible|meaningful|printed|written|detectable).*",
        r"^there is no (visible|readable|printed|detectable|written|clear) text.*",
        r"^no (visible\s+)?text (is|has been|was|could be) (found|detected|visible|present|identified).*",
        r"^i'm sorry, but i (can't|cannot) (assist|help|extract|translate).*",
        r"^this is a (blank|empty|landscape|scenery|nature|object|photo).*"
    ]
    for p in no_text_patterns:
        if re.match(p, t_clean, re.IGNORECASE):
            return "No text has been found."

    return text

