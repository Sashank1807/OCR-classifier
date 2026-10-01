from pathlib import Path
from typing import Dict, Any, List, Tuple
import docx
from app.core.logger import logger


class WordDocumentService:
    """High-speed native Microsoft Word (.docx) document extraction engine."""

    def extract_word_content(self, file_path: Path) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Parses all paragraphs, headings, bullet lists, and tables in a Word (.docx) file
        directly into clean Markdown and plain text.
        Returns: (combined_markdown, combined_plain_text, per_page_results)
        """
        try:
            doc = docx.Document(str(file_path))
            md_lines = []
            plain_lines = []

            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue

                style_name = paragraph.style.name.lower() if paragraph.style else ""
                if "heading 1" in style_name or "title" in style_name:
                    md_lines.append(f"# {text}")
                elif "heading 2" in style_name:
                    md_lines.append(f"## {text}")
                elif "heading 3" in style_name:
                    md_lines.append(f"### {text}")
                elif "list" in style_name or "bullet" in style_name:
                    md_lines.append(f"- {text}")
                else:
                    md_lines.append(text)

                plain_lines.append(text)

            # Process Word Tables
            for table in doc.tables:
                table_rows = []
                for row in table.rows:
                    cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                    table_rows.append(cells)

                if table_rows and len(table_rows) > 0:
                    header = table_rows[0]
                    num_cols = len(header)
                    hdr_row = "| " + " | ".join(header) + " |"
                    sep_row = "| " + " | ".join(["---"] * num_cols) + " |"
                    data_rows = ["| " + " | ".join(r) + " |" for r in table_rows[1:]]

                    table_md = "\n".join([hdr_row, sep_row] + data_rows)
                    md_lines.append("\n" + table_md + "\n")
                    plain_lines.append(table_md)

            combined_markdown = "<!-- Page 1 -->\n" + "\n\n".join(md_lines)
            combined_plain = "\n\n".join(plain_lines)

            per_page_results = [{
                "page_number": 1,
                "image_path": "",
                "markdown": "\n\n".join(md_lines),
                "processing_time": 0.01
            }]

            logger.info(f"Successfully extracted Word document '{file_path.name}' ({len(md_lines)} blocks).")
            return combined_markdown, combined_plain, per_page_results

        except Exception as e:
            logger.error(f"Failed to parse Word document '{file_path}': {str(e)}", exc_info=True)
            raise RuntimeError(f"Invalid or corrupted Word document: {str(e)}")


word_service = WordDocumentService()
