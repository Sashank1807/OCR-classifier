import json
import csv
import io
from pathlib import Path
from typing import Dict, Any, Tuple
from app.models.document import DocumentRecord, DocumentResult
from app.utils.file_utils import strip_markdown
from app.core.logger import logger


class ExportService:
    """Service generating downloadable export content (JSON, TXT, CSV) without markdown clutter."""

    def generate_export(self, record: DocumentRecord, result: DocumentResult, export_format: str) -> Tuple[str, str, str]:
        """
        Generates exported text content, content_type, and output filename.
        Ensures flow: First Structured JSON Data, At Last Clean Plain Text.
        Returns: (content_string, media_type, download_filename)
        """
        fmt = export_format.lower().strip()
        stem = Path(record.original_filename).stem
        clean_plain_text = strip_markdown(result.raw_text or "")

        if fmt == "json":
            # Markdown plus structured JSON, matching the API contract. This
            # format previously carried plain_text and no markdown, which
            # made it the odd one out: the same content as markdown with the
            # formatting stripped. The txt/csv formats below still render
            # plain text, because there it IS the requested output.
            export_dict = {
                "filename": record.original_filename,
                "document_type": record.document_type,
                "project_name": record.project_name or record.department,
                "structured_data": result.structured_json,
                "markdown": result.markdown_text or ""
            }
            content = json.dumps(export_dict, indent=2, ensure_ascii=False)
            return content, "application/json", f"{stem}_ocr.json"

        elif fmt in ["erp", "erp_data", "business"]:
            # The business projection, byte-identical to what the API returns
            # for response_format=erp - so a mapping built against a
            # downloaded sample holds when the integration goes live.
            from app.services.erp_payload import erp_from_record
            content = json.dumps(erp_from_record(record, result), indent=2, ensure_ascii=False)
            return content, "application/json", f"{stem}_erp.json"

        elif fmt in ["txt", "text", "md", "markdown"]:
            struct_str = json.dumps(result.structured_json, indent=2, ensure_ascii=False) if result.structured_json else "{}"
            
            content = f"==================================================\n"
            content += f"STRUCTURED JSON DATA\n"
            content += f"==================================================\n"
            content += f"{struct_str}\n\n"
            content += f"==================================================\n"
            content += f"PLAIN TEXT EXTRACTED\n"
            content += f"==================================================\n"
            content += f"{clean_plain_text}\n"
            return content, "text/plain", f"{stem}_ocr.txt"

        elif fmt == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Filename", "Document Type", "Project Name", "Plain Text"])
            writer.writerow([
                record.original_filename,
                record.document_type,
                record.project_name or record.department,
                clean_plain_text.replace("\n", " ")
            ])

            # Write Structured Fields if present
            s_data = result.structured_json
            if isinstance(s_data, dict) and s_data:
                writer.writerow([])
                writer.writerow(["Structured Field", "Value"])
                for k, v in s_data.items():
                    writer.writerow([k, str(v)])

            content = output.getvalue()
            return content, "text/csv", f"{stem}_ocr.csv"

        else:
            raise ValueError(f"Unsupported export format: {export_format}")


export_service = ExportService()
