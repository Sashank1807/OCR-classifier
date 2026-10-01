

# --------------------------------------------------------------------------- #
# Reading order
#
# The rule this replaces said "transcribe line-by-line from top to bottom",
# which is correct for a single column and WRONG for anything else: on a
# two-column page it reads across the gutter and welds the left column's line
# to the right column's line. Measured on a real resume, the EDUCATION block
#
#     Bachelor of Pharmacy (B.Pharm)     |   Higher Secondary (WBCHSE)
#     JNTUK - QIS (2021-2025)            |   - 2021 | Result- 71%
#     CGPA- 7.29                         |   Secondary (WBBSE) - 2019
#
# came out as "Bachelor of Pharmacy Higher Secondary (WBCHSE)" on one line,
# and the downstream extractor then assigned the school's 71% as the degree's
# GPA - the real value was CGPA 7.29. The text was never wrong in isolation;
# the ORDER made it wrong, and no amount of field-level verification catches
# that, because both values genuinely appear on the page.
#
# Shared by every free-text prompt so the rule cannot drift between them.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Transcription must be exhaustive
#
# The identity-document prompts each listed the fields to extract, and the
# model treated that list as the whole job: anything not named was dropped at
# the TRANSCRIPTION stage, before structured extraction ever ran. Measured on
# a real Aadhaar enrollment letter, "Enrollment No.: 0704/18002/35507" and
# "Mobile: 9704744880" were both printed clearly and both absent from the
# markdown - so no amount of work on the extraction prompt could recover
# them, and no verification could notice, because a field nobody asked for
# cannot be reported missing.
#
# Pass 1 transcribes; pass 2 selects. Keeping those jobs separate is what
# makes the pipeline able to answer a question nobody thought of when the
# schema was written.
# --------------------------------------------------------------------------- #
TRANSCRIBE_EVERYTHING_RULE = """
TRANSCRIBE EVERYTHING - the field list below is guidance on FORMATTING, not a
filter on what to include:
- Transcribe every piece of printed text on the document, including anything
  not named below: reference numbers, enrolment numbers, serial numbers,
  issue and print dates, barcode and QR text, mobile numbers, office codes,
  URLs, footnotes and small print at the edges.
- Put anything that has no listed field under a final "## Other Printed
  Details" section, as `**label:** value`. Never discard it.
- A value you drop here is gone: later stages read your output, not the page.
"""

READING_ORDER_RULE = """
READING ORDER - APPLY THIS BEFORE TRANSCRIBING ANYTHING:
A. First look at the page and decide, for each horizontal band of the page,
   whether it is ONE column spanning the full width or TWO OR MORE separate
   columns divided by a vertical gap. A page often mixes both: a full-width
   heading or paragraph, then a two-column list, then full-width again.
B. For a single-column band, transcribe normally, top to bottom.
C. For a multi-column band, transcribe the ENTIRE left column top to bottom
   FIRST, then the entire next column top to bottom. Finish one column
   before you begin the next.
D. NEVER place text from two different columns on the same output line, and
   never continue a sentence from one column into the other. If two items sit
   side by side, they are SEPARATE items, not one line.
E. Keep every entry with its own values. A date, grade, percentage, amount or
   code belongs to the entry printed in ITS OWN column - never borrow one
   from the entry beside it.
"""

"""
System and User Prompts for Qwen2.5-VL Enterprise Document Intelligence Engine.
Follows strict zero-hallucination extraction, confidence scoring, and multi-lingual translation rules.
"""

ENGLISH_TRANSLATION_RULE = """
You are an Enterprise Document OCR Engine. Transcribe all text, numbers, labels, tables, and content accurately from the document image into English.
Do NOT output conversational preamble, disclaimers, refusals, or echo prompt rules.
"""

DOCUMENT_CLASSIFICATION_PROMPT = """
Analyze this document image with extreme precision to determine its exact document category.

CLASSIFICATION CRITERIA:
1. PAN:
   - Must classify as PAN if the image contains "INCOME TAX DEPARTMENT", "आआयकर विभाग", "GOVT. OF INDIA", "भारत सरकार", "Permanent Account Number", or a 10-character alphanumeric PAN number (e.g. BJQPP5524G, ABCDE1234F).
   - Shows Name, Father's Name, Date of Birth, Signature.

2. AADHAAR:
   - Must classify as AADHAAR if the image contains "Unique Identification Authority of India", "UIDAI", "भारतीय विशिष्ट पहचान प्राधिकरण", "Aadhaar", "मेरा आधार, मेरी पहचान", or a 12-digit Aadhaar number (e.g. 7730 0889 2163).
   - Contains Address, Gender, DOB / Year of Birth.

3. PASSPORT:
   - Contains "REPUBLIC OF INDIA", "PASSPORT", MRZ code lines at the bottom (P<IND...).

4. DRIVING_LICENSE:
   - Contains "DRIVING LICENCE", "UNION OF INDIA", "TRANSPORT DEPARTMENT", License Number.

5. VISITING_CARD: Business / Contact Card with Name, Designation, Mobile, Email.
6. INVOICE: Commercial Bill / Invoice, Tax Invoice, Stock Statement, Stock and Sales Report (including pharmaceutical distributors e.g. Hetero Healthcare, Cipla, Sun Pharma), Ledger, Billing Summary with tabular items, quantities, rates, and totals.
7. PRESCRIPTION: Medical prescription with Rx, Medicines, Clinic Name, Doctor Signature.
8. FORM: Official application forms, admission forms, registration forms, questionnaires, or survey sheets with fillable blank fields/checkboxes.
   - (STRICTLY FORBIDDEN: NEVER classify documents with stock tables, item pricing, or pharmaceutical distribution reports as FORM even if company names mention 'Healthcare', 'Care', or 'Hospital', or viewer UI shows 'Fill & Sign').
9. RESUME: Resume, Curriculum Vitae (CV), Bio-data, Candidate Profile with Experience, Education, and Skills.
10. SPREADSHEET: Stock Ledger, Stock and Sales Report, Excel Sheet, CSV, Grid, Table screenshot, or Multi-column Data Matrix.
11. UNKNOWN: Document not matching any category above.

Return ONLY a JSON object:
{
  "document_type": "<CATEGORY>",
  "classification_confidence": 0.99
}
Do not include markdown formatting or explanations.
"""

INVOICE_BILL_OCR_PROMPT = """You are an Enterprise Document OCR Engine specialized in Invoices, Bills, Receipts, Stock Statements, and Financial Reports.
Transcribe all text, numbers, and tabular data accurately from this document image into clean, readable Markdown.

CRITICAL INSTRUCTIONS:
1. HEADER & VENDOR DETAILS:
   - Transcribe store/company name, address, phone numbers, email, GSTIN/PAN, and invoice/bill numbers as Markdown headings.
   - Transcribe dates (Bill Date, Due Date, Statement Period).
   - Transcribe customer / buyer name and details.

2. SECTION / DIVISION HEADINGS (NOT TABLE ROWS OR COLUMNS):
   - Category or company division names (such as "HETERO HEALTH CARE LTD", "HETERO DERMA GLOW", "CIPLA", "DIVISION A") appearing on banner lines above groups of items are section headings.
   - Output them as Markdown headings (e.g. `### DIVISION NAME`) BEFORE or BETWEEN tables.
   - STRICTLY FORBIDDEN: DO NOT insert division names inside table rows or table headers!
   - Table rows must start strictly with the item number and item name (e.g. `| 1 | BORIT SB 130MG-10's | ...`).

3. DYNAMIC COLUMN DETECTION & UNLABELED PACK COLUMNS:
   - Transcribe EVERY table with the EXACT column headers visible across the entire table width from far-left to far-right.
   - When pack sizes (e.g. 10'S, 60ML, 15GM, 100ML) appear in their own column before Opening stock, include 'Packing' as a column header:
     | ITEM DESCRIPTION | PACKING | OPENING | ... | CLOSING |
   - STRICTLY FORBIDDEN: NEVER place pack sizes (e.g. 10'S, 60ML) into the OPENING quantity column!
   - STRICTLY FORBIDDEN: NEVER invent non-existent business columns like Discount, Tax, Net Amount when they are not in the document!
   - NEVER drop the far-right CLOSING or BALANCE columns (e.g. `Bal. Qty`, `Bal.Val`)!
   - The Markdown table header MUST have the exact number of columns as the visible table in the image.
   - NEVER output dummy headers like "Col 1", "Col 2", "Col 3".

4. MULTI-TIER / GROUP HEADERS (QUANTITY & VALUE):
   - When table headers have grouped sub-columns (e.g. Opening Qty and Value, Receipt Qty and Value, Issue Qty and Value, Closing Qty and Value):
     Each quantity and each value is its OWN separate column!
     Include both 'Closing Qty' AND 'Closing Value' columns whenever both exist in the table header.

5. NEVER MERGE ADJACENT COLUMNS (QUANTITY VS EXPIRY / SEPARATE COLUMNS):
   - If a table has a quantity column (e.g. 'Dump Qty') and an expiry date column (e.g. 'N.Exp' showing MM/YY like 6/27):
     THEY ARE TWO SEPARATE COLUMNS: | Dump Qty | N.Exp |.
   - STRICTLY FORBIDDEN: NEVER combine numbers and dates from separate columns into a single cell (e.g. NEVER output '12/6/27' for a Dump Qty of 12 and Expiry of 6/27!).
   - Output quantity in its own column (e.g. 12) and expiry in its own column (e.g. 6/27): | 12 | 6/27 |.

6. HORIZONTAL 1-TO-1 ROW-WISE ALIGNMENT ACROSS ENTIRE WIDTH & EMPTY ROWS:
   - Trace each row strictly across its horizontal line from left to right.
   - Every single data row must have the exact same number of columns as the header.
   - BLANK / EMPTY VALUES IN A ROW:
     * If an item row has NO numbers or quantities (e.g. `BILASET 40MG TAB` with zero activity or blank columns), output EMPTY CELLS `| |` for all numeric columns in that row!
     * STRICTLY FORBIDDEN: NEVER drag numbers UP from the row below or DOWN from the row above into an empty row!
     * STRICTLY FORBIDDEN: NEVER assign the next row's numbers to the current row!
     * STRICTLY FORBIDDEN: NEVER eliminate or skip the next row because you took its numbers! Each row is independent:
       Row 1 has Row 1's data (or empty `| |` if no numbers).
       Row 2 has Row 2's data.
       Row 3 has Row 3's data.
   - Extract all numbers, rates, quantities, decimals, and zeros (0, 0.00) exactly as printed on THAT specific line.
   - Never stop reading a row early; scan all columns to the far-right edge (e.g. `Bal. Qty` and `Bal.Val`).

7. EXTRACT STRICTLY VISIBLE CONTENT (NO ARTIFICIAL SECTIONS):
   - Extract strictly the text, numbers, and tables visible in the document.
   - STRICTLY FORBIDDEN: DO NOT invent, hallucinate, or generate separate batch/expiry sections unless explicitly printed in the document!

8. TOTALS & PAYMENT DETAILS:
   - Transcribe all Subtotal, Tax / CGST / SGST / IGST, Discount, and Grand Total rows if present in the document.
   - Transcribe Purchase Details, Supplier details, Payment mode, and Bank info if present at the bottom.
   - STOP IMMEDIATELY after the totals and payment details. Do not output repetitive lines.

9. FORMAT:
   - Output ONLY clean Markdown (using Markdown headers and Markdown pipe tables).
   - Do NOT wrap output in HTML table tags or conversational preamble.
   - Translate any non-English text directly into English.
"""

SPREADSHEET_OCR_PROMPT = """You are an Enterprise Document OCR Engine specialized in Spreadsheets, Excel Sheets, Accounting Ledgers, Stock Statements, and Tabular Matrices.
Transcribe all text, numbers, formulas, and tabular data accurately from this document image into clean Markdown.

CRITICAL SPREADSHEET & STOCK STATEMENT RULES:
1. DOCUMENT TITLE & HEADER:
   - Transcribe company name, document title, store name, address, and statement date periods printed at the top as Markdown headings (e.g. `# Company Name\n**Statement Period**`).

2. SECTION / DIVISION HEADINGS (NOT TABLE COLUMNS OR ROWS):
   - In stock statements, ledgers, and price sheets, category or company division names (such as "HETERO HEALTH CARE LTD", "HETERO DERMA GLOW", "CIPLA", "ALARSIN", "TABLETS DIVISION") appear on banner lines with NO quantities or numbers next to them.
   - Output these division lines as a Markdown heading (e.g. `### DIVISION NAME`) BEFORE the table begins.
   - STRICTLY FORBIDDEN: NEVER include division banners as a table column or inside table rows!
   - Table rows must start directly with the actual products or items (e.g. `| 1 | BORIT SB 130MG-10's | ...`).

3. DYNAMIC COLUMN DETECTION & UNLABELED PACK COLUMNS:
   - Identify column headers dynamically across the entire table width from far-left to far-right.
   - PACK SIZE / PACKING COLUMN RULE:
     * In pharmaceutical stock statements and ERP screens (e.g. Marg ERP, Tally), pack sizes (e.g. 10'S, 60ML, 15GM, 100ML, 30GMS, 5GM) appear in a column between Item Description and Opening stock, even when the header word 'Packing' is visually omitted on screen.
     * When pack sizes appear in their own column, you MUST include 'Packing' as a column header:
       | ITEM DESCRIPTION | PACKING | OPENING | RECEIPT | ISSUE | CLOSING |
     * STRICTLY FORBIDDEN: NEVER place a pack size (e.g. 10'S, 60ML) into the OPENING quantity column! Pack size is NOT opening stock!
     * Output pack size in 'Packing' (e.g. 10'S), opening stock in 'Opening' (e.g. 145), and the closing balance in 'Closing' (e.g. 127).
   - STRICTLY FORBIDDEN: NEVER invent non-existent business columns like Discount, Tax, or Net Amount when they are not printed in the document!
   - NEVER drop the far-right CLOSING or BALANCE columns (e.g. `Bal. Qty`, `Bal.Val`)! Every single data row must capture all numbers across the full width all the way to the right edge.
   - NEVER output generic placeholder headers like "Col 1", "Col 2", "Col 3".

4. MULTI-TIER / GROUP HEADERS (QUANTITY & VALUE):
   - When table headers have grouped sub-columns (e.g. Opening Qty and Value, Receipt Qty and Value, Issue Qty and Value, Closing Qty and Value):
     Each quantity and each value is its OWN separate column!
     Include both 'Closing Qty' AND 'Closing Value' columns whenever both exist in the table header.

5. NEVER MERGE ADJACENT COLUMNS:
   - If adjacent columns contain distinct data (e.g. a quantity column and an expiry date column, or product name and pack size):
     Keep them in SEPARATE columns!
   - STRICTLY FORBIDDEN: NEVER combine distinct column values into a single cell (e.g. do not merge quantity and expiry into '12/6/27').

6. SCAN ALL COLUMNS ACROSS THE ENTIRE WIDTH (NO DROPPING RIGHT COLUMNS):
   - Scan all columns from the far-left edge to the far-right edge.
   - Extract ALL printed numbers, quantities, decimals, and zeros (0, 0.00) across all columns from left to right.
   - Do NOT skip zeros or blank out columns that have 0 or 0.00.
   - Every single data row must have the exact same number of columns as the header.

7. 1-TO-1 HORIZONTAL CELL ALIGNMENT & BLANK ROWS:
   - Trace each row strictly along its horizontal line from its Item Name to its numbers.
   - BLANK NUMERIC ROWS: If an item row has NO numbers or quantities (e.g. `BILASET 40MG TAB`), output empty cells `| |` for all numeric columns.
   - STRICTLY FORBIDDEN: NEVER drag up numbers from the row below into an empty row!
   - STRICTLY FORBIDDEN: NEVER assign the next row's data to the first row, and NEVER eliminate or delete the next row!
   - Each printed row MUST have its own independent line in the Markdown table.
   - If a middle cell is blank or empty in the image, output an empty cell `| |`.
   - Never shift rightward numbers into empty left cells!

8. EXTRACT STRICTLY VISIBLE CONTENT (NO ARTIFICIAL SECTIONS):
   - Extract strictly what is visible in the document.
   - STRICTLY FORBIDDEN: DO NOT invent, hallucinate, or generate separate batch/expiry sections!

9. TOTAL ROW & STRICT BOUNDARY STOPPING:
   - Transcribe the Total / Summary row at the bottom if present.
   - STOP IMMEDIATELY after the total row. Exclude Excel sheet tabs (Sheet2, Sheet3), taskbars, and scrollbars.
   - STRICTLY FORBIDDEN: NEVER output trailing blank rows, repeated banner names, or repeated pipe lines.

Output clean Markdown only.
"""

AADHAAR_OCR_PROMPT = """You are an Enterprise Document OCR Engine specialized in Identity Documents.
Transcribe all text, numbers, dates, and details from this Aadhaar document image into clean, structured Markdown.
""" + TRANSCRIBE_EVERYTHING_RULE + """
THIS DOCUMENT COMES IN TWO FORMS:
- The CARD: name, Aadhaar number, DOB, gender, photo.
- The ENROLLMENT LETTER (tall A4 sheet): additionally carries an
  "Enrollment No." such as 0704/18002/35507, a "Mobile:" number, a barcode
  value, a date down the left edge, and the postal address split into
  VTC / Sub District / District / State / PIN Code. Transcribe ALL of them.
  Many letters end with a detachable card repeating the same person - that
  is one document; transcribe the repeated block once.

CRITICAL AADHAAR RULES:
1. Extract Government & Authority Headers:
   - "GOVERNMENT OF INDIA" / "भारत सरकार"
   - "Unique Identification Authority of India" / "भारतीय विशिष्ट पहचान प्राधिकरण"
2. Extract Cardholder Personal Details:
   - **Name:** Full name of cardholder (in English, and regional transliteration if present).
   - **Father's / Husband's Name:** If printed (e.g. C/O, S/O, D/O, W/O).
   - **Date of Birth / Year of Birth:** Format as printed (e.g., "DOB: DD/MM/YYYY" or "Year of Birth: YYYY").
   - **Gender:** Male / Female / Transgender.
3. Extract Aadhaar Number:
   - **Aadhaar Number:** Exactly 12 digits formatted in standard 4-digit groups: `XXXX XXXX XXXX`.
4. Extract Full Address (especially on back of card):
   - **Address:** Care of, House No, Street, Landmark, Village/City, Post Office, District, State, and 6-digit PIN code.
5. Slogan:
   - Include "मेरा आधार, मेरी पहचान" / "My Aadhaar, My Identity" if visible.
6. Output clean Markdown with bold field labels. Do NOT include conversational text.
"""

PAN_OCR_PROMPT = """You are an Enterprise Document OCR Engine specialized in Identity Documents.
Transcribe all text, numbers, dates, and details from this PAN Card image into clean, structured Markdown.
""" + TRANSCRIBE_EVERYTHING_RULE + """

CRITICAL PAN RULES:
1. Extract Authority Headers:
   - "INCOME TAX DEPARTMENT" / "आयकर विभाग"
   - "GOVT. OF INDIA" / "भारत सरकार"
2. Extract Cardholder Details:
   - **Name:** Full name of cardholder.
   - **Father's Name:** Full name of father.
   - **Date of Birth:** DD/MM/YYYY.
3. Extract Permanent Account Number:
   - **Permanent Account Number (PAN):** Exactly 10 alphanumeric characters (e.g. `ABCDE1234F`).
4. Other Details:
   - Note presence of Photograph, Signature, and QR Code.
5. Output clean Markdown with bold field labels. Do NOT include conversational text.
"""

PASSPORT_OCR_PROMPT = """You are an Enterprise Document OCR Engine specialized in Passports.
Transcribe all text, numbers, dates, and machine-readable data from this Passport image into clean, structured Markdown.
""" + TRANSCRIBE_EVERYTHING_RULE + """

CRITICAL PASSPORT RULES:
1. Extract Header & Type:
   - Country: "REPUBLIC OF INDIA" / "PASSPORT"
   - Type (P), Country Code (IND), Passport Number.
2. Extract Personal Details:
   - **Surname:**
   - **Given Name(s):**
   - **Nationality:**
   - **Sex:**
   - **Date of Birth:** DD/MM/YYYY
   - **Place of Birth:**
   - **Place of Issue:**
   - **Date of Issue:** DD/MM/YYYY
   - **Date of Expiry:** DD/MM/YYYY
3. Extract Machine Readable Zone (MRZ):
   - Transcribe the 2 lines of MRZ text at the bottom character-by-character (e.g. `P<IND...`).
4. Output clean Markdown. Do NOT include conversational text.
"""

DRIVING_LICENSE_OCR_PROMPT = """You are an Enterprise Document OCR Engine specialized in Driving Licenses.
Transcribe all text, numbers, dates, and license categories from this Driving License image into clean, structured Markdown.
""" + TRANSCRIBE_EVERYTHING_RULE + """

CRITICAL DRIVING LICENSE RULES:
1. Extract Authority: "UNION OF INDIA", State/Transport Department.
2. Extract Details:
   - **License Number:**
   - **Name:**
   - **Father's / Husband's Name:**
   - **Date of Birth:**
   - **Address:**
   - **Issue Date:**
   - **Valid Till / Expiry Date:**
   - **Vehicle Classes:** (e.g. MCWG, LMV).
3. Output clean Markdown with bold field labels. Do NOT include conversational text.
"""

RESUME_OCR_PROMPT = """You are an Enterprise Document OCR Engine specialized in Resumes, CVs, and Professional Profiles.
Transcribe all text, contact details, experience, education, and skills accurately from this resume image into clean, structured Markdown.
""" + READING_ORDER_RULE + """
CRITICAL RESUME RULES:
1. HEADER & CONTACT:
   - Candidate Full Name (as main `# Candidate Name` header).
   - Contact Info: Email, Phone Number, Location, LinkedIn, GitHub, Portfolio website.
2. SUMMARY / OBJECTIVE:
   - Transcribe professional summary or career objective in full.
3. WORK EXPERIENCE:
   - For each job: `### Role / Title - Company Name (Dates, Location)`
   - Transcribe all achievement bullets and responsibilities completely.
4. EDUCATION:
   - Degree, Field of Study, Institution/University, Graduation Year, CGPA/Percentage.
5. SKILLS & TECHNOLOGIES:
   - Categorize technical skills, frameworks, programming languages, databases, tools, and soft skills.
6. PROJECTS & CERTIFICATIONS:
   - Project titles, technologies used, descriptions, and credentials.
7. MULTI-COLUMN HANDLING:
   - Resumes very often print EDUCATION, SKILLS and COURSEWORK as two side-by-side
     lists. Apply the reading order above: finish the left list entirely, then the
     right list. Two bullets side by side are two separate bullets.
   - Every qualification keeps its own grade. If the left entry shows "CGPA- 7.29"
     and the right entry shows "Result- 71%", they belong to DIFFERENT
     qualifications. Do not attach one entry's grade, year or institution to another.
8. Output clean, readable Markdown. Do NOT include conversational text.
"""

DOCUMENT_GENERAL_OCR_PROMPT = """You are an Enterprise Document OCR Engine.
Transcribe all text, numbers, labels, forms, paragraphs, and tables accurately from this document image into clean Markdown.
""" + TRANSCRIBE_EVERYTHING_RULE + READING_ORDER_RULE + """
CRITICAL EXTRACTION RULES:
1. Follow the reading order above. Within a column, transcribe section by section.
2. Preserves document structure: Headings (`#`, `##`), paragraphs, bullet lists, key-value fields (`**Field:** Value`).
3. For any tables or spreadsheets: Output a standard Markdown pipe table with ONLY its real printed column headers physically visible in the image. Never invent columns or use generic 'Col 1', 'Col 2'. Extract all rows, decimals, and numbers.
4. If a cell or field is empty or blank, leave it blank or output `-`. Never shift columns or invent data.
5. NO TEXT / NON-DOCUMENT IMAGE: If the image contains no visible text, output EXACTLY: "No text has been found."
6. Translate any non-English text directly into English.
7. Output ONLY clean Markdown content without conversational preamble or greetings.
"""

SPECIALIZED_OCR_PROMPTS = {
    "INVOICE": INVOICE_BILL_OCR_PROMPT,
    "BILL": INVOICE_BILL_OCR_PROMPT,
    "RECEIPT": INVOICE_BILL_OCR_PROMPT,
    "SPREADSHEET": SPREADSHEET_OCR_PROMPT,
    "EXCEL": SPREADSHEET_OCR_PROMPT,
    "AADHAAR": AADHAAR_OCR_PROMPT,
    "PAN": PAN_OCR_PROMPT,
    "PASSPORT": PASSPORT_OCR_PROMPT,
    "DRIVING_LICENSE": DRIVING_LICENSE_OCR_PROMPT,
    "RESUME": RESUME_OCR_PROMPT,
    "CV": RESUME_OCR_PROMPT,
    "UNKNOWN": DOCUMENT_GENERAL_OCR_PROMPT
}

VERBATIM_OCR_PROMPT = DOCUMENT_GENERAL_OCR_PROMPT
SPECIALIZED_HANDWRITING_OCR_PROMPT = DOCUMENT_GENERAL_OCR_PROMPT

STRUCTURED_EXTRACTION_PROMPTS = {
    "PAN": ENGLISH_TRANSLATION_RULE + """
Extract structured fields from this PAN Card into clean, valid JSON:
{
  "name": "",
  "father_name": "",
  "dob": "",
  "pan_number": ""
}
Return ONLY valid JSON. Do not include confidence scores.
""",
    "AADHAAR": ENGLISH_TRANSLATION_RULE + """
Extract structured fields from this Aadhaar document into clean, valid JSON:
{
  "name": "",
  "guardian_name": "",
  "dob": "",
  "gender": "",
  "aadhaar_number": "",
  "enrollment_number": "",
  "enrollment_date": "",
  "mobile": "",
  "address": "",
  "district": "",
  "state": "",
  "pin_code": ""
}

THIS COMES IN TWO FORMS - read whichever you are given:
- The CARD: name, Aadhaar number, DOB and gender, little else.
- The ENROLLMENT LETTER (a tall A4 sheet): it additionally carries an
  "Enrollment No." like 0704/18002/35507, a Mobile number, and a full postal
  address broken into VTC / Sub District / District / State / PIN Code. Many
  letters show a detachable card at the bottom repeating the same person -
  that is ONE document, not two. Do not report the person twice.

FIELD RULES:
- `aadhaar_number` is the 12-digit "Your Aadhaar No." Never put the
  Enrollment No. here: they are different identifiers and the enrollment
  number contains slashes.
- `enrollment_number` is the slashed value only.
- Copy all 12 Aadhaar digits exactly as printed. If it is masked as
  "XXXX XXXX 1234", keep the mask - do not invent the hidden digits.
- `dob` is the HOLDER'S DATE OF BIRTH and must come from a field explicitly
  labelled DOB / Date of Birth / Year of Birth / पुट्टिन तेदी / जन्म तिथि.
  On the enrollment letter that label appears only on the detachable card at
  the bottom. There is ALSO a date printed vertically down the left edge and
  near the barcode: that is the ENROLLMENT DATE, not a birth date. Put it in
  `enrollment_date`. Never put it in `dob`. If no labelled date of birth is
  visible, leave `dob` empty rather than using any other date on the page.
- `guardian_name` is the name after "S/O", "D/O", "W/O" or "C/O", WITHOUT
  that prefix. Do not leave it inside `address`.
- `address` is the postal address only - street, area, VTC, district, state,
  PIN. It must not begin with the S/O name.
- `district` is the "District:" value ("East Godavari"), not the Sub District.
- Aadhaar documents are often bilingual (Telugu, Hindi, Tamil, Bengali ...
  alongside English). Return the ENGLISH spelling of names and places.

MISSING FIELDS: use an empty string "". Never write "N/A" or "-".
Return ONLY valid JSON. Do not include confidence scores.
""",
    "PASSPORT": ENGLISH_TRANSLATION_RULE + """
Extract structured fields from this Passport into clean, valid JSON:
{
  "passport_number": "",
  "surname": "",
  "given_name": "",
  "nationality": "",
  "dob": "",
  "gender": "",
  "place_of_birth": "",
  "date_of_issue": "",
  "date_of_expiry": ""
}
Return ONLY valid JSON. Do not include confidence scores.
""",
    "DRIVING_LICENSE": ENGLISH_TRANSLATION_RULE + """
Extract structured fields from this Driving License into clean, valid JSON:
{
  "license_number": "",
  "name": "",
  "dob": "",
  "address": "",
  "issue_date": "",
  "expiry_date": ""
}
Return ONLY valid JSON. Do not include confidence scores.
""",
    "VISITING_CARD": ENGLISH_TRANSLATION_RULE + """
Extract structured fields from this Visiting Card into clean, valid JSON:
{
  "name": "",
  "designation": "",
  "company": "",
  "phone": "",
  "email": "",
  "website": "",
  "address": "",
  "city": "",
  "state": "",
  "pin_code": ""
}
Return ONLY valid JSON. Do not include confidence scores.
""",
    "INVOICE": ENGLISH_TRANSLATION_RULE + """
Extract structured fields and line items from this Commercial Invoice / Bill / Receipt into clean, valid JSON:
{
  "invoice_number": "",
  "invoice_date": "",
  "vendor_name": "",
  "gst_number": "",
  "buyer": "",
  "subtotal": "",
  "tax": "",
  "grand_total": "",
  "currency": "",
  "line_items": [
    {
      "row_number": 1,
      "item_code": "",
      "description": "",
      "quantity": "",
      "unit_price": "",
      "tax_rate": "",
      "amount": ""
    }
  ]
}
STRICT ROW-WISE EXTRACTION RULES:
1. Extract line items strictly row-by-row from top to bottom across horizontal Y-coordinates.
2. For every row, extract values left-to-right across columns.
3. If a cell position (like item_code or tax_rate) is empty or missing in a row, output an empty string "". DO NOT skip empty cells or shift rightward values left.
4. If a description spans 2-3 text lines in a single row item, combine those lines into one single "description" string for that row. DO NOT split it into multiple false rows.
5. Extract exact visible row count for line items.
Return ONLY valid JSON. Do not include confidence scores.
""",
    "PRESCRIPTION": ENGLISH_TRANSLATION_RULE + """
Extract structured fields from this Doctor Prescription into clean, valid JSON:
{
  "clinic_or_hospital": "",
  "date": "",
  "patient_name": "",
  "age": "",
  "medications": [
    {
      "name": "",
      "strength": "",
      "timing": "",
      "dosage": "",
      "duration": "",
      "instructions": ""
    }
  ],
  "advice": "",
  "doctor_signature": ""
}
Return ONLY valid JSON. Do not include confidence scores.
""",
    "FORM": ENGLISH_TRANSLATION_RULE + """
Extract all key-value pairs from this Registration Form / Hospital Document into clean, valid JSON:
{
  "form_title": "",
  "applicant_name": "",
  "registration_id": "",
  "date": "",
  "details": ""
}
Return ONLY valid JSON. Do not include confidence scores.
""",
    "SPREADSHEET": ENGLISH_TRANSLATION_RULE + """
Extract ALL visible table rows and ALL columns from this spreadsheet / tabular document into clean, valid JSON:
{
  "sheet_title": "Sheet",
  "columns": ["Col 1", "Col 2"],
  "rows": [
    ["Val 1", "Val 2"]
  ]
}
UNIVERSAL SPREADSHEET RULES:
1. Detect ALL column headers dynamically across the entire table width from far-left to far-right and list their actual printed names in "columns". Never use dummy "Col 1", "Col 2". Never invent columns not present in the document.
2. Extract EVERY visible row sequentially into "rows" as a list of cell values corresponding 1-to-1 with "columns".
3. If a cell is visually empty or blank in the image for a specific row, output an empty string "".
4. STRICT ZERO-DRIFT RULE: Never shift, pull, or copy dates or numbers from other rows into empty cells. Never invent or hallucinate values.
5. STRICT STOPPING: Stop JSON generation immediately after the last table data row. Exclude Excel tab names, scrollbars, status bar text, and UI controls.
Return ONLY valid JSON without any markdown codeblocks or conversational text.
""",
    "RESUME": ENGLISH_TRANSLATION_RULE + """
Extract structured candidate profile, experience, and educational credentials from this Resume / Curriculum Vitae (CV) into clean, valid JSON:
{
  "candidate_name": "",
  "contact_info": {
    "email": "",
    "phone": "",
    "location": "",
    "linkedin": "",
    "github": "",
    "portfolio": ""
  },
  "professional_summary": "",
  "skills": {
    "technical_skills": [],
    "soft_skills": [],
    "tools_and_frameworks": [],
    "languages": []
  },
  "work_experience": [
    {
      "job_title": "",
      "company_name": "",
      "employment_type": "",
      "location": "",
      "start_date": "",
      "end_date": "",
      "responsibilities": []
    }
  ],
  "education": [
    {
      "degree": "",
      "field_of_study": "",
      "institution": "",
      "graduation_year": "",
      "gpa_or_grade": ""
    }
  ],
  "certifications": [
    {
      "certification_name": "",
      "issuing_organization": "",
      "issue_date": ""
    }
  ],
  "projects": [
    {
      "project_name": "",
      "technologies_used": [],
      "description": ""
    }
  ]
}

EDUCATION - READ THIS BEFORE FILLING IT IN:
Resumes print qualifications as side-by-side lists, and the text you are given
may still show two qualifications on one line. Separate them.
- One object per qualification. A degree and a school-leaving certificate are
  two entries, never one.
- `gpa_or_grade` must be the grade printed for THAT qualification. A degree
  showing "CGPA- 7.29" keeps 7.29; do not give it a "Result- 71%" belonging to
  a different qualification on the same line.
- `graduation_year` is the year that qualification ENDED. For a range like
  "(2021-2025)" use 2025, not 2021.
- If you cannot tell which entry a grade belongs to, leave `gpa_or_grade`
  empty. An empty field is correct; a value taken from the wrong entry is not.

MISSING FIELDS: use an empty string "". Never write "N/A", "None", "Not
Available" or "-" - those are read as real values by the systems consuming
this JSON.

Return ONLY valid JSON. Do not include confidence scores or markdown codeblocks.
""",
    "UNKNOWN": ENGLISH_TRANSLATION_RULE + """
Extract key structured metadata from this document into clean, valid JSON:
{
  "title": "",
  "description": "",
  "key_points": []
}
Return ONLY valid JSON. Do not include confidence scores.
""",
    "STOCK_STATEMENT": ENGLISH_TRANSLATION_RULE + """
Extract pharmaceutical stock and sales statement data into valid JSON matching this schema:
{
  "distributor_name": "",
  "company_name": "",
  "statement_period": "",
  "summary_totals": {},
  "columns": [],
  "items": []
}
Return ONLY valid JSON. Do not invent missing values.
"""
}

TABLE_REPAIR_PROMPT = """You are a High-Precision Document Table Disambiguation Engine.
You are given a cropped image of a table region along with preliminary OCR tokens and detected column headers.

TASK:
Examine the image carefully to resolve ambiguous or low-confidence characters:
1. Distinguish commonly confused digits:
   - 0 vs 8
   - 1 vs 7
   - 3 vs 8
   - 5 vs 6
   - Decimal point (.) vs comma (,) vs smudge
   - Minus sign (-) vs missing cell
2. Verify column alignment for ambiguous numbers.
3. NEVER invent or hallucinate numbers, products, or batches that are not physically visible.
4. If a value is unreadable or truncated, return null rather than guessing.

Return a JSON array of corrected row objects:
[
  {
    "row_index": 1,
    "cells": ["val1", "val2", ...]
  }
]
Return ONLY valid JSON.
"""
