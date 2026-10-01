# OCR Microservice — Bruno Collection

Verified against the running service on 2026-09-10. Every URL, parameter,
default and response shape below was checked against `app/api/routes_ocr.py`,
`routes_history.py` and `routes_analytics.py`, and the main flow was executed
end to end.

## Setup

Pick an environment in Bruno's top-right selector:

| Environment | baseUrl |
|---|---|
| Local  | `http://127.0.0.1:8080` |
| Server | `http://192.168.172.246:8080` |

Requests 01 and 03 store `requestId` and `docId` into the environment
automatically, so 02, 05, 06, 07, 11 and 12 run without editing any URL. Run
them in order.

Sample files are referenced relative to this folder (`../test_data_june/...`)
so the collection works on any machine with the repo checked out.

## Requests

| # | Request | Purpose |
|---|---|---|
| 01 | Submit Document (Async) | Returns `request_id` immediately |
| 02 | Get OCR Status & Result | Poll until `COMPLETED` |
| 03 | Submit Document (Sync) | Blocks, returns the full result inline |
| 04 | Submit Multiple (Batch) | Repeated `files` field, different response shape |
| 05 | Get Document Details | Full record by numeric document id |
| 06 | **Export Structured JSON** | **The extracted table** |
| 07 | Reprocess & Rotate | Re-run an existing document |
| 08 | Search History | Full-text search |
| 09 | Analytics (all projects) | Metrics and token ledger |
| 10 | Analytics (one project) | Project-scoped metrics |
| 11 | **Get ERP Data** | **Business fields only, for an ERP write** |
| 12 | Delete Document | Destructive — deliberately last |

## Four things that catch integrators out

**1. Only `file` is required.** Everything else has a server-side default, so
those fields are marked optional in the collection and can be toggled off:

| Field | Required | Default |
|---|---|---|
| `file` | **Yes** | — |
| `project_name` | No | `DefaultProject` |
| `user_id` | No | `system` |
| `document_type` | No | `AUTO` |
| `async_mode` | No | **`false`** |
| `response_format` | No | `full` |

**2. `async_mode` defaults to `false`.** Omit it and you get the synchronous
response — `success` + `data`, with no `request_id` at all. That is a different
shape, not a variation of the async one.

**3. While processing, `status` is not the bare string `"PROCESSING"`.** It
carries progress, e.g. `"PROCESSING (Page 1 of 1)"`. Poll with
`status.startsWith("PROCESSING")`, never `==`. `"COMPLETED"` and `"FAILED"`
are exact.

**4. Every result endpoint returns the extracted table.** `/status/...`,
`/document/...` and the synchronous `/process` response all carry
`structured_data` (`fields.columns`, `fields.rows`, and per document type
`fields.items` / `fields.summary_totals`). Request 06
(`/export/{doc_id}?format=json`) is for when you want the export *file*.

`/status/` accepts either the `request_id` from `/process` (`req_...`) or the
numeric document id — issued request_ids are always `req_<hex>`, so an
all-digit value is unambiguously a document id.

## Two response formats

`response_format` picks who the response is written for. It is accepted as a
form field on `/process` and as a query parameter on `/status` and
`/document`; `GET /api/v1/ocr/erp/{doc_id}` is the same thing on its own URL.

| | `full` (default) | `erp` |
|---|---|---|
| Written for | a person reviewing the document | an ERP writing a record |
| Carries | markdown, `structured_data`, `field_verification`, pages | named values, `line_items`, `confidence`, `review_required` |
| Table shape | `fields.columns` + `fields.rows`, two parallel lists | `line_items`, objects keyed by column name |
| Empty fields | present as `""` / `{}` | omitted |

The `erp` shape is stable across document types - read `data` and
`line_items` without branching on `document_type` first - while the keys
*inside* `data` stay the ones the document actually yielded.

Three things to know before you map against it:

* **Duplicate column names are suffixed.** A PDF statement here produced six
  columns all named `Qty`; they arrive as `Qty`, `Qty_2` ... `Qty_6`. Blank
  header cells become `column_N`. No column is ever silently dropped.
* **Values are verbatim.** No number parsing, no date reformatting.
  `"1,234.50"` stays that string. Typing is yours, because only you know your
  ERP's schema.
* **`review_required` is the field to gate on.** When it is true,
  `review_fields` names which fields to hold - the rest of the document can
  still be posted.

An unrecognised `response_format` returns **400** rather than quietly falling
back to `full`.

## Payload contract: markdown + structured JSON

Four redundant or internal fields were removed from API responses. Each was
either an exact duplicate of data already in the payload or internal machinery,
and all of them grew with document size:

| Removed | Was | Read instead |
|---|---|---|
| `plain_text` | `markdown` with formatting stripped | `markdown` |
| `items[].raw_cells` | byte-for-byte copy of `rows[i]` | `rows` |
| `tables[].cells_metadata` | per-cell OCR tokens: pixel bboxes, `quad_slope`, per-token confidence, glyph sizes | `rows` |
| `tables` (whole block) | same columns/rows as `structured_data.fields`, same markdown as the document markdown | `structured_data.fields` |

`rows` is positionally aligned with `columns`.

One 20-row statement was carrying the same table **four times**: as document
`markdown`, as `fields.rows`, as `fields.all_tables[0].rows`, and again as
`tables[0]`. Together the four removals took that response from **34,795 to
3,625 bytes — 90% smaller**, with nothing lost.

`column_model_uncertain` lived only on `tables` and is a real review signal (a
candidate column was detected but rejected for want of corroboration, so the
table may be *missing* a column). It is now published on `structured_data`,
alongside `needs_manual_review`.

All of them are still produced and used server-side — `cells_metadata` drives the
closed-loop cell re-OCR, history search queries the plain text, and the
`txt`/`csv` export formats still render it, since there it is the requested
output. Only what crosses the wire changed.

**Documents processed before this change still have the old fields stored.**
Re-submit a document to see the slim payload; fetching an old `doc_id` returns
what was stored at the time.

## Field aliases

`project_name` is also accepted as `Project-name`, `project-name`,
`ProjectName`, `project` or `department`, or via the `X-Project-Name` header.
`user_id` is also accepted via the `X-User-Id` header. Prefer the plain
`project_name` / `user_id` form.

## Error responses

| Code | When |
|---|---|
| 400 | No valid file in the upload |
| 404 | Unknown `request_id`, document id, or missing source file on reprocess |
| 500 | Processing failed |
