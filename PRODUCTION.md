# Running this in production

What changed to make it deployable, what you must configure, and what is still
open. Written 2026-09-30 against the running service.

---

## 1. Configure before starting

All of these live in `.env`. The app **refuses to start** when `APP_ENV` is
`production` and any of the first four are missing — serving openly is worse
than not serving.

| Variable | Why it matters |
| --- | --- |
| `SECRET_KEY` | Signs viewer sessions and media links. Keep it **stable** — rotating it signs everyone out and breaks outstanding image links. Generate with `python -c "import secrets;print(secrets.token_urlsafe(48))"` |
| `API_KEYS` | Comma-separated. **Issue one per consuming department** so a single key can be revoked without disturbing the others. |
| `UI_PASSWORD` | Password for the browser viewer. |
| `DEBUG` | Must be `false`. `true` enables autoreload — see §4. |
| `APP_ENV` | `production` turns on the startup safety check and the `Secure` cookie flag. |

Optional but worth setting: `CORS_ORIGINS` (empty = no browser callers, which
is what a server-to-server API wants), `ENABLE_DOCS=false` if the host is
reachable from outside your network.

---

## 2. Authentication

Three doors, three credentials, because they have three different callers.

**Services** send `X-API-Key: <key>` (or `?api_key=` where headers are
awkward). Every `/api/v1/*` route requires it — the dependency is attached at
the *router*, so a new route cannot be added unguarded by forgetting a
decorator.

**People** sign in at `/login` and get an `HttpOnly`, `SameSite=Lax`,
`Secure`-in-production session cookie. Every page that shows document content
is gated.

**Images** are the interesting one. `/uploads` and `/outputs` used to be public
static mounts — every Aadhaar and PAN scan ever processed was downloadable by
anyone who could reach the host, no credential, nothing logged. They are not
mounted any more. The viewer now embeds a signed link per image:

```
/media/uploads/<file>?t=<hmac>
```

The signature covers **that one filename** and expires after
`MEDIA_TOKEN_TTL_SECONDS` (default 15 min), so a link copied out of a page
stops working shortly after rather than exposing an ID document indefinitely.
An API key or a viewer session also works. Anything else gets **404, not 403**,
so the endpoint cannot be used to probe whether a document exists.

---

## 3. Retention — read this before enabling

`ENABLE_RETENTION_SWEEP` is **off by default, deliberately.**

When on, it permanently deletes uploads, page renders and database rows older
than `RETENTION_DAYS` (default 30). There is no undo. It was briefly enabled
during this work and removed 192 uploads, 387 renders and 30 records on its
first pass — correct behaviour, but not something a default should decide.

Before enabling: set `RETENTION_DAYS` to how long you actually need documents
kept, and take a backup. For identity documents, indefinite retention is its
own risk — a policy is the right end state, just a chosen one.

`outputs/` grows fastest (page renders). `SAVE_DEBUG_IMAGES=false` keeps
per-cell debug crops off disk.

---

## 4. Restarts and in-flight work

Async jobs run in an in-process `ThreadPoolExecutor(max_workers=4)`. They do
**not** survive a restart, and nothing used to reconcile that — the row stayed
`PROCESSING` forever and a caller polling `/status` waited on a document nobody
was working on.

Two fixes:

* Autoreload is now off unless `DEBUG` is true *and* `APP_ENV` is not
  production. It was previously on in the deployed service, so any touched file
  restarted the process and silently killed every running job. (One orphaned
  reloader was still holding port 8080 during this work — worth checking for
  after a crash: `Get-NetTCPConnection -LocalPort 8080 -State Listen`.)
* On startup, any job left `PROCESSING` is marked `FAILED` with
  `"Processing was interrupted by a service restart. Re-submit the document."`
  A caller gets a terminal answer it can act on.

**Deploy during a quiet period regardless** — a restart still loses whatever is
mid-flight. Durable queuing is the real fix and is not done.

---

## 5. Operations

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness. Public, cheap, reveals nothing. |
| `GET /readiness` | Returns 503 until the database answers and the model is loaded. Point the load balancer here. |

Rate limits: `RATE_LIMIT_PER_MINUTE` (60) and `UPLOAD_RATE_LIMIT_PER_MINUTE`
(10), keyed by API key where present and by client address otherwise. Counters
are **per process and in memory**, so with multiple workers each holds its own
— a guard rail, not a quota.

500 responses now carry an `incident_id` instead of the exception text, which
was leaking absolute file paths. The same id is in the log line.

---

## 6. Database — MySQL

Runs on **MySQL 8** (`ocr_platform`, utf8mb4) as a dedicated `ocr_app` user
scoped to that schema only — not root, and with no access to the other
databases on the server.

Long-text columns are `LONGTEXT` on MySQL. Plain `TEXT` caps at 65,535 bytes
and truncates **silently**; the largest document here is 43,691 bytes of
markdown, so the ceiling was closer than it looks.

Migration from the old SQLite file is re-runnable and skips ids already
present:

```
python tools/migrate_sqlite_to_mysql.py --dry-run
python tools/migrate_sqlite_to_mysql.py
```

`ocr_database.db` is left untouched. Keep it until you are satisfied.

Backups (`tools/backup_db.py`) stream a logical dump through the driver, so no
`mysqldump` on PATH is needed. The dump has been verified by restoring it into
a scratch schema — 999 records, longest markdown intact.

## 7. Confidence — now measured, not assumed

`overall_confidence` used to be the constant 0.95 for any document without
table validation. It is now derived from checking each field against the page
(`app/services/field_verification.py`):

| Check | Catches |
| --- | --- |
| **Grounding** — does the value appear in the OCR text? | fabrication |
| **Format** — PAN / Aadhaar / email / PIN / date shapes | malformed values |
| **Placeholder** — `[Description]`, `N/A`, `Your Company` | template filler |
| **Legibility** — image dimensions, recognised word count | unreadable sources |

Each field returns `exact` / `verified` / `unchecked` / `flagged`, published as
`structured_data.field_verification`, so a consumer can route on *which* field
is suspect. Nothing is ever auto-corrected — a failed check identifies a
suspect value, not the right one.

Measured before and after on the same documents:

| Document | Before | After |
| --- | --- | --- |
| PAN card (correct) | 0.95 | **1.0**, no review |
| Aadhaar (correct) | 0.30, review | **1.0**, no review |
| Resume (correct) | 0.30, review | **1.0**, no review |
| Pamphlet, fabricated | **0.95, no review** | **0.10, review, 6 flagged** |
| Pamphlet, thumbnail | 0.50 | **0.10, review** |

**The limit, stated plainly:** grounding asks whether a value is *on the page*,
not whether it is in the *right field*. The resume's education block put the
school's 71% into the degree's GPA field (real value: CGPA 7.29) and that
still passes, because 71% genuinely appears. Catching misplacement needs
positional evidence. There is a test pinning this as known.

## 8. Two response formats

`response_format` decides who a response is written for. `full` is the
default and is unchanged, so existing callers see no difference.

| | `full` | `erp` |
|---|---|---|
| For | a person reviewing the document | an ERP writing a record |
| Carries | markdown, `structured_data`, `field_verification`, page list | named values, `line_items`, `confidence`, `review_required` |
| Tables | `fields.columns` + `fields.rows`, parallel lists | `line_items`, objects keyed by column name |
| Empty fields | present as `""` / `{}` | omitted |

Where to ask for it:

```
POST /api/v1/ocr/process            form field  response_format=erp   (sync only)
GET  /api/v1/ocr/status/{req_id}    ?response_format=erp
GET  /api/v1/ocr/document/{id}      ?response_format=erp
GET  /api/v1/ocr/erp/{id}           the same payload, no parameter to get wrong
GET  /api/v1/ocr/export/{id}        ?format=erp   identical bytes, as a download
```

An unrecognised value returns **400**, deliberately. Silently serving the full
envelope for a misspelled `erp` would look like the slim format does not work
and send the integrator looking in the wrong place.

The projection lives in `app/services/erp_payload.py` and is the *only* place
that decides what an ERP sees - the viewer's ERP Data tab renders the same
function, so what a reviewer signs off on cannot drift from what is sent.

Two properties worth knowing when mapping against it:

- **Duplicate column names are suffixed** (`Qty`, `Qty_2`, ...) and blank
  header cells become `column_N`. A PDF statement here genuinely produced six
  columns named `Qty`; keyed naively, five values would vanish.
- **Values are verbatim** - no number parsing, no date reformatting.
  Auto-correction is already recorded in this project as having corrupted
  cells the OCR read correctly, and a projection is not the place to
  reintroduce it.

## 9. The viewer's API key field

The upload page posts to `/api/v1/ocr/process` like any other caller, so it
needs a key like any other caller. **Before this it sent none**, which means
the browser upload returned 401 wherever `API_KEYS` was configured — the UI
was only usable on a server running without keys. There is now a field for it
on `/upload`; the key is attached as `X-API-Key` to the upload and to every
status poll that follows.

"Remember on this browser" writes it to `localStorage`, readable by any script
served from this origin, so it is **off unless asked for**. Unremembered, the
key lives only in the field for that page view.

## 10. Roles — viewer and administrator

Two sign-in roles, because two different things are being protected.

| | Credential | Can see |
|---|---|---|
| **viewer** | `UI_PASSWORD` (username left blank) | documents, upload, history |
| **admin** | `ADMIN_USERNAME` + `ADMIN_PASSWORD` | the above, **plus analytics** |

Analytics aggregates across every department — document volumes, token
spend, per-project activity. That is a different kind of access from reading
one document you were sent, so it gets its own credential.

Both live in `.env`, never in source: a password committed to a repo is
published to everyone who can read the repo, and cannot be rotated without a
code change.

```
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<set this>
```

Three details worth knowing:

- **The role is inside the signed session payload**, not a separate cookie,
  so editing `viewer` to `admin` invalidates the signature rather than
  granting the rights.
- **A viewer gets 404 on analytics, not 403.** Telling them the page exists
  and is forbidden confirms it is worth attacking. The nav link is hidden
  too — an offered link that 404s reads as a broken app.
- **An API key still reaches the analytics API.** The integrations that poll
  it are services holding a per-department key with no session to carry a
  role. What the gate stops is an ordinary viewer reading the whole
  organisation's activity.
- Sessions issued before roles existed stay valid, as viewers. An upgrade
  does not sign everybody out.

If `ADMIN_PASSWORD` is left empty there is no admin role to withhold, so
analytics stays open to any signed-in user — the pre-existing behaviour,
rather than a page nobody can reach.

## 11. Department API keys

Keys can be issued at runtime from **Analytics -> Department API Keys**
(administrator only). Adding a department is no longer a `.env` edit and a
restart; a new key works on the next request.

**Only a hash is stored.** An API key is a bearer credential - whoever holds
it can read every extracted document - so keeping the plaintext would mean a
database dump, a backup file or a careless SELECT hands over live access to
the whole corpus. Verification only ever compares, never reads back, so the
key is shown **once** at creation and then exists nowhere on this server. If
it is lost, revoke it and issue another.

The hash is HMAC-SHA256 under `SECRET_KEY`, which is what lets a presented
key be found by index instead of re-hashing every row on every request. The
trade: **rotating `SECRET_KEY` invalidates every issued key**, exactly as it
already invalidates sessions and media links.

Other things worth knowing:

- **Keys in `API_KEYS` keep working.** They are the bootstrap credential -
  something has to call the API before anyone has logged in to issue a key.
- **Revoked, not deleted.** "Who had access, and until when" is the question
  an audit asks, and a deleted row cannot answer it. Revoked keys stay in
  the list, greyed out.
- **Active hashes are cached in process for 30s**, invalidated immediately on
  issue or revoke. With several worker processes, a revoke takes effect
  everywhere within that window rather than instantly.
- `last_used_at` is written at most once a minute per key, so the hot path
  does not take a database write to store a timestamp nobody reads at that
  resolution.

Endpoints, all admin-gated:

```
GET    /api/v1/analytics/api-keys          list, prefixes only
POST   /api/v1/analytics/api-keys          {"department": "Radiology"}
DELETE /api/v1/analytics/api-keys/{id}     revoke
```

## 12. Token accounting

`prompt_tokens` / `completion_tokens` / `total_tokens` are now **measured**.

They used to be the constant **512 / 256 / 768** on every document — the
placeholder from the envelope template — while Ollama's real counts
(`prompt_eval_count`, `eval_count`) were written to the log and discarded.
The analytics token ledger was therefore that constant multiplied by the
document count: it moved only when the document count moved.

Measured afterwards on real documents: a PAN card costs **5,297** tokens and
an Aadhaar enrollment letter **6,107**, across two model calls each
(transcription, then structured extraction). The ledger had been
understating usage by roughly 85%.

The counter is **thread-local** (`app/services/model_service.py`), because
async jobs run in a `ThreadPoolExecutor` with four workers and a shared
counter would bill one document for another's tokens. A document that never
invokes the model — a digital PDF, a spreadsheet — records 0 rather than a
placeholder.

**Rows written before this change still carry 768.** They are left alone
rather than rewritten, because inventing history is exactly the failure this
fix was correcting. Historical averages will read low until those rows age
out; `UPDATE document_records SET prompt_tokens=0, completion_tokens=0,
total_tokens=0 WHERE total_tokens=768;` will exclude them from the ledger if
you would rather see only measured usage. Take a backup first.

## 13. Before you ship a change — the accuracy gate

Most faults found in this system were found by a user, in production, one at
a time. Three commands now stand between a change and that. Run them in this
order; the first is fast and the other two need the GPU and the model.

```bash
python -m pytest tests/ -q                    # ~2 min, no GPU
python tools/doctype_smoke_test.py            # ~1 min, every production type
python tools/accuracy_harness.py --runs 3     # ~6 min, the real gate
```

**`accuracy_harness.py` is the one that matters.** It scores every field of
every production document against ground truth a human read off the page
(`tests/ground_truth/production/*.json`), and it does it `--runs` times.

The repeats are not padding. A stochastic extractor can be right on Tuesday
and wrong on Wednesday, and a field that is correct 70% of the time looks
perfect when you check it once — while being exactly what a user experiences
as "it keeps getting things wrong". The harness reports both numbers:

```
  ok  3/3  stable      the same correct value every run
  ok  2/3  UNSTABLE    right twice, wrong once — the worst kind of fault
  BAD 0/3  stable      consistently wrong, which is at least easy to fix
```

It has already earned this: it caught `dob` on the Aadhaar enrollment letter
coming back as the enrollment date printed down the left edge, on about half
of all runs. A wrong date of birth on an identity record, invisible to
single-run testing.

Exit code is non-zero if any field is wrong, any field is unstable, or any
check fails. **A green run is the release gate. An UNSTABLE field is a
failure, not a flake to re-run.**

Adding a document to the gate: put it in `test_data_prod/corpus/`, write a
fixture beside the others, and read the expected values off the image
yourself. Do not paste in what the pipeline currently produces — that
fixture would only ever confirm today's behaviour.

## 14. Still open

- **SQLite → MySQL is done, but there is no migration tooling.** Schema changes
  go through the ad-hoc `ALTER TABLE` list in `database.py`. Add Alembic before
  the next structural change.
- **No durable job queue.** Startup now *re-queues* interrupted work (up to
  `MAX_PROCESSING_ATTEMPTS`, 3) rather than failing it, and a document whose
  file is gone or that has burned its attempts is marked FAILED. That covers
  crash recovery; it is still not a queue, so work submitted during a restart
  window can be lost.
- **TLS.** `tools/install_windows_service.ps1` registers the app and a nightly
  backup as scheduled tasks with restart-on-failure, but it serves plain HTTP.
  The session cookie is `Secure` and will not travel over http — put IIS,
  Caddy or nginx in front before exposing the host.
- **Rate-limit counters are per process.** With multiple uvicorn workers the
  effective limit multiplies.

### Accuracy coverage, stated honestly

The gate in §10 covers PAN (upright and upside down), the Aadhaar enrollment
letter, and a two-page image-only resume — every field, against ground truth
read off the page by a human, stable across repeated runs.

It does **not** yet cover:

- **Visiting cards.** No sample exists in the repo. Untested, not "working".
- **Pamphlets.** The only samples are 180×261 thumbnails whose body text is
  illegible to a human too. The system handles them correctly — UNKNOWN,
  confidence 0.10, every field flagged — but that verifies the *refusal*
  path, not extraction. A real pamphlet is needed.
- **Mirrored scans.** Orientation correction handles 180° rotation. A
  mirror-flipped page is not detected; the viewer's manual "Mirror Unflip"
  button remains the remedy.
- **Aadhaar card format.** Only the enrollment letter has been measured.
- **Handwriting**, beyond the signature line on the PAN card.

Each of these needs one real sample and a fixture before anyone says it
works.

## 15. Quick verification after deploying

```bash
curl -s http://HOST:8080/health                                  # 200
curl -s -o /dev/null -w '%{http_code}\n' http://HOST:8080/api/v1/analytics/metrics   # 401
curl -s -o /dev/null -w '%{http_code}\n' http://HOST:8080/uploads/anything.jpg       # 404
curl -s -o /dev/null -w '%{http_code}\n' -H "X-API-Key: $KEY" \
     http://HOST:8080/api/v1/analytics/metrics                                       # 200

# The ERP format, against any completed document id
curl -s -H "X-API-Key: $KEY" http://HOST:8080/api/v1/ocr/erp/$DOC_ID
curl -s -o /dev/null -w '%{http_code}\n' -H "X-API-Key: $KEY" \
     "http://HOST:8080/api/v1/ocr/document/$DOC_ID?response_format=nonsense"         # 400
```

If the second or third returns anything else, stop and check `.env`.
