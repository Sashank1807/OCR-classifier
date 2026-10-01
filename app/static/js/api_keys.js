// ===========================================================================
// Department API keys (administrator only)
//
// A separate file from analytics.js because it is a separate concern. It
// PREFERS that file's `analyticsFetch` / `analyticsHeaders` helpers, but it
// does not require them: the fallbacks below mean this file works whatever
// order the scripts load in, or if one of them fails outright.
//
// That is not hypothetical caution. A stale cached analytics.js against a
// fresh api_keys.js produced exactly one symptom - "analyticsHeaders is not
// defined" on clicking Generate - and a cross-file dependency on a plain
// global is what made a caching problem look like a broken feature.
//
// The plaintext key appears exactly once, in the reveal modal, with the
// warning beside it rather than in documentation nobody opens.
// ===========================================================================

// Credential for the admin endpoints. The session cookie alone satisfies
// them; the key saved on the Upload page is sent as well so the page also
// works for a service account or a browser without the cookie.
function apiKeyHeaders() {
    if (typeof analyticsHeaders === 'function') return analyticsHeaders();
    try {
        const key = (localStorage.getItem('ocr.apiKey') || '').trim();
        if (key) return { 'X-API-Key': key };
    } catch (e) { /* blocked site data: the session cookie still applies */ }
    return {};
}

async function apiKeyFetch(url) {
    if (typeof analyticsFetch === 'function') return analyticsFetch(url);
    const response = await fetch(url, { headers: apiKeyHeaders() });
    if (response.status === 401 || response.status === 403 || response.status === 404) {
        throw new Error('Administrator access is required. Sign in as admin.');
    }
    if (!response.ok) throw new Error('Request failed (' + response.status + ')');
    return response;
}

// main.js owns the toast, but this file must not fail if it is absent.
function apiKeyToast(message, kind) {
    if (typeof showToast === 'function') return showToast(message, kind);
    if (kind === 'danger') alert(message.replace(/<[^>]*>/g, ''));
}

document.addEventListener('DOMContentLoaded', () => {
    if (!document.getElementById('apiKeysCard')) return;   // not the admin view
    loadApiKeys();
    wireApiKeyControls();
});

// Department names are free text typed by an administrator and are rendered
// back into the table, so they are escaped rather than trusted.
function escapeHtml(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (c) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
}

function shortDate(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return isNaN(d) ? '—'
        : d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

async function loadApiKeys() {
    const body = document.getElementById('apiKeysBody');
    if (!body) return;

    try {
        const res = await apiKeyFetch('/api/v1/analytics/api-keys');
        const keys = (await res.json()).api_keys || [];

        if (!keys.length) {
            body.innerHTML =
                '<tr><td colspan="6" class="text-center text-body-secondary py-4">' +
                'No keys issued yet. Generate one per department.</td></tr>';
            return;
        }

        body.innerHTML = keys.map(function (k) {
            const action = k.active
                ? '<button class="btn btn-sm btn-outline-danger rounded-pill px-3 revoke-key-btn" ' +
                  'data-id="' + k.id + '" data-dept="' + escapeHtml(k.department) + '">Revoke</button>'
                : '<span class="small text-body-tertiary">—</span>';

            return '<tr class="' + (k.active ? '' : 'opacity-50') + '">' +
                '<td class="ps-3 fw-semibold small">' + escapeHtml(k.department) + '</td>' +
                '<td><code class="small">' + escapeHtml(k.prefix) + '…</code></td>' +
                '<td class="small text-body-secondary">' + shortDate(k.created_at) + '</td>' +
                '<td class="small text-body-secondary">' + shortDate(k.last_used_at) + '</td>' +
                '<td><span class="badge ' + (k.active ? 'text-bg-success' : 'text-bg-secondary') + '">' +
                    (k.active ? 'Active' : 'Revoked') + '</span></td>' +
                '<td class="text-end pe-3">' + action + '</td>' +
            '</tr>';
        }).join('');

        body.querySelectorAll('.revoke-key-btn').forEach(function (btn) {
            btn.addEventListener('click', function () { revokeApiKey(btn); });
        });
    } catch (err) {
        body.innerHTML =
            '<tr><td colspan="6" class="text-center text-danger py-4 small">' +
            escapeHtml(err.message) + '</td></tr>';
    }
}

function wireApiKeyControls() {
    const generateBtn = document.getElementById('generateKeyBtn');
    const input = document.getElementById('keyDepartmentInput');
    const errorBox = document.getElementById('keyFormError');
    const generateModal = document.getElementById('generateKeyModal');

    function showError(message) {
        if (!errorBox) return;
        errorBox.textContent = message;
        errorBox.classList.remove('d-none');
    }

    if (generateModal) {
        generateModal.addEventListener('shown.bs.modal', function () {
            if (input) input.focus();
        });
    }

    if (input) {
        input.addEventListener('input', function () {
            if (errorBox) errorBox.classList.add('d-none');
        });
        input.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                if (generateBtn) generateBtn.click();
            }
        });
    }

    if (generateBtn) {
        generateBtn.addEventListener('click', async function () {
            const department = ((input && input.value) || '').trim();
            if (department.length < 2) {
                showError('Enter the department this key is for.');
                if (input) input.focus();
                return;
            }

            const original = generateBtn.innerHTML;
            generateBtn.disabled = true;
            generateBtn.innerHTML =
                '<span class="spinner-border spinner-border-sm me-2"></span>Generating…';

            try {
                const res = await fetch('/api/v1/analytics/api-keys', {
                    method: 'POST',
                    headers: Object.assign({ 'Content-Type': 'application/json' }, apiKeyHeaders()),
                    body: JSON.stringify({ department: department })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Could not generate a key.');

                const openModal = bootstrap.Modal.getInstance(generateModal);
                if (openModal) openModal.hide();
                if (input) input.value = '';

                document.getElementById('newKeyValue').value = data.api_key;
                document.getElementById('newKeyDepartment').textContent = 'For ' + data.key.department;
                new bootstrap.Modal(document.getElementById('newKeyModal')).show();

                loadApiKeys();
                apiKeyToast('<i class="bi bi-key me-1"></i> API key issued.');
            } catch (err) {
                showError(err.message);
            } finally {
                generateBtn.disabled = false;
                generateBtn.innerHTML = original;
            }
        });
    }

    const copyBtn = document.getElementById('copyNewKeyBtn');
    if (copyBtn) {
        copyBtn.addEventListener('click', function () {
            const field = document.getElementById('newKeyValue');
            field.select();
            navigator.clipboard.writeText(field.value).then(function () {
                copyBtn.innerHTML = '<i class="bi bi-check-lg me-1"></i> Copied';
                setTimeout(function () {
                    copyBtn.innerHTML = '<i class="bi bi-clipboard me-1"></i> Copy';
                }, 1800);
            }).catch(function () {
                // Clipboard access can be refused; the field is selected, so
                // the key is still one keystroke away rather than lost.
                apiKeyToast('Could not copy automatically — the key is selected, press Ctrl+C.', 'danger');
            });
        });
    }
}

async function revokeApiKey(btn) {
    const id = btn.getAttribute('data-id');
    const dept = btn.getAttribute('data-dept');
    const warning = 'Revoke the key for ' + dept + '?\n\n' +
        'Anything still using it will start getting 401 on its next request.';
    if (!confirm(warning)) return;

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';

    try {
        const res = await fetch('/api/v1/analytics/api-keys/' + id, {
            method: 'DELETE',
            headers: apiKeyHeaders()
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Could not revoke that key.');
        apiKeyToast('<i class="bi bi-slash-circle me-1"></i> Key revoked.');
        loadApiKeys();
    } catch (err) {
        apiKeyToast('Error: ' + err.message, 'danger');
        btn.disabled = false;
        btn.textContent = 'Revoke';
    }
}
