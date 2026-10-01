// Upload Form Drag & Drop and Processing Handler
document.addEventListener('DOMContentLoaded', () => {
    // Check URL parameters for project_name pre-fill
    const urlParams = new URLSearchParams(window.location.search);
    const projFromUrl = urlParams.get('project_name');
    if (projFromUrl) {
        const projInput = document.getElementById('projectNameInput');
        if (projInput) {
            projInput.value = projFromUrl;
            projInput.classList.add('border-success', 'fw-semibold');
            
            // Add visual project context badge
            const label = projInput.closest('.card-body')?.querySelector('label');
            if (label) {
                const badge = document.createElement('span');
                badge.className = 'badge text-bg-success ms-2';
                badge.innerHTML = `<i class="bi bi-check-circle me-1"></i>Pre-selected: ${projFromUrl}`;
                label.appendChild(badge);
            }
        }
    }

    initApiKeyField();
    initUploadDropzone('mainDropZone', 'mainFileInput', 'mainUploadForm', 'startOcrBtn');
    initUploadDropzone('quickDropZone', 'quickFileInput', 'quickUploadForm', 'quickSubmitBtn');
});

// --------------------------------------------------------------------------- //
// API key
//
// This form posts to /api/v1/ocr/process, which requires a key like any other
// caller - so the key entered here is attached as the X-API-Key header to the
// upload and to every status poll that follows it.
//
// Remembering it means writing a credential into localStorage, where any
// script on this origin can read it, so that is opt-in rather than the
// default. Unremembered, it lives only in the field for this page view.
// --------------------------------------------------------------------------- //

const API_KEY_STORAGE = 'ocr.apiKey';

function getApiKey() {
    const input = document.getElementById('apiKeyInput');
    return input ? input.value.trim() : '';
}

// Only set the header when there is a key. An empty X-API-Key is a supplied
// credential that fails validation, which reads as "wrong key" rather than
// "no key" - and on a server with no API_KEYS configured the request would be
// rejected where it should have been allowed through.
function apiHeaders(extra) {
    const headers = Object.assign({}, extra || {});
    const key = getApiKey();
    if (key) headers['X-API-Key'] = key;
    return headers;
}

function initApiKeyField() {
    const input = document.getElementById('apiKeyInput');
    if (!input) return;

    const remember = document.getElementById('apiKeyRemember');
    const badge = document.getElementById('apiKeyStateBadge');
    const toggle = document.getElementById('apiKeyToggle');
    const clear = document.getElementById('apiKeyClear');

    function refreshBadge() {
        if (!badge) return;
        const set = !!input.value.trim();
        badge.textContent = set ? 'Key set' : 'Not set';
        badge.className = set ? 'badge text-bg-success' : 'badge text-bg-secondary';
    }

    let stored = null;
    try {
        stored = localStorage.getItem(API_KEY_STORAGE);
    } catch (e) {
        // Private browsing or blocked site data. The field still works.
    }
    if (stored) {
        input.value = stored;
        if (remember) remember.checked = true;
    }
    refreshBadge();

    input.addEventListener('input', () => {
        refreshBadge();
        if (remember && remember.checked) persist();
    });

    function persist() {
        try {
            const key = input.value.trim();
            if (remember && remember.checked && key) {
                localStorage.setItem(API_KEY_STORAGE, key);
            } else {
                localStorage.removeItem(API_KEY_STORAGE);
            }
        } catch (e) {
            // Nothing to do - the key is still usable for this page view.
        }
    }

    if (remember) remember.addEventListener('change', persist);

    if (toggle) {
        toggle.addEventListener('click', () => {
            input.type = input.type === 'password' ? 'text' : 'password';
        });
    }

    if (clear) {
        clear.addEventListener('click', () => {
            input.value = '';
            if (remember) remember.checked = false;
            try {
                localStorage.removeItem(API_KEY_STORAGE);
            } catch (e) { /* see above */ }
            refreshBadge();
            input.focus();
        });
    }
}

// Custom Linear Styled Dropdown Selection Handler
document.addEventListener('click', (e) => {
    const item = e.target.closest('.custom-dropdown-item');
    if (item) {
        e.preventDefault();

        const value = item.getAttribute('data-value');
        const hiddenInput = document.getElementById('docCategoryInput');
        const labelContainer = document.getElementById('selectedFormatLabel');
        const toggleBtn = document.getElementById('customFormatDropdown');

        if (hiddenInput) {
            hiddenInput.value = value;
        }

        if (labelContainer) {
            labelContainer.innerHTML = item.innerHTML;
        }

        document.querySelectorAll('.custom-dropdown-item').forEach(el => el.classList.remove('active'));
        item.classList.add('active');

        if (toggleBtn) {
            if (window.bootstrap && bootstrap.Dropdown) {
                const bsDropdown = bootstrap.Dropdown.getInstance(toggleBtn) || new bootstrap.Dropdown(toggleBtn);
                bsDropdown.hide();
            } else {
                toggleBtn.click();
            }
        }
    }
});

function initUploadDropzone(dropZoneId, fileInputId, formId, submitBtnId) {
    const dropZone = document.getElementById(dropZoneId);
    const fileInput = document.getElementById(fileInputId);
    const form = document.getElementById(formId);
    const submitBtn = document.getElementById(submitBtnId);
    const previewCard = document.getElementById('filePreviewCard');
    const previewFileName = document.getElementById('previewFileName');
    const previewFileSize = document.getElementById('previewFileSize');
    const removeFileBtn = document.getElementById('removeFileBtn');

    if (!dropZone || !fileInput || !form) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (submitBtn) submitBtn.disabled = false;
        
        // Quick upload info handle
        const quickFileName = document.getElementById('quickFileName');
        const quickFileInfo = document.getElementById('quickFileInfo');
        if (quickFileName && quickFileInfo) {
            quickFileName.textContent = file.name;
            quickFileInfo.classList.remove('d-none');
            quickFileInfo.classList.add('d-flex');
        }

        if (previewCard && previewFileName && previewFileSize) {
            previewFileName.textContent = file.name;
            previewFileSize.textContent = formatBytes(file.size);
            previewCard.classList.remove('d-none');
        }
    }

    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', () => {
            fileInput.value = '';
            if (submitBtn) submitBtn.disabled = true;
            if (previewCard) previewCard.classList.add('d-none');
        });
    }

    const quickClearFile = document.getElementById('quickClearFile');
    if (quickClearFile) {
        quickClearFile.addEventListener('click', () => {
            fileInput.value = '';
            if (submitBtn) submitBtn.disabled = true;
            const quickFileInfo = document.getElementById('quickFileInfo');
            if (quickFileInfo) quickFileInfo.classList.add('d-none');
        });
    }

    // Submit Processing with Live Async Per-Page Progress Polling
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!fileInput.files.length) return;

        const projInput = document.getElementById('projectNameInput');
        if (projInput && !projInput.value.trim()) {
            alert("Please enter a Project Name. Project Name is mandatory.");
            projInput.focus();
            return;
        }

        const progressWrapper = document.getElementById('progressWrapper');
        const progressBar = document.getElementById('progressBar');
        const progressPercent = document.getElementById('progressPercent');
        const progressStatusText = document.getElementById('progressStatusText');

        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status"></span>Submitting Document...`;
        }

        if (progressWrapper) progressWrapper.classList.remove('d-none');

        const formData = new FormData(form);
        formData.append('async_mode', 'true');

        try {
            if (progressStatusText) progressStatusText.textContent = "Uploading document...";
            if (progressBar) progressBar.style.width = "5%";
            if (progressPercent) progressPercent.textContent = "5%";

            const response = await fetch('/api/v1/ocr/process', {
                method: 'POST',
                headers: apiHeaders(),
                body: formData
            });

            const data = await response.json();
            if (!response.ok || !data.success) {
                if (response.status === 401) {
                    throw new Error('The API key was missing or rejected. Enter the key issued to your department above.');
                }
                throw new Error(data.detail || data.error || 'Failed to submit document');
            }

            const requestId = data.request_id;
            if (progressStatusText) progressStatusText.textContent = "Document submitted. Starting vision pipeline...";

            // Poll real per-page job status every 1.5 seconds
            const pollInterval = setInterval(async () => {
                try {
                    const statusRes = await fetch(`/api/v1/ocr/status/${requestId}`, {
                        headers: apiHeaders()
                    });

                    // Without this the poll would retry a rejected credential
                    // every 3 seconds forever, with the progress bar frozen and
                    // nothing on screen saying why.
                    if (statusRes.status === 401 || statusRes.status === 403) {
                        clearInterval(pollInterval);
                        if (progressStatusText) {
                            progressStatusText.textContent = 'Submitted, but the API key was rejected while checking progress.';
                        }
                        showToast('The document was submitted, but the API key was rejected while polling its status.', 'danger');
                        return;
                    }

                    const statusData = await statusRes.json();

                    if (statusData.status === 'PROCESSING') {
                        const pct = statusData.percent || 10;
                        const msg = statusData.message || `Processing Page ${statusData.current_page || 1} of ${statusData.total_pages || 1}...`;
                        if (progressBar) progressBar.style.width = `${pct}%`;
                        if (progressPercent) progressPercent.textContent = `${pct}%`;
                        if (progressStatusText) progressStatusText.textContent = msg;
                    } else if (statusData.status === 'COMPLETED') {
                        clearInterval(pollInterval);
                        if (progressBar) progressBar.style.width = `100%`;
                        if (progressPercent) progressPercent.textContent = `100%`;
                        if (progressStatusText) progressStatusText.textContent = "Extraction Complete! Redirecting...";

                        showToast('<i class="bi bi-check-circle me-1"></i> Document processed successfully!');
                        setTimeout(() => {
                            window.location.href = `/result/${statusData.document.id}`;
                        }, 500);
                    } else if (statusData.status === 'FAILED') {
                        clearInterval(pollInterval);
                        throw new Error(statusData.error_message || 'OCR processing failed.');
                    }
                } catch (pollErr) {
                    console.error(pollErr);
                }
            }, 3000);

        } catch (error) {
            showToast(`Error: ${error.message}`, 'danger');
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = `<i class="bi bi-play-circle-fill fs-5"></i> Start AI Extraction`;
            }
        }
    });
}

function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}
