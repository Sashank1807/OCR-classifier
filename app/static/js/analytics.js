// Analytics dashboard
//
// These endpoints need a credential. An admin session cookie satisfies them
// on its own, but the key saved on the Upload page is sent as well so the
// page also works for a service account or a browser without the cookie -
// the fetches here previously sent neither and returned 401 silently, which
// rendered an empty dashboard with no error anywhere.
function analyticsHeaders() {
    try {
        const key = (localStorage.getItem('ocr.apiKey') || '').trim();
        if (key) return { 'X-API-Key': key };
    } catch (e) { /* blocked site data: the session cookie still applies */ }
    return {};
}

async function analyticsFetch(url) {
    const response = await fetch(url, { headers: analyticsHeaders() });
    if (response.status === 401 || response.status === 403 || response.status === 404) {
        throw new Error('Analytics is restricted to administrators. Sign in as admin.');
    }
    if (!response.ok) throw new Error(`Request failed (${response.status})`);
    return response;
}

// Enterprise Analytics & Project Monitoring Client Script
document.addEventListener('DOMContentLoaded', () => {
    fetchAnalyticsMetrics();

    const refreshBtn = document.getElementById('refreshAnalyticsBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            const projectInput = document.getElementById('projectSearchInput');
            const query = projectInput ? projectInput.value.trim() : '';
            if (query) {
                searchProjectMetrics(query);
            } else {
                fetchAnalyticsMetrics();
            }
        });
    }

    const projectSearchBtn = document.getElementById('projectSearchBtn');
    const projectSearchInput = document.getElementById('projectSearchInput');

    if (projectSearchBtn && projectSearchInput) {
        projectSearchBtn.addEventListener('click', () => {
            const query = projectSearchInput.value.trim();
            if (query) {
                searchProjectMetrics(query);
            } else {
                fetchAnalyticsMetrics();
            }
        });

        projectSearchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const query = projectSearchInput.value.trim();
                if (query) {
                    searchProjectMetrics(query);
                } else {
                    fetchAnalyticsMetrics();
                }
            }
        });
    }
});

async function fetchAnalyticsMetrics() {
    try {
        const response = await analyticsFetch('/api/v1/analytics/metrics');
        const data = await response.json();

        if (response.ok && data.success) {
            renderMetrics(data);
        } else {
            throw new Error(data.detail || 'Failed to load analytics metrics');
        }
    } catch (err) {
        console.error('Analytics Fetch Error:', err);
    }
}

async function searchProjectMetrics(projectName) {
    try {
        const response = await analyticsFetch(`/api/v1/analytics/metrics/${encodeURIComponent(projectName)}`);
        const data = await response.json();

        if (response.ok && data.success) {
            renderMetrics(data);
        } else {
            throw new Error(data.detail || 'Failed to load project metrics');
        }
    } catch (err) {
        console.error('Project Search Error:', err);
    }
}

function renderMetrics(data) {
    const summary = data.summary || {};
    const tokens = data.token_monitoring || {};
    const projAnalytics = data.project_analytics || data.department_analytics || [];
    const docDist = data.document_type_distribution || {};

    // Cards
    document.getElementById('metricTotalRequests').textContent = summary.total_requests || 0;
    document.getElementById('metricTotalTokens').textContent = (tokens.total_tokens_consumed || 0).toLocaleString();
    document.getElementById('metricAvgTime').textContent = `${summary.average_processing_time || 0}s`;
    document.getElementById('metricAvgConfidence').textContent = `${Math.round((summary.average_confidence || 1.0) * 100)}%`;

    // Tokens
    document.getElementById('promptTokensCount').textContent = (tokens.total_prompt_tokens || 0).toLocaleString();
    document.getElementById('completionTokensCount').textContent = (tokens.total_completion_tokens || 0).toLocaleString();
    document.getElementById('totalTokensCount').textContent = (tokens.total_tokens_consumed || 0).toLocaleString();

    // Project Table
    const tableBody = document.getElementById('departmentTableBody');
    if (tableBody) {
        if (projAnalytics.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="5" class="text-center py-4 text-body-secondary">
                        No project metrics found${data.project_name ? ` for '${escapeHtml(data.project_name)}'` : ''}. Submit OCR requests with project_name parameter.
                    </td>
                </tr>
            `;
        } else {
            tableBody.innerHTML = projAnalytics.map(proj => {
                const name = proj.project_name || proj.department || 'DefaultProject';
                return `
                    <tr style="cursor: pointer;" onclick="openProjectModal('${escapeHtml(name)}')">
                        <td class="ps-4 fw-semibold text-primary">
                            <i class="bi bi-folder-symlink me-2"></i>${escapeHtml(name)}
                        </td>
                        <td><span class="badge text-bg-primary-subtle text-primary border border-primary-subtle fs-6">${proj.total_requests}</span></td>
                        <td class="font-monospace text-body-secondary">${(proj.total_tokens || 0).toLocaleString()}</td>
                        <td>
                            <span class="badge text-bg-success-subtle text-success border border-success-subtle">
                                ${Math.round((proj.average_confidence || 1.0) * 100)}%
                            </span>
                        </td>
                        <td class="pe-4 text-end">
                            <button class="btn btn-sm btn-outline-primary rounded-pill px-3 shadow-sm" onclick="event.stopPropagation(); openProjectModal('${escapeHtml(name)}')">
                                <i class="bi bi-bar-chart-line me-1"></i> Insights
                            </button>
                        </td>
                    </tr>
                `;
            }).join('');
        }
    }

    // Document Distribution Container
    const distContainer = document.getElementById('documentTypeDistributionContainer');
    if (distContainer) {
        const types = Object.keys(docDist);
        if (types.length === 0) {
            distContainer.innerHTML = `<div class="text-center py-4 text-body-secondary">No document classification data.</div>`;
        } else {
            const total = summary.total_requests || 1;
            distContainer.innerHTML = types.map(t => {
                const count = docDist[t];
                const pct = Math.round((count / total) * 100);
                return `
                    <div class="mb-3">
                        <div class="d-flex justify-content-between small fw-medium mb-1">
                            <span class="badge text-bg-secondary-subtle text-body border">${escapeHtml(t)}</span>
                            <span class="text-body-secondary">${count} doc(s) (${pct}%)</span>
                        </div>
                        <div class="progress" style="height: 6px;">
                            <div class="progress-bar bg-primary" role="progressbar" style="width: ${pct}%"></div>
                        </div>
                    </div>
                `;
            }).join('');
        }
    }
}

async function openProjectModal(projectName) {
    const modalEl = document.getElementById('projectDetailModal');
    if (!modalEl) return;

    const bsModal = new bootstrap.Modal(modalEl);
    document.getElementById('projectModalTitle').textContent = `Project: ${projectName}`;
    document.getElementById('projectModalSubtitle').textContent = `Tokens usage breakdown and document formats used for '${projectName}'`;

    const modalProjInput = document.getElementById('modalProjectNameInput');
    if (modalProjInput) {
        modalProjInput.value = projectName;
    }

    const externalUploadBtn = document.getElementById('uploadToProjectExternalBtn');
    if (externalUploadBtn) {
        externalUploadBtn.href = `/upload?project_name=${encodeURIComponent(projectName)}`;
    }

    const switchToUploadBtn = document.getElementById('modalSwitchToUploadBtn');
    if (switchToUploadBtn) {
        switchToUploadBtn.onclick = () => {
            const uploadTabBtn = document.getElementById('tab-upload');
            if (uploadTabBtn) {
                const tab = new bootstrap.Tab(uploadTabBtn);
                tab.show();
            }
        };
    }

    // Default to first tab
    const insightsTabBtn = document.getElementById('tab-insights');
    if (insightsTabBtn) {
        const tab = new bootstrap.Tab(insightsTabBtn);
        tab.show();
    }

    initProjectModalUpload(projectName);

    const bodyEl = document.getElementById('projectModalBody');
    bodyEl.innerHTML = `
        <div class="text-center py-5 text-body-secondary">
            <span class="spinner-border spinner-border-sm me-2"></span>Loading breakdown for '${escapeHtml(projectName)}'...
        </div>
    `;

    bsModal.show();

    try {
        const response = await analyticsFetch(`/api/v1/analytics/metrics/${encodeURIComponent(projectName)}`);
        const data = await response.json();

        if (response.ok && data.success) {
            renderProjectModalContent(data, bodyEl);
        } else {
            bodyEl.innerHTML = `<div class="alert alert-danger mb-0">${escapeHtml(data.detail || 'Failed to load project metrics')}</div>`;
        }
    } catch (err) {
        console.error('Project Modal Fetch Error:', err);
        bodyEl.innerHTML = `<div class="alert alert-danger mb-0">Error connecting to server to load project metrics.</div>`;
    }
}

let modalUploadInitialized = false;
function initProjectModalUpload(currentProjectName) {
    const fileInput = document.getElementById('modalFileInput');
    const dropZone = document.getElementById('modalDropZone');
    const form = document.getElementById('projectModalUploadForm');
    const submitBtn = document.getElementById('modalSubmitBtn');
    const fileBanner = document.getElementById('modalFileBanner');
    const fileNameText = document.getElementById('modalFileNameText');
    const clearFileBtn = document.getElementById('modalClearFileBtn');
    const progressWrapper = document.getElementById('modalProgressWrapper');
    const progressBar = document.getElementById('modalProgressBar');
    const progressPercent = document.getElementById('modalProgressPercent');
    const progressText = document.getElementById('modalProgressText');

    if (!form || modalUploadInitialized) return;
    modalUploadInitialized = true;

    function handleFile(file) {
        if (!file) return;
        if (fileNameText) fileNameText.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        if (fileBanner) {
            fileBanner.classList.remove('d-none');
            fileBanner.classList.add('d-flex');
        }
        if (submitBtn) submitBtn.disabled = false;
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
    }

    if (dropZone) {
        ['dragenter', 'dragover'].forEach(name => {
            dropZone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('border-primary');
            });
        });
        ['dragleave', 'drop'].forEach(name => {
            dropZone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('border-primary');
            });
        });
        dropZone.addEventListener('drop', (e) => {
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (clearFileBtn) {
        clearFileBtn.addEventListener('click', () => {
            if (fileInput) fileInput.value = '';
            if (fileBanner) fileBanner.classList.add('d-none');
            if (submitBtn) submitBtn.disabled = true;
        });
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!fileInput.files || !fileInput.files.length) {
            alert("Please select a file to upload.");
            return;
        }

        const projInput = document.getElementById('modalProjectNameInput');
        const projNameVal = projInput ? projInput.value.trim() : currentProjectName;
        if (!projNameVal) {
            alert("Project Name is mandatory. Please provide a project name.");
            if (projInput) projInput.focus();
            return;
        }

        const formData = new FormData(form);
        formData.set('project_name', projNameVal);
        formData.append('async_mode', 'true');

        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Processing in ${escapeHtml(projNameVal)}...`;
        }

        if (progressWrapper) progressWrapper.classList.remove('d-none');
        if (progressBar) progressBar.style.width = '10%';
        if (progressPercent) progressPercent.textContent = '10%';
        if (progressText) progressText.textContent = `Uploading document to project '${projNameVal}'...`;

        try {
            const res = await fetch('/api/v1/ocr/process', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (!res.ok || !data.success) {
                throw new Error(data.detail || data.error || 'Failed to start processing');
            }

            const reqId = data.request_id;
            if (progressBar) progressBar.style.width = '30%';
            if (progressPercent) progressPercent.textContent = '30%';
            if (progressText) progressText.textContent = "Processing with Vision AI Pipeline...";

            const poller = setInterval(async () => {
                try {
                    const stRes = await fetch(`/api/v1/ocr/status/${reqId}`);
                    const stData = await stRes.json();
                    if (stData.status === 'PROCESSING') {
                        const pct = stData.percent || 40;
                        if (progressBar) progressBar.style.width = `${pct}%`;
                        if (progressPercent) progressPercent.textContent = `${pct}%`;
                        if (progressText) progressText.textContent = stData.message || 'Extracting tables & text...';
                    } else if (stData.status === 'COMPLETED') {
                        clearInterval(poller);
                        if (progressBar) progressBar.style.width = '100%';
                        if (progressPercent) progressPercent.textContent = '100%';
                        if (progressText) progressText.textContent = 'Extraction finished! Redirecting...';
                        setTimeout(() => {
                            window.location.href = `/result/${stData.document.id}`;
                        }, 500);
                    } else if (stData.status === 'FAILED') {
                        clearInterval(poller);
                        throw new Error(stData.error_message || 'Processing failed.');
                    }
                } catch (pollErr) {
                    console.error(pollErr);
                }
            }, 2500);

        } catch (err) {
            alert(`Upload Error: ${err.message}`);
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = `<i class="bi bi-play-circle-fill"></i> Upload & Process in Project`;
            }
            if (progressWrapper) progressWrapper.classList.add('d-none');
        }
    });
}

function renderProjectModalContent(data, container) {
    const summary = data.summary || {};
    const tokens = data.token_monitoring || {};
    const docDist = data.document_type_distribution || {};
    const recentDocs = data.recent_documents || [];
    const totalReqs = summary.total_requests || 0;

    const docTypes = Object.keys(docDist);
    let docFormatsHtml = '';
    if (docTypes.length === 0) {
        docFormatsHtml = `<span class="text-body-secondary">No document formats recorded yet.</span>`;
    } else {
        docFormatsHtml = docTypes.map(t => {
            const count = docDist[t];
            const pct = totalReqs > 0 ? Math.round((count / totalReqs) * 100) : 0;
            return `
                <div class="mb-2">
                    <div class="d-flex justify-content-between small fw-medium mb-1">
                        <span class="fw-semibold text-dark"><i class="bi bi-file-earmark-text me-1 text-primary"></i>${escapeHtml(t)}</span>
                        <span class="text-body-secondary">${count} document(s) (${pct}%)</span>
                    </div>
                    <div class="progress" style="height: 6px;">
                        <div class="progress-bar bg-success" role="progressbar" style="width: ${pct}%"></div>
                    </div>
                </div>
            `;
        }).join('');
    }

    let recentDocsHtml = '';
    if (recentDocs.length === 0) {
        recentDocsHtml = `<tr><td colspan="4" class="text-center text-body-secondary py-3">No processed documents found for this project.</td></tr>`;
    } else {
        recentDocsHtml = recentDocs.map(d => `
            <tr>
                <td class="fw-medium text-truncate" style="max-width: 220px;">
                    <a href="/result/${d.id}" target="_blank" class="text-decoration-none fw-semibold text-primary">
                        ${escapeHtml(d.filename)}
                    </a>
                </td>
                <td><span class="badge text-bg-primary-subtle text-primary border border-primary-subtle">${escapeHtml(d.document_type || 'UNKNOWN')}</span></td>
                <td>
                    <span class="badge text-bg-${d.status === 'COMPLETED' ? 'success' : d.status === 'FAILED' ? 'danger' : 'warning'}">
                        ${escapeHtml(d.status)}
                    </span>
                </td>
                <td class="text-end small text-body-secondary">${d.processing_time || 0}s</td>
            </tr>
        `).join('');
    }

    container.innerHTML = `
        <!-- Top Metrics Cards -->
        <div class="row g-3 mb-4">
            <div class="col-md-3">
                <div class="p-3 border rounded-3 bg-body-tertiary text-center">
                    <small class="text-body-secondary d-block mb-1">Total Requests</small>
                    <h4 class="fw-bold mb-0 text-primary">${totalReqs}</h4>
                </div>
            </div>
            <div class="col-md-3">
                <div class="p-3 border rounded-3 bg-body-tertiary text-center">
                    <small class="text-body-secondary d-block mb-1">Avg Accuracy</small>
                    <h4 class="fw-bold mb-0 text-success">${Math.round((summary.average_confidence || 1.0) * 100)}%</h4>
                </div>
            </div>
            <div class="col-md-3">
                <div class="p-3 border rounded-3 bg-body-tertiary text-center">
                    <small class="text-body-secondary d-block mb-1">Avg Processing Time</small>
                    <h4 class="fw-bold mb-0 text-info">${summary.average_processing_time || 0}s</h4>
                </div>
            </div>
            <div class="col-md-3">
                <div class="p-3 border rounded-3 bg-body-tertiary text-center">
                    <small class="text-body-secondary d-block mb-1">Avg Tokens / Request</small>
                    <h4 class="fw-bold mb-0 text-warning">${summary.average_tokens_per_request || 0}</h4>
                </div>
            </div>
        </div>

        <!-- Token Monitoring Breakdown -->
        <div class="card border-0 bg-light-subtle shadow-sm rounded-3 mb-4">
            <div class="card-body p-3">
                <h6 class="fw-bold mb-3"><i class="bi bi-cpu me-2 text-primary"></i>LLM Token Usage Breakdown</h6>
                <div class="row g-3 text-center">
                    <div class="col-md-4">
                        <div class="p-2 border rounded bg-body">
                            <small class="text-body-secondary d-block">Prompt Tokens (Input)</small>
                            <span class="fw-bold fs-5 text-primary">${(tokens.total_prompt_tokens || 0).toLocaleString()}</span>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="p-2 border rounded bg-body">
                            <small class="text-body-secondary d-block">Completion Tokens (Output)</small>
                            <span class="fw-bold fs-5 text-success">${(tokens.total_completion_tokens || 0).toLocaleString()}</span>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="p-2 border rounded bg-body">
                            <small class="text-body-secondary d-block">Total Tokens Consumed</small>
                            <span class="fw-bold fs-5 text-warning">${(tokens.total_tokens_consumed || 0).toLocaleString()}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Document Formats Used Before -->
        <div class="card border-0 bg-light-subtle shadow-sm rounded-3 mb-4">
            <div class="card-body p-3">
                <h6 class="fw-bold mb-3"><i class="bi bi-folder-check me-2 text-success"></i>Document Formats Used Before</h6>
                ${docFormatsHtml}
            </div>
        </div>

        <!-- Recent Project Extractions Table -->
        <div class="card border-0 shadow-sm rounded-3">
            <div class="card-body p-0">
                <div class="p-3 border-bottom d-flex justify-content-between align-items-center">
                    <h6 class="fw-bold mb-0"><i class="bi bi-journal-text me-2 text-primary"></i>Recent Document Extractions</h6>
                    <span class="badge text-bg-secondary">Latest 10</span>
                </div>
                <div class="table-responsive">
                    <table class="table table-hover align-middle mb-0">
                        <thead class="table-light">
                            <tr>
                                <th>Filename</th>
                                <th>Format</th>
                                <th>Status</th>
                                <th class="text-end">Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${recentDocsHtml}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
