// Result Page View interactions (Page switcher, Reprocess & Image Rotation)

// Reprocess posts to /api/v1/ocr/*, which requires a key like every other
// caller - the same gap the upload form had. The key is whatever the upload
// page stored; without it these buttons return 401 on any server with
// API_KEYS set, which is every production config.
function resultApiHeaders() {
    try {
        const key = (localStorage.getItem('ocr.apiKey') || '').trim();
        if (key) return { 'X-API-Key': key };
    } catch (e) {
        // Blocked site data: fall through to the session cookie.
    }
    return {};
}

function reportAuthFailure(response) {
    if (response.status === 401 || response.status === 403) {
        showToast('The API key was rejected. Set it on the Upload page, then retry.', 'danger');
        return true;
    }
    return false;
}
document.addEventListener('DOMContentLoaded', () => {
    const pageSelect = document.getElementById('pageSelect');
    const docPreviewImg = document.getElementById('docPreviewImg');
    const reprocessBtn = document.getElementById('reprocessBtn');

    if (pageSelect && docPreviewImg) {
        pageSelect.addEventListener('change', () => {
            const selectedOption = pageSelect.options[pageSelect.selectedIndex];
            const url = selectedOption.getAttribute('data-url');
            if (url) {
                docPreviewImg.src = url;
            }
        });
    }

    if (reprocessBtn) {
        const docId = reprocessBtn.getAttribute('data-id');

        reprocessBtn.addEventListener('click', async () => {
            if (!docId) return;

            if (!confirm('Are you sure you want to reprocess this document?')) return;

            reprocessBtn.disabled = true;
            reprocessBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-1" role="status"></span> Reprocessing...`;

            try {
                const response = await fetch(`/api/v1/ocr/reprocess/${docId}`, {
                    method: 'POST',
                    headers: resultApiHeaders()
                });
                if (reportAuthFailure(response)) {
                    reprocessBtn.disabled = false;
                    reprocessBtn.innerHTML = `<i class="bi bi-arrow-repeat me-1"></i> Reprocess`;
                    return;
                }
                const data = await response.json();
                if (response.ok && data.success) {
                    showToast('Document reprocessed successfully!');
                    setTimeout(() => {
                        window.location.href = `/result/${data.data.id}`;
                    }, 500);
                } else {
                    throw new Error(data.detail || 'Reprocessing failed.');
                }
            } catch (err) {
                showToast(`Error: ${err.message}`, 'danger');
                reprocessBtn.disabled = false;
                reprocessBtn.innerHTML = `<i class="bi bi-arrow-repeat me-1"></i> Reprocess`;
            }
        });

        // Image Correction & Rotation Handling
        const transformBtns = document.querySelectorAll('.transform-btn');
        if (transformBtns.length) {
            transformBtns.forEach(btn => {
                btn.addEventListener('click', async () => {
                    const rotate = btn.getAttribute('data-rotate') || 0;
                    const flip = btn.getAttribute('data-flip') === 'true';

                    btn.disabled = true;
                    const originalHtml = btn.innerHTML;
                    btn.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span> Correcting...`;

                    try {
                        const response = await fetch(`/api/v1/ocr/reprocess/${docId}?rotate_deg=${rotate}&flip_horizontal=${flip}`, {
                            method: 'POST',
                            headers: resultApiHeaders()
                        });
                        if (reportAuthFailure(response)) {
                            btn.disabled = false;
                            btn.innerHTML = originalHtml;
                            return;
                        }
                        const data = await response.json();
                        if (response.ok && data.success) {
                            showToast('Document orientation corrected & reprocessed!');
                            setTimeout(() => {
                                window.location.reload();
                            }, 500);
                        } else {
                            throw new Error(data.detail || 'Transformation failed');
                        }
                    } catch (err) {
                        showToast(`Error: ${err.message}`, 'danger');
                        btn.disabled = false;
                        btn.innerHTML = originalHtml;
                    }
                });
            });
        }
    }
});
