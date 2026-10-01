// Global Application Theme & Helper Functions
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initCopyButtons();
});

// Theme
//
// STYLE.md defines ONE palette - light. There is no prefers-color-scheme
// block, no .dark class and no dark token set, so a dark mode here would fall
// through to undefined variables and render, for example, a near-black input
// holding near-black text. The document is pinned to light and any stale
// 'dark' left in localStorage by the old toggle is cleared.
function initTheme() {
    document.documentElement.setAttribute('data-bs-theme', 'light');
    try { localStorage.removeItem('theme'); } catch (e) { /* private mode */ }
}

// Global Toast Notification Helper
function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const bgClass = type === 'success' ? 'text-bg-success' : (type === 'danger' ? 'text-bg-danger' : 'text-bg-primary');
    const toastEl = document.createElement('div');
    toastEl.className = `toast align-items-center ${bgClass} border-0 shadow`;
    toastEl.setAttribute('role', 'alert');
    toastEl.setAttribute('aria-live', 'assertive');
    toastEl.setAttribute('aria-atomic', 'true');

    toastEl.innerHTML = `
        <div class="d-flex">
            <div class="toast-body small fw-medium">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;

    container.appendChild(toastEl);
    const bsToast = new bootstrap.Toast(toastEl, { delay: 4000 });
    bsToast.show();

    toastEl.addEventListener('hidden.bs.toast', () => {
        toastEl.remove();
    });
}

// Copy to Clipboard Utility
function initCopyButtons() {
    document.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-target');
            const targetEl = document.getElementById(targetId);
            if (targetEl) {
                const textToCopy = targetEl.innerText || targetEl.textContent;
                navigator.clipboard.writeText(textToCopy).then(() => {
                    showToast('<i class="bi bi-check-circle me-1"></i> Content copied to clipboard!');
                }).catch(err => {
                    showToast('Failed to copy content.', 'danger');
                });
            }
        });
    });
}
