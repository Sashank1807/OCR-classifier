// History page live search, filtering, and delete actions
document.addEventListener('DOMContentLoaded', () => {
    const historySearchForm = document.getElementById('historySearchForm');
    const searchQuery = document.getElementById('searchQuery');
    const filterCategory = document.getElementById('filterCategory');
    const filterLanguage = document.getElementById('filterLanguage');
    const resetFilterBtn = document.getElementById('resetFilterBtn');
    const tableBody = document.getElementById('historyTableBody');

    if (historySearchForm) {
        historySearchForm.addEventListener('submit', (e) => {
            e.preventDefault();
            performSearch();
        });
    }

    if (resetFilterBtn) {
        resetFilterBtn.addEventListener('click', () => {
            if (searchQuery) searchQuery.value = '';
            if (filterCategory) filterCategory.value = 'All';
            if (filterLanguage) filterLanguage.value = 'All';
            performSearch();
        });
    }

    async function performSearch() {
        const q = searchQuery ? searchQuery.value.trim() : '';
        const cat = filterCategory ? filterCategory.value : 'All';
        const lang = filterLanguage ? filterLanguage.value : 'All';

        const params = new URLSearchParams();
        if (q) params.append('q', q);
        if (cat !== 'All') params.append('doc_type', cat);
        if (lang !== 'All') params.append('language', lang);

        try {
            const response = await fetch(`/api/v1/history/search?${params.toString()}`);
            const data = await response.json();
            if (response.ok && data.success) {
                renderHistoryRows(data.documents);
            }
        } catch (err) {
            showToast('Search failed.', 'danger');
        }
    }

    function renderHistoryRows(documents) {
        if (!tableBody) return;
        tableBody.innerHTML = '';

        if (!documents || documents.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="9" class="text-center py-5 text-body-secondary">No matching OCR records found.</td>
                </tr>
            `;
            return;
        }

        documents.forEach(doc => {
            const row = document.createElement('tr');
            row.id = `row-${doc.id}`;
            row.innerHTML = `
                <td class="ps-4 text-body-secondary small">#${doc.id}</td>
                <td class="fw-medium text-truncate" style="max-width: 200px;" title="${doc.filename}">
                    <i class="bi bi-file-earmark-text me-2 text-primary"></i>${doc.filename}
                </td>
                <td><span class="badge text-bg-primary-subtle text-primary border border-primary-subtle">${doc.document_type}</span></td>
                <td><span class="badge text-bg-secondary-subtle text-secondary">${doc.language}</span></td>
                <td>
                    ${doc.has_handwriting ? 
                        '<span class="badge text-bg-warning-subtle text-warning border border-warning-subtle">Handwritten</span>' : 
                        '<span class="badge text-bg-info-subtle text-info border border-info-subtle">Printed</span>'}
                </td>
                <td>${doc.page_count}</td>
                <td class="small text-body-secondary">${doc.processing_time}s</td>
                <td class="small text-body-secondary">${doc.created_at}</td>
                <td class="pe-4 text-end">
                    <div class="btn-group btn-group-sm">
                        <a href="/result/${doc.id}" class="btn btn-outline-primary" title="View Result"><i class="bi bi-eye"></i></a>
                        <a href="/api/v1/ocr/export/${doc.id}?format=json" class="btn btn-outline-secondary" title="Download JSON"><i class="bi bi-download"></i></a>
                        <button class="btn btn-outline-danger delete-doc-btn" data-id="${doc.id}" title="Delete Record"><i class="bi bi-trash"></i></button>
                    </div>
                </td>
            `;
            tableBody.appendChild(row);
        });

        attachDeleteListeners();
    }

    function attachDeleteListeners() {
        document.querySelectorAll('.delete-doc-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const docId = btn.getAttribute('data-id');
                if (!docId) return;

                if (!confirm(`Are you sure you want to delete Document #${docId}?`)) return;

                try {
                    const response = await fetch(`/api/v1/history/delete/${docId}`, { method: 'DELETE' });
                    const data = await response.json();
                    if (response.ok && data.success) {
                        showToast('Document record deleted.');
                        const row = document.getElementById(`row-${docId}`);
                        if (row) row.remove();
                    } else {
                        throw new Error(data.detail || 'Delete failed.');
                    }
                } catch (err) {
                    showToast(`Error: ${err.message}`, 'danger');
                }
            });
        });
    }

    attachDeleteListeners();
});
