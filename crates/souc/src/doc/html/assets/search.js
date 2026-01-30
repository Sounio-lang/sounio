// Demetrios Documentation - Search JavaScript

(function() {
    'use strict';

    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');

    if (!searchInput || !searchResults) return;

    let searchIndex = null;
    let debounceTimer = null;

    // Load search index when available
    if (window.searchIndex) {
        searchIndex = window.searchIndex;
    }

    function search(query) {
        if (!searchIndex || query.length < 2) {
            searchResults.innerHTML = '';
            searchResults.classList.remove('active');
            return;
        }

        const queryLower = query.toLowerCase();
        const results = [];
        const seen = new Set();

        // Search by name
        if (searchIndex.by_name) {
            for (const [name, entries] of Object.entries(searchIndex.by_name)) {
                if (name.includes(queryLower)) {
                    for (const entry of entries) {
                        if (!seen.has(entry.path)) {
                            seen.add(entry.path);
                            results.push({
                                ...entry,
                                score: name === queryLower ? 100 :
                                       name.startsWith(queryLower) ? 50 : 10
                            });
                        }
                    }
                }
            }
        }

        // Search in terms for full-text search
        if (searchIndex.terms) {
            for (const [term, paths] of Object.entries(searchIndex.terms)) {
                if (term.includes(queryLower)) {
                    for (const path of paths) {
                        if (!seen.has(path)) {
                            // Find the entry for this path
                            for (const entries of Object.values(searchIndex.by_name)) {
                                for (const entry of entries) {
                                    if (entry.path === path && !seen.has(path)) {
                                        seen.add(path);
                                        results.push({ ...entry, score: 5 });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by score
        results.sort((a, b) => b.score - a.score);

        // Limit results
        const topResults = results.slice(0, 20);

        // Render results
        if (topResults.length === 0) {
            searchResults.innerHTML = '<div class="search-result">No results found</div>';
        } else {
            searchResults.innerHTML = topResults.map(result => `
                <div class="search-result" data-path="${escapeHtml(result.path)}">
                    <div>
                        <span class="search-result-name">${escapeHtml(result.name)}</span>
                        <span class="search-result-kind">${escapeHtml(result.kind)}</span>
                    </div>
                    ${result.desc ? `<div class="search-result-desc">${escapeHtml(result.desc)}</div>` : ''}
                </div>
            `).join('');
        }

        searchResults.classList.add('active');
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/&/g, '&amp;')
                  .replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;')
                  .replace(/"/g, '&quot;');
    }

    function pathToUrl(path) {
        // Convert module::path::item to URL
        const parts = path.split('::');
        if (parts.length === 1) {
            return '/index.html';
        }

        // Determine item type from search index
        let itemType = 'fn';
        if (searchIndex && searchIndex.by_name) {
            const name = parts[parts.length - 1].toLowerCase();
            const entries = searchIndex.by_name[name];
            if (entries) {
                const entry = entries.find(e => e.path === path);
                if (entry) {
                    switch (entry.kind) {
                        case 'struct': itemType = 'struct'; break;
                        case 'enum': itemType = 'enum'; break;
                        case 'trait': itemType = 'trait'; break;
                        case 'const': itemType = 'const'; break;
                        case 'type': itemType = 'type'; break;
                        case 'mod': itemType = 'index'; break;
                        default: itemType = 'fn';
                    }
                }
            }
        }

        const moduleParts = parts.slice(0, -1);
        const itemName = parts[parts.length - 1];

        if (itemType === 'index') {
            return '/' + parts.join('/') + '/index.html';
        }

        return '/' + moduleParts.slice(1).join('/') + '/' + itemType + '.' + itemName + '.html';
    }

    // Event listeners
    searchInput.addEventListener('input', function(e) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(function() {
            search(e.target.value);
        }, 150);
    });

    searchInput.addEventListener('focus', function() {
        if (this.value.length >= 2) {
            searchResults.classList.add('active');
        }
    });

    document.addEventListener('click', function(e) {
        if (!searchResults.contains(e.target) && e.target !== searchInput) {
            searchResults.classList.remove('active');
        }
    });

    searchResults.addEventListener('click', function(e) {
        const result = e.target.closest('.search-result');
        if (result && result.dataset.path) {
            window.location.href = pathToUrl(result.dataset.path);
        }
    });

    // Keyboard navigation
    searchInput.addEventListener('keydown', function(e) {
        const results = searchResults.querySelectorAll('.search-result');
        const activeResult = searchResults.querySelector('.search-result.active');
        let index = -1;

        if (activeResult) {
            index = Array.from(results).indexOf(activeResult);
        }

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (activeResult) activeResult.classList.remove('active');
            const nextIndex = (index + 1) % results.length;
            results[nextIndex].classList.add('active');
            results[nextIndex].scrollIntoView({ block: 'nearest' });
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (activeResult) activeResult.classList.remove('active');
            const prevIndex = index <= 0 ? results.length - 1 : index - 1;
            results[prevIndex].classList.add('active');
            results[prevIndex].scrollIntoView({ block: 'nearest' });
        } else if (e.key === 'Enter') {
            e.preventDefault();
            const selected = activeResult || results[0];
            if (selected && selected.dataset.path) {
                window.location.href = pathToUrl(selected.dataset.path);
            }
        } else if (e.key === 'Escape') {
            searchResults.classList.remove('active');
            searchInput.blur();
        }
    });

    // Global shortcut
    document.addEventListener('keydown', function(e) {
        if ((e.key === '/' || e.key === 's') && !e.ctrlKey && !e.metaKey) {
            const activeElement = document.activeElement;
            if (activeElement.tagName !== 'INPUT' && activeElement.tagName !== 'TEXTAREA') {
                e.preventDefault();
                searchInput.focus();
            }
        }
    });
})();
