import { useState, useEffect, useRef, useCallback, type MouseEvent } from 'react';

declare global {
  interface Window {
    pagefind?: {
      search: (query: string) => Promise<{
        results: Array<{
          data: () => Promise<{
            title: string;
            excerpt: string;
            url: string;
          }>;
        }>;
      }>;
    };
  }
}

type PagefindSearchResult = {
  data: () => Promise<{ title: string; excerpt: string; url: string }>;
};

interface SearchResult {
  title: string;
  excerpt: string;
  url: string;
}

interface SearchStrings {
  button: string;
  placeholder: string;
  searching: string;
  noResults: string;
  noResultsHint: string;
  startTyping: string;
  startTypingHint: string;
  toSelect: string;
  toNavigate: string;
  poweredBy: string;
}

interface Props {
  locale: string;
  strings: SearchStrings;
}

const NON_DEFAULT_LOCALES = ['pt', 'el', 'zh', 'ja', 'es'] as const;

function urlMatchesLocale(url: string, locale: string): boolean {
  // Pagefind returns site-relative urls (e.g. "/pt/docs/...").
  // Treat the default locale ("en") as "no locale prefix".
  const path = url.startsWith('http') ? new URL(url).pathname : url;

  if (locale === 'en') {
    return !NON_DEFAULT_LOCALES.some((l) => path === `/${l}` || path.startsWith(`/${l}/`));
  }

  return path === `/${locale}` || path.startsWith(`/${locale}/`);
}

export default function Search({ locale, strings }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const searchTimeout = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const ensurePagefind = useCallback(async () => {
    if (window.pagefind) return window.pagefind;

    const pagefindUrl = ['', 'pagefind', 'pagefind.js'].join('/');
    const mod = await import(/* @vite-ignore */ pagefindUrl);
    window.pagefind = mod;
    return mod;
  }, []);

  // Keyboard shortcut to open search (Cmd/Ctrl + K, or /)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const active = document.activeElement;
      const isTyping =
        active &&
        (active.tagName === 'INPUT' ||
          active.tagName === 'TEXTAREA' ||
          (active as HTMLElement).isContentEditable);

      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(true);
      }

      if (!isTyping && !e.metaKey && !e.ctrlKey && e.key === '/') {
        e.preventDefault();
        setIsOpen(true);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsOpen(false);
        setQuery('');
        setResults([]);
      }
    };
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isOpen]);

  // Focus input when modal opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Debounced search using Pagefind API
  const performSearch = useCallback(async (q: string) => {
    if (!q.trim()) {
      setResults([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    try {
      const pagefind = await ensurePagefind();
      const search = await pagefind.search(q);
      const formattedResults = await Promise.all(
        search.results.slice(0, 10).map(async (result: PagefindSearchResult) => {
          const data = await result.data();
          return {
            title: data.title,
            excerpt: data.excerpt,
            url: data.url,
          };
        })
      );
      setResults(formattedResults.filter((r) => urlMatchesLocale(r.url, locale)));
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [ensurePagefind, locale]);

  // Debounce search input
  useEffect(() => {
    if (searchTimeout.current) {
      clearTimeout(searchTimeout.current);
    }

    searchTimeout.current = setTimeout(() => {
      performSearch(query);
    }, 200);

    return () => {
      if (searchTimeout.current) {
        clearTimeout(searchTimeout.current);
      }
    };
  }, [query, performSearch]);

  const closeModal = () => {
    setIsOpen(false);
    setQuery('');
    setResults([]);
  };

  const handleBackdropClick = (e: MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      closeModal();
    }
  };

  // Trigger button for opening search
  const SearchButton = () => (
    <button
      onClick={() => {
        void ensurePagefind();
        setIsOpen(true);
      }}
      className="flex items-center gap-2 px-3 py-1.5 text-sm text-[var(--color-text-muted)] bg-[var(--color-bg-alt)] border border-[var(--color-border)] rounded-lg hover:border-[var(--color-gold-500)] transition-colors"
      aria-label="Open search"
    >
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
      <span className="hidden sm:inline">{strings.button}</span>
      <kbd className="hidden sm:inline-block px-1.5 py-0.5 text-xs bg-[var(--color-bg)] border border-[var(--color-border)] rounded">⌘K</kbd>
    </button>
  );

  if (!isOpen) {
    return <SearchButton />;
  }

  return (
    <>
      <SearchButton />
      <div
        className="fixed inset-0 z-50 flex items-start justify-center pt-[10vh] px-4 bg-black/50 backdrop-blur-sm"
        onClick={handleBackdropClick}
        role="dialog"
        aria-modal="true"
        aria-label="Search documentation"
      >
        <div className="glass glass-elevated w-full max-w-2xl rounded-xl border border-[var(--glass-border)] overflow-hidden">
          {/* Search input */}
          <div className="flex items-center gap-3 p-4 border-b border-[var(--color-border)]">
            <svg className="w-5 h-5 text-[var(--color-text-muted)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={strings.placeholder}
              className="flex-1 bg-transparent text-lg text-[var(--color-text)] placeholder-[var(--color-text-muted)] focus:outline-none"
              aria-label="Search input"
            />
            <button
              onClick={closeModal}
              className="p-1 text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors"
              aria-label="Close search"
            >
              <kbd className="px-1.5 py-0.5 text-xs bg-[var(--color-bg-alt)] border border-[var(--color-border)] rounded">Esc</kbd>
            </button>
          </div>

          {/* Results area */}
          <div className="max-h-[60vh] overflow-y-auto">
            {loading ? (
              <div className="flex items-center justify-center py-12 text-[var(--color-text-muted)]">
                <svg className="animate-spin w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                {strings.searching}
              </div>
            ) : query.trim() ? (
              results.length > 0 ? (
                <ul className="py-2" role="list">
                  {results.map((result, index) => (
                    <li key={index}>
                      <a
                        href={result.url}
                        className="block px-4 py-3 hover:bg-[var(--color-bg-alt)] transition-colors"
                        onClick={closeModal}
                      >
                        <h3 className="font-medium text-[var(--color-text)] mb-1">{result.title}</h3>
                        <p
                          className="text-sm text-[var(--color-text-muted)] line-clamp-2 [&_mark]:bg-[var(--color-gold-500)]/30 [&_mark]:text-[var(--color-text)] [&_mark]:rounded"
                          dangerouslySetInnerHTML={{ __html: result.excerpt }}
                        />
                      </a>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="py-12 text-center text-[var(--color-text-muted)]">
                  <svg className="w-12 h-12 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p>
                    {strings.noResults} <span className="font-mono">"{query}"</span>
                  </p>
                  <p className="text-sm mt-1">{strings.noResultsHint}</p>
                </div>
              )
            ) : (
              <div className="py-12 text-center text-[var(--color-text-muted)]">
                <p>{strings.startTyping}</p>
                <p className="text-sm mt-1">{strings.startTypingHint}</p>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between px-4 py-3 border-t border-[var(--color-border)] text-xs text-[var(--color-text-muted)]">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1">
                <kbd className="px-1 py-0.5 bg-[var(--color-bg-alt)] border border-[var(--color-border)] rounded">↵</kbd>
                {strings.toSelect}
              </span>
              <span className="flex items-center gap-1">
                <kbd className="px-1 py-0.5 bg-[var(--color-bg-alt)] border border-[var(--color-border)] rounded">↑↓</kbd>
                {strings.toNavigate}
              </span>
            </div>
            <span>{strings.poweredBy}</span>
          </div>
        </div>
      </div>
    </>
  );
}
