export function urlMatchesLocale(value, locale, availableLocales, defaultLocale = 'en') {
  let pathname;
  try { pathname = new URL(value, 'https://www.souniolang.org').pathname; } catch { return false; }
  if (!availableLocales.includes(locale)) return false;
  if (locale === defaultLocale) return !availableLocales.filter(entry => entry !== defaultLocale).some(entry => pathname === `/${entry}` || pathname.startsWith(`/${entry}/`));
  return pathname === `/${locale}` || pathname.startsWith(`/${locale}/`);
}

export function createRequestGuard() {
  let current = null;
  return {
    begin() { current = {}; return current; },
    isCurrent(request) { return request !== null && request === current; },
    invalidate() { current = null; },
  };
}

/** Pagefind records directory pages as `/about` or `/about/index.html`. Canonical HTML routes end with `/`. */
export function withDirectorySlash(value) {
  let url;
  try { url = new URL(value, 'https://www.souniolang.org'); } catch { return value; }
  let pathname = url.pathname;
  if (pathname.endsWith('/index.html')) pathname = pathname.slice(0, -'index.html'.length);
  const last = pathname.split('/').filter(Boolean).pop() ?? '';
  if (last.endsWith('.html')) {
    pathname = `${pathname.slice(0, -'.html'.length)}/`;
  } else if (!last.includes('.') && pathname !== '/' && !pathname.endsWith('/')) {
    pathname += '/';
  }
  const next = `${pathname}${url.search}${url.hash}`;
  if (value.startsWith('/') || value.startsWith('#') || value.startsWith('?')) return next;
  return new URL(next, url.origin).href;
}

export async function searchResultsForLocale(results, locale, availableLocales, isActive = () => true, limit = 10) {
  const matches = [];
  for (let offset = 0; offset < results.length && matches.length < limit; offset += limit) {
    if (!isActive()) return [];
    const batch = await Promise.all(results.slice(offset, offset + limit).map(result => result.data()));
    if (!isActive()) return [];
    matches.push(...batch
      .filter(result => urlMatchesLocale(result.url, locale, availableLocales))
      .map(result => ({ ...result, url: withDirectorySlash(result.url) })));
  }
  return matches.slice(0, limit);
}
