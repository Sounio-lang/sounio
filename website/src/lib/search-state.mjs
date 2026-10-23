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

export async function searchResultsForLocale(results, locale, availableLocales, isActive = () => true, limit = 10) {
  const matches = [];
  for (let offset = 0; offset < results.length && matches.length < limit; offset += limit) {
    if (!isActive()) return [];
    const batch = await Promise.all(results.slice(offset, offset + limit).map(result => result.data()));
    if (!isActive()) return [];
    matches.push(...batch.filter(result => urlMatchesLocale(result.url, locale, availableLocales)));
  }
  return matches.slice(0, limit);
}
