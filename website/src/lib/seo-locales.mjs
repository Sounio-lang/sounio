/**
 * Canonical URLs and hreflang clusters for souniolang.org.
 *
 * Every locale URL self-canonicalizes to its own trailing-slash URL on
 * https://www.souniolang.org. A non-English URL joins the hreflang cluster
 * and the sitemap only when its main document is a translation of the
 * current English page. Anything else is noindex (BaseLayout) and stays
 * out of every hreflang set. Pointing hreflang at a URL that is not that
 * translation is what let a stale Chinese stub outrank the English page.
 *
 * hreflang and <html lang> use one BCP 47 tag per URL prefix, the same tags
 * the sitemap already published: pt-BR, zh-CN, zh-HK. URL prefixes stay short
 * (`/pt`, `/zh`, `/zh-hk`). See website/docs/I18N_POLICY.md.
 */
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';

// Resolved from the website directory. `astro build`, `astro.config.mjs`,
// and `npm run check:*` all run with that directory as cwd. import.meta.url
// is not safe here: Vite copies this module into dist/.prerender and the
// bundled URL no longer sits next to src/content.
const websiteRoot = process.cwd();
const contentRoot = path.join(websiteRoot, 'src/content');

export const SITE = 'https://www.souniolang.org';

/** URL prefix → BCP 47 hreflang / html lang. Order is the emitted order. */
export const LOCALES = ['en', 'pt', 'el', 'zh', 'ja', 'es', 'zh-hk'];

export const HREFLANG = {
  en: 'en',
  pt: 'pt-BR',
  el: 'el',
  zh: 'zh-CN',
  ja: 'ja',
  es: 'es',
  'zh-hk': 'zh-HK',
};

const NON_EN = LOCALES.filter((locale) => locale !== 'en');
const LOCALE_SET = new Set(LOCALES);

export function htmlLang(locale) {
  return HREFLANG[locale] ?? 'en';
}

function readJson(relativePath) {
  return JSON.parse(readFileSync(path.join(websiteRoot, relativePath), 'utf8'));
}

function walk(dir, out = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, out);
    else if (entry.isFile() && /\.mdx?$/.test(entry.name)) out.push(full);
  }
  return out;
}

function splitFrontmatter(text) {
  if (!text.startsWith('---')) return { frontmatter: '', body: text };
  const end = text.indexOf('\n---', 3);
  if (end === -1) return { frontmatter: '', body: text };
  return { frontmatter: text.slice(3, end), body: text.slice(end + 4) };
}

function normalizeBody(body) {
  return body.replace(/\s+/g, ' ').trim();
}

function docSlug(parts) {
  let slug = parts.join('/').replace(/\.mdx?$/, '');
  if (slug.endsWith('/index')) slug = slug.slice(0, -'/index'.length);
  return slug;
}

/**
 * Map of collection slug → Map(locale → normalized prose).
 * English blog/tutorial/showcase files live at the collection root.
 * Docs use an `en/` directory. A locale file whose prose matches English
 * is stored, then dropped by localesForEntry.
 */
function loadCollection(name, slugFor) {
  const root = path.join(contentRoot, name);
  const entries = new Map();
  for (const file of walk(root)) {
    const rel = path.relative(root, file).split(path.sep);
    let locale = 'en';
    if (LOCALE_SET.has(rel[0])) {
      locale = rel[0];
      rel.shift();
    }
    const { frontmatter, body } = splitFrontmatter(readFileSync(file, 'utf8'));
    const slug = slugFor(rel, frontmatter);
    if (!slug) continue;
    if (!entries.has(slug)) entries.set(slug, new Map());
    entries.get(slug).set(locale, normalizeBody(body));
  }
  return entries;
}

function seriesFromFrontmatter(frontmatter) {
  const match = /^series:\s*(.+)$/m.exec(frontmatter);
  if (!match) return '';
  return match[1].trim().replace(/^['"]|['"]$/g, '');
}

const docs = loadCollection('docs', (parts) => docSlug(parts));
const showcases = loadCollection('showcases', (parts) => docSlug(parts));
const blog = loadCollection('blog', (parts) => docSlug(parts));
const tutorials = loadCollection('tutorials', (parts, frontmatter) => {
  const series = seriesFromFrontmatter(frontmatter);
  const fileSlug = docSlug(parts);
  if (!series || !fileSlug) return '';
  return `${series}/${fileSlug}`;
});

function localesForEntry(byLocale) {
  if (!byLocale || !byLocale.has('en')) return ['en'];
  const english = byLocale.get('en');
  const translated = NON_EN.filter((locale) => byLocale.has(locale) && byLocale.get(locale) !== english);
  return ['en', ...translated];
}

/**
 * Identifiers the current English page actually talks about: code spans,
 * E0xx diagnostics, and camelCase or snake_case names. A locale file whose
 * prose merely differs can still be an older page (the effect-system stubs
 * are titled "Effect System" and share none of the checker names).
 */
function documentAnchors(body) {
  const found = new Set();
  for (const match of body.matchAll(/`([^`]{3,80})`/g)) found.add(match[1]);
  for (const match of body.matchAll(/\bE\d{3}\b/g)) found.add(match[0]);
  for (const match of body.matchAll(/\b[A-Za-z_][A-Za-z0-9_]{5,}\b/g)) {
    const word = match[0];
    if ((/[A-Z]/.test(word) && /[a-z]/.test(word)) || word.includes('_')) found.add(word);
  }
  return found;
}

function sameDocument(english, localized) {
  const anchors = documentAnchors(english);
  if (anchors.size < 20) return true;
  let shared = 0;
  for (const token of documentAnchors(localized)) {
    if (anchors.has(token)) shared += 1;
  }
  return shared > 2;
}

/**
 * `compiler/effect-system` and `effects` were rewritten in English. The
 * locale files are the previous short pages, so a prose inequality is not
 * a translation. Other documents keep the prose test.
 */
const SAME_DOCUMENT_SLUGS = new Set(['compiler/effect-system', 'effects']);

function localesForDoc(slug, byLocale) {
  const locales = localesForEntry(byLocale);
  if (!SAME_DOCUMENT_SLUGS.has(slug) || !byLocale || !byLocale.has('en')) return locales;
  const english = byLocale.get('en');
  return locales.filter((locale) => locale === 'en' || sameDocument(english, byLocale.get(locale)));
}

function localesWithAny(collection) {
  const found = new Set(['en']);
  for (const byLocale of collection.values()) {
    for (const locale of localesForEntry(byLocale)) found.add(locale);
  }
  return LOCALES.filter((locale) => found.has(locale));
}

function seriesLocales(series) {
  const found = new Set(['en']);
  for (const [slug, byLocale] of tutorials) {
    if (!slug.startsWith(`${series}/`)) continue;
    for (const locale of localesForEntry(byLocale)) found.add(locale);
  }
  return LOCALES.filter((locale) => found.has(locale));
}

function jsonLocales(relativePath) {
  const data = readJson(relativePath);
  const english = JSON.stringify(data.en);
  return LOCALES.filter((locale) => {
    if (locale === 'en') return true;
    if (data[locale] == null) return false;
    return JSON.stringify(data[locale]) !== english;
  });
}

const homeLocales = jsonLocales('src/data/publicSurface.json');
const scientistLocales = jsonLocales('src/data/scientistCopy.json');
const learnHubLocales = LOCALES.filter((locale) => homeLocales.includes(locale) || localesWithAny(docs).includes(locale));
const scienceHubLocales = localesWithAny(showcases);
const tutorialHubLocales = localesWithAny(tutorials);

/**
 * Shell routes whose main prose is translated in code or JSON rather than
 * in a content collection. Anything not listed here, and not a content
 * document, is English-only: translated chrome or a handful of labels do
 * not count (proof, packages, playground, plan, credits,
 * dissertation, the insights index, the showcases index).
 *
 * Explicit shells:
 * - `/` and `/learn/` follow publicSurface.json (the learn hub also joins
 *   any locale that has a translated doc, because those titles are the index).
 * - `/science/`, `/tutorials/` and `/tutorials/<series>/` follow locale
 *   documents listed on the page.
 * - `/insights/` stays English-only: the index reads English posts only.
 * - `/language/` and `/compiler/` branch the article for `el` only.
 * - `/scientist/` follows scientistCopy.json (`pt`, `el`).
 * - `/honesty/` has a full Portuguese article; other locales render the
 *   English article.
 * - `/about/` has a full Portuguese article; other locales render the
 *   English article. `/about/vision/` and `/about/roadmap/` stay English.
 * - `/releases/` carries a distinct notes array for every locale, including
 *   `zh-hk`, in src/pages/releases/index.astro.
 * - `/learn/epistemic/expanded-uncertainty/` has an English page and a
 *   Japanese article. hreflang is only en, ja, and x-default (English).
 *   `/pt/learn/epistemic/expanded-uncertainty/` is a stub of the English
 *   article: self-canonical, not indexable, not an hreflang target.
 */
const EXPLICIT_SHELLS = {
  '/': homeLocales,
  '/learn/': learnHubLocales,
  '/science/': scienceHubLocales,
  '/tutorials/': tutorialHubLocales,
  '/releases/': [...LOCALES],
  '/language/': ['en', 'el'],
  '/compiler/': ['en', 'el'],
  '/scientist/': scientistLocales,
  '/honesty/': ['en', 'pt'],
  '/about/': ['en', 'pt'],
  '/learn/epistemic/expanded-uncertainty/': ['en', 'ja'],
};

const REDIRECT_EXACT = new Set([
  '/blog/',
  '/changelog/',
  '/manifesto/',
  '/roadmap/',
  '/docs/',
  '/platform/',
]);

export function barePath(pathname) {
  const clean = pathname.split('?')[0].split('#')[0];
  const parts = clean.split('/').filter(Boolean);
  if (parts.length > 0 && LOCALE_SET.has(parts[0]) && parts[0] !== 'en') parts.shift();
  if (parts.length === 0) return '/';
  return `/${parts.join('/')}/`;
}

export function localeFromPath(pathname) {
  const parts = pathname.split('?')[0].split('#')[0].split('/').filter(Boolean);
  if (parts.length > 0 && LOCALE_SET.has(parts[0]) && parts[0] !== 'en') return parts[0];
  return 'en';
}

export function localizedPath(bare, locale) {
  const pathName = bare === '/' ? '/' : (bare.endsWith('/') ? bare : `${bare}/`);
  if (locale === 'en') return pathName;
  return pathName === '/' ? `/${locale}/` : `/${locale}${pathName}`;
}

export function absoluteUrl(pathname) {
  const withSlash = pathname === '/' ? '/' : (pathname.endsWith('/') ? pathname : `${pathname}/`);
  return new URL(withSlash, SITE).href;
}

export function isRedirectPath(pathname) {
  const bare = barePath(pathname);
  if (REDIRECT_EXACT.has(bare)) return true;
  if (bare.startsWith('/docs/')) return true;
  if (bare.startsWith('/registry/')) return true;
  if (bare.startsWith('/showcases/') && bare !== '/showcases/') return true;
  return false;
}

function contentLocales(bare) {
  const learn = /^\/learn\/(.+)\/$/.exec(bare);
  if (learn) return docs.has(learn[1]) ? localesForDoc(learn[1], docs.get(learn[1])) : null;
  const science = /^\/science\/([^/]+)\/$/.exec(bare);
  if (science) return showcases.has(science[1]) ? localesForEntry(showcases.get(science[1])) : null;
  const insight = /^\/insights\/([^/]+)\/$/.exec(bare);
  if (insight) return blog.has(insight[1]) ? localesForEntry(blog.get(insight[1])) : null;
  const lesson = /^\/tutorials\/([^/]+)\/([^/]+)\/$/.exec(bare);
  if (lesson) {
    const key = `${lesson[1]}/${lesson[2]}`;
    return tutorials.has(key) ? localesForEntry(tutorials.get(key)) : null;
  }
  return null;
}

/** Locales whose URL for this path is an indexable translation. Always includes `en` for a real page. */
export function indexableLocales(pathname) {
  const bare = barePath(pathname);
  if (isRedirectPath(bare)) return ['en'];
  const fromContent = contentLocales(bare);
  if (fromContent) return fromContent;
  if (Object.prototype.hasOwnProperty.call(EXPLICIT_SHELLS, bare)) return EXPLICIT_SHELLS[bare];
  const series = /^\/tutorials\/([^/]+)\/$/.exec(bare);
  if (series && seriesLocales(series[1]).length > 1) return seriesLocales(series[1]);
  return ['en'];
}

/** The URL's own path, with a trailing slash. Canonicals do not cross locales. */
export function canonicalPathname(pathname) {
  return localizedPath(barePath(pathname), localeFromPath(pathname));
}

export function isIndexableUrl(pathname) {
  if (isRedirectPath(pathname)) return false;
  return indexableLocales(pathname).includes(localeFromPath(pathname));
}

/**
 * @param {string} pathname
 * @returns {{ hreflang: string, href: string }[]}
 */
export function hreflangLinks(pathname) {
  const bare = barePath(pathname);
  const links = indexableLocales(bare).map((locale) => ({
    hreflang: HREFLANG[locale],
    href: absoluteUrl(localizedPath(bare, locale)),
  }));
  links.push({
    hreflang: 'x-default',
    href: absoluteUrl(localizedPath(bare, 'en')),
  });
  return links;
}

/** Sitemap <loc> values: self-canonical, not a redirect. */
export function isIndexableSitemapUrl(urlString) {
  return isIndexableUrl(new URL(urlString).pathname);
}

export function sitemapLinks(urlString) {
  return hreflangLinks(new URL(urlString).pathname).map((link) => ({
    lang: link.hreflang,
    url: link.href,
  }));
}
