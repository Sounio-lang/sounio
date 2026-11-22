/**
 * Reciprocal hreflang + canonical check against dist/ and the sitemap.
 *
 * Expectations come from src/lib/seo-locales.mjs, the same module BaseLayout
 * and the sitemap filter use. Redirect stubs (meta refresh) are not indexed.
 */
import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import {
  HREFLANG,
  SITE,
  absoluteUrl,
  canonicalPathname,
  hreflangLinks,
  htmlLang,
  indexableLocales,
  isIndexableUrl,
  isRedirectPath,
  localeFromPath,
} from '../src/lib/seo-locales.mjs';

const root = path.resolve(process.cwd());
const distDir = path.join(root, 'dist');
const ALLOWED_CODES = new Set([...Object.values(HREFLANG), 'x-default']);

function assertExamples() {
  const errors = [];
  const packages = canonicalPathname('/es/packages/');
  if (packages !== '/es/packages/') errors.push(`expected /es/packages/ to self-canonicalize, got ${packages}`);
  if (isIndexableUrl('/es/packages/')) errors.push('/es/packages/ is not a Spanish translation and must be noindex');
  const effect = canonicalPathname('/zh/learn/compiler/effect-system/');
  if (effect !== '/zh/learn/compiler/effect-system/') {
    errors.push(`expected zh effect-system to self-canonicalize, got ${effect}`);
  }
  if (isIndexableUrl('/zh/learn/compiler/effect-system/')) {
    errors.push('zh effect-system stub is not a translation of the checker page');
  }
  const units = canonicalPathname('/zh-hk/learn/language/units-and-uncertainty/');
  if (units !== '/zh-hk/learn/language/units-and-uncertainty/') {
    errors.push(`expected zh-hk units page to self-canonicalize, got ${units}`);
  }
  if (isIndexableUrl('/zh-hk/learn/language/units-and-uncertainty/')) {
    errors.push('zh-hk units page is not a translation');
  }
  const effectCodes = hreflangLinks('/learn/compiler/effect-system/').map((link) => link.hreflang);
  const effectDefault = hreflangLinks('/learn/compiler/effect-system/').find((link) => link.hreflang === 'x-default');
  if (effectCodes.includes('zh-CN') || effectCodes.includes('zh') || !effectCodes.includes('en') || effectDefault?.href !== absoluteUrl('/learn/compiler/effect-system/')) {
    errors.push(`effect-system hreflang codes drifted: ${effectCodes.join(',')}`);
  }
  const effectsDefault = hreflangLinks('/learn/effects/').find((link) => link.hreflang === 'x-default');
  if (indexableLocales('/learn/effects/').some((locale) => locale !== 'en') || effectsDefault?.href !== absoluteUrl('/learn/effects/')) {
    errors.push(`effects hreflang still lists stale locales: ${indexableLocales('/learn/effects/').join(',')}`);
  }
  const packageCodes = hreflangLinks('/packages/').map((link) => link.hreflang);
  if (packageCodes.includes('es') || packageCodes.includes('pt-BR')) {
    errors.push(`packages hreflang still lists untranslated locales: ${packageCodes.join(',')}`);
  }
  const aboutPt = canonicalPathname('/pt/about/');
  if (aboutPt !== '/pt/about/' || !isIndexableUrl('/pt/about/')) errors.push(`expected /pt/about/ to stay indexable, got ${aboutPt}`);
  const aboutEl = canonicalPathname('/el/about/');
  if (aboutEl !== '/el/about/' || isIndexableUrl('/el/about/')) {
    errors.push(`expected /el/about/ to self-canonicalize and stay out of the index, got ${aboutEl}`);
  }
  const roadmapPt = canonicalPathname('/pt/about/roadmap/');
  if (roadmapPt !== '/pt/about/roadmap/' || isIndexableUrl('/pt/about/roadmap/')) {
    errors.push(`expected /pt/about/roadmap/ to self-canonicalize and stay out of the index, got ${roadmapPt}`);
  }
  const aboutCodes = hreflangLinks('/about/').map((link) => link.hreflang);
  const aboutExpected = ['en', 'pt-BR', 'x-default'];
  if (aboutCodes.length !== aboutExpected.length || aboutExpected.some((code) => !aboutCodes.includes(code))) {
    errors.push(`about hreflang codes drifted: ${aboutCodes.join(',')}`);
  }
  const expandedCodes = hreflangLinks('/learn/epistemic/expanded-uncertainty/').map((link) => link.hreflang);
  const expandedExpected = ['en', 'ja', 'x-default'];
  if (
    expandedCodes.length !== expandedExpected.length
    || expandedExpected.some((code) => !expandedCodes.includes(code))
    || expandedCodes.includes('pt-BR')
    || expandedCodes.includes('zh-CN')
  ) {
    errors.push(`expanded-uncertainty hreflang drifted: ${expandedCodes.join(',')}`);
  }
  const expandedJa = canonicalPathname('/ja/learn/epistemic/expanded-uncertainty/');
  if (expandedJa !== '/ja/learn/epistemic/expanded-uncertainty/') {
    errors.push(`expected ja expanded-uncertainty to self-canonicalize, got ${expandedJa}`);
  }
  const expandedPt = canonicalPathname('/pt/learn/epistemic/expanded-uncertainty/');
  if (expandedPt !== '/pt/learn/epistemic/expanded-uncertainty/' || isIndexableUrl('/pt/learn/epistemic/expanded-uncertainty/')) {
    errors.push(`expected pt expanded-uncertainty to self-canonicalize and stay out of the index, got ${expandedPt}`);
  }
  const expandedDefault = hreflangLinks('/ja/learn/epistemic/expanded-uncertainty/').find((link) => link.hreflang === 'x-default');
  if (expandedDefault?.href !== absoluteUrl('/learn/epistemic/expanded-uncertainty/')) {
    errors.push(`expanded-uncertainty x-default is ${expandedDefault?.href ?? '(missing)'}`);
  }
  const homeDefault = hreflangLinks('/').find((link) => link.hreflang === 'x-default');
  if (homeDefault?.href !== `${SITE}/`) errors.push(`home x-default is ${homeDefault?.href}`);
  const esHome = canonicalPathname('/es/');
  if (esHome !== '/es/' || !isIndexableUrl('/es/')) errors.push(`expected /es/ to stay an indexable self-canonical home, got ${esHome}`);
  const esLanguage = canonicalPathname('/es/language/');
  if (esLanguage !== '/es/language/' || isIndexableUrl('/es/language/')) {
    errors.push(`expected /es/language/ to self-canonicalize and stay out of the index, got ${esLanguage}`);
  }
  errors.push(...assertNoindexSitemapFixture());
  return errors;
}

/**
 * A self-canonical noindex URL must not be accepted as a sitemap row, even
 * when its xhtml:link set is the English cluster the old checker expected.
 */
export function sitemapListingErrors(loc, page) {
  const errors = [];
  let pathname = '';
  try {
    pathname = new URL(loc).pathname;
  } catch {
    return [`sitemap loc is not a URL: ${loc}`];
  }
  if (!isIndexableUrl(pathname)) {
    errors.push(`sitemap lists ${loc} which is not an indexable translation`);
  }
  if (page?.head?.noindex) {
    errors.push(`sitemap lists ${loc} whose HTML is noindex`);
  }
  return errors;
}

function assertNoindexSitemapFixture() {
  const loc = absoluteUrl('/es/packages/');
  const englishCluster = hreflangLinks('/es/packages/');
  const accepted = sitemapListingErrors(loc, {
    head: {
      canonicals: [loc],
      noindex: true,
      alternates: englishCluster,
    },
  });
  const errors = [];
  if (accepted.length === 0) {
    errors.push('negative fixture: noindex /es/packages/ sitemap row was accepted');
  }
  if (!accepted.some((error) => error.includes('noindex'))) {
    errors.push('negative fixture: noindex /es/packages/ was not rejected for noindex');
  }
  if (!accepted.some((error) => error.includes('not an indexable translation'))) {
    errors.push('negative fixture: noindex /es/packages/ was not rejected as non-indexable');
  }
  const unmarked = sitemapListingErrors(loc, { head: { canonicals: [loc], noindex: false, alternates: englishCluster } });
  if (!unmarked.some((error) => error.includes('not an indexable translation'))) {
    errors.push('negative fixture: non-indexable /es/packages/ sitemap row was accepted');
  }
  const home = absoluteUrl('/');
  const homeErrors = sitemapListingErrors(home, { head: { canonicals: [home], noindex: false } });
  if (homeErrors.length > 0) {
    errors.push(`negative fixture: indexable home was rejected: ${homeErrors.join('; ')}`);
  }
  return errors;
}

function fileToUrl(relFile) {
  const posix = relFile.split(path.sep).join('/');
  if (posix === '404.html') return null;
  let route;
  if (posix === 'index.html') route = '/';
  else if (posix.endsWith('/index.html')) route = `/${posix.slice(0, -'/index.html'.length)}/`;
  else return null;
  return absoluteUrl(route);
}

async function* walk(dir) {
  const entries = await readdir(dir);
  for (const entry of entries) {
    if (entry === 'pagefind') continue;
    const full = path.join(dir, entry);
    const info = await stat(full);
    if (info.isDirectory()) yield* walk(full);
    else if (entry.endsWith('.html')) yield full;
  }
}

function parseHead(html) {
  const htmlLangMatch = /<html\b[^>]*\slang="([^"]+)"/.exec(html);
  const robots = /<meta\b[^>]*name="robots"[^>]*content="([^"]+)"/.exec(html);
  const refresh = /http-equiv="refresh"/i.test(html);
  const links = [];
  for (const tag of html.matchAll(/<link\b[^>]*>/g)) {
    const rel = /\brel="([^"]+)"/.exec(tag[0])?.[1];
    const href = /\bhref="([^"]+)"/.exec(tag[0])?.[1];
    const hreflang = /\bhreflang="([^"]+)"/.exec(tag[0])?.[1];
    if (rel && href) links.push({ rel, href, hreflang });
  }
  return {
    htmlLang: htmlLangMatch?.[1] ?? '',
    robots: robots?.[1] ?? '',
    noindex: Boolean(robots?.[1]?.includes('noindex')),
    refresh,
    canonicals: links.filter((link) => link.rel === 'canonical').map((link) => link.href),
    alternates: links.filter((link) => link.rel === 'alternate' && link.hreflang),
  };
}

function sameSet(actual, expected) {
  if (actual.length !== expected.length) return false;
  const left = [...actual].sort();
  const right = [...expected].sort();
  return left.every((value, index) => value === right[index]);
}

async function readSitemapLocs() {
  const indexPath = path.join(distDir, 'sitemap-index.xml');
  const index = await readFile(indexPath, 'utf8');
  const children = [...index.matchAll(/<loc>([^<]+)<\/loc>/g)].map((match) => match[1]);
  const docs = [];
  for (const child of children) {
    const name = new URL(child).pathname.split('/').pop();
    const xml = await readFile(path.join(distDir, name), 'utf8');
    const blocks = xml.split(/<url>/).slice(1);
    for (const block of blocks) {
      const loc = /<loc>([^<]+)<\/loc>/.exec(block)?.[1];
      const alternates = [...block.matchAll(/hreflang="([^"]+)"\s+href="([^"]+)"/g)].map((match) => ({
        hreflang: match[1],
        href: match[2],
      }));
      docs.push({ loc, alternates });
    }
  }
  return docs;
}

async function run() {
  const errors = assertExamples();
  const pages = new Map();

  for await (const file of walk(distDir)) {
    const rel = path.relative(distDir, file);
    const url = fileToUrl(rel);
    if (!url) continue;
    const html = await readFile(file, 'utf8');
    const head = parseHead(html);
    const { pathname } = new URL(url);
    if (head.refresh || isRedirectPath(pathname)) {
      if (head.alternates.length > 0) errors.push(`${rel}: redirect stub emits hreflang`);
      continue;
    }
    pages.set(url, { rel, head, pathname });
  }

  for (const [url, page] of pages) {
    const locale = localeFromPath(page.pathname);
    const expectedLang = htmlLang(locale);
    if (page.head.htmlLang !== expectedLang) {
      errors.push(`${page.rel}: html lang ${page.head.htmlLang || '(missing)'} != ${expectedLang}`);
    }
    if (page.head.canonicals.length !== 1) {
      errors.push(`${page.rel}: expected one canonical, found ${page.head.canonicals.length}`);
      continue;
    }
    const expectedCanonical = absoluteUrl(canonicalPathname(page.pathname));
    if (page.head.canonicals[0] !== expectedCanonical) {
      errors.push(`${page.rel}: canonical ${page.head.canonicals[0]} != ${expectedCanonical}`);
    }
    if (!page.head.canonicals[0].startsWith(`${SITE}/`) && page.head.canonicals[0] !== `${SITE}/`) {
      errors.push(`${page.rel}: canonical is not on ${SITE}`);
    }
    if (!page.head.canonicals[0].endsWith('/')) {
      errors.push(`${page.rel}: canonical is missing a trailing slash`);
    }

    if (!isIndexableUrl(page.pathname)) {
      if (!page.head.noindex) errors.push(`${page.rel}: untranslated page is missing noindex`);
      if (page.head.alternates.length > 0) errors.push(`${page.rel}: untranslated page emits hreflang`);
      continue;
    }
    if (page.head.noindex) {
      errors.push(`${page.rel}: indexable page is noindex`);
      if (page.head.alternates.length > 0) errors.push(`${page.rel}: noindex page emits hreflang`);
      continue;
    }

    const expected = hreflangLinks(page.pathname);
    const actualPairs = page.head.alternates.map((link) => `${link.hreflang} ${link.href}`);
    const expectedPairs = expected.map((link) => `${link.hreflang} ${link.href}`);
    if (!sameSet(actualPairs, expectedPairs)) {
      errors.push(`${page.rel}: hreflang set mismatch\n  actual: ${actualPairs.join(' | ')}\n  expected: ${expectedPairs.join(' | ')}`);
    }
    for (const link of page.head.alternates) {
      if (!ALLOWED_CODES.has(link.hreflang)) errors.push(`${page.rel}: invalid hreflang code ${link.hreflang}`);
      if (!link.href.startsWith(`${SITE}/`) || !link.href.endsWith('/')) {
        errors.push(`${page.rel}: hreflang href is not an absolute trailing-slash URL: ${link.href}`);
      }
      const target = pages.get(link.href);
      if (!target) {
        errors.push(`${page.rel}: hreflang ${link.hreflang} points at missing or redirect URL ${link.href}`);
        continue;
      }
      if (target.head.canonicals[0] !== link.href) {
        errors.push(`${page.rel}: hreflang ${link.hreflang} points at ${link.href} which canonicalizes to ${target.head.canonicals[0]}`);
      }
    }
    if (expectedCanonical !== url && page.head.alternates.some((link) => link.href === url)) {
      errors.push(`${page.rel}: fallback URL is listed in its own hreflang set`);
    }
  }

  let sitemapLocs = [];
  try {
    sitemapLocs = await readSitemapLocs();
  } catch (error) {
    errors.push(`sitemap read failed: ${error.message}`);
  }

  const localizedLocs = sitemapLocs.filter((entry) => localeFromPath(new URL(entry.loc).pathname) !== 'en');
  for (const entry of sitemapLocs) {
    const page = pages.get(entry.loc);
    if (!page) {
      errors.push(`sitemap lists ${entry.loc} which is not a self-canonical HTML page`);
      continue;
    }
    if (page.head.canonicals[0] !== entry.loc) {
      errors.push(`sitemap lists ${entry.loc} but its canonical is ${page.head.canonicals[0]}`);
    }
    errors.push(...sitemapListingErrors(entry.loc, page));
    if (isRedirectPath(new URL(entry.loc).pathname)) errors.push(`sitemap lists redirect ${entry.loc}`);
    const expected = hreflangLinks(new URL(entry.loc).pathname).map((link) => `${link.hreflang} ${link.href}`);
    const actual = entry.alternates.map((link) => `${link.hreflang} ${link.href}`);
    if (!sameSet(actual, expected)) {
      errors.push(`sitemap alternates for ${entry.loc} mismatch\n  actual: ${actual.join(' | ')}\n  expected: ${expected.join(' | ')}`);
    }
  }

  const effectsLoc = absoluteUrl('/learn/effects/');
  if (!sitemapLocs.some((entry) => entry.loc === effectsLoc)) {
    errors.push(`sitemap missing ${effectsLoc}`);
  }

  const ptExpanded = absoluteUrl('/pt/learn/epistemic/expanded-uncertainty/');
  const ptPage = pages.get(ptExpanded);
  if (!ptPage) {
    errors.push(`missing pt expanded-uncertainty stub ${ptExpanded}`);
  } else {
    if (ptPage.head.robots !== 'noindex, follow') {
      errors.push(`pt expanded-uncertainty robots ${ptPage.head.robots || '(missing)'} != noindex, follow`);
    }
    if (ptPage.head.alternates.length > 0) {
      errors.push('pt expanded-uncertainty stub emits hreflang');
    }
    if (ptPage.head.canonicals[0] !== ptExpanded) {
      errors.push(`pt expanded-uncertainty canonical ${ptPage.head.canonicals[0]} != ${ptExpanded}`);
    }
    if (!ptPage.head.noindex) {
      errors.push('pt expanded-uncertainty stub is indexable');
    }
  }
  if (sitemapLocs.some((entry) => entry.loc === ptExpanded)) {
    errors.push(`sitemap lists noindex pt expanded-uncertainty stub ${ptExpanded}`);
  }
  for (const [url, page] of pages) {
    if (!url.includes('/pt/') || page.head.noindex) continue;
    if (new URL(url).pathname.includes('/learn/epistemic/expanded-uncertainty/')) {
      errors.push(`${page.rel}: indexable dist/pt expanded-uncertainty page`);
    }
  }

  if (errors.length > 0) {
    console.error(`hreflang validation failed (${errors.length}):`);
    for (const error of errors) console.error(`- ${error}`);
    process.exit(1);
  }

  console.log(`OK: hreflang validated (${pages.size} pages, sitemap locs ${sitemapLocs.length}, localized sitemap locs ${localizedLocs.length}).`);
}

await run();
