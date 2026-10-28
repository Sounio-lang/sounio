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
  isRedirectPath,
  localeFromPath,
} from '../src/lib/seo-locales.mjs';

const root = path.resolve(process.cwd());
const distDir = path.join(root, 'dist');
const ALLOWED_CODES = new Set([...Object.values(HREFLANG), 'x-default']);

function assertExamples() {
  const errors = [];
  const packages = canonicalPathname('/es/packages/');
  if (packages !== '/packages/') errors.push(`expected /es/packages/ to canonicalize to /packages/, got ${packages}`);
  const effect = canonicalPathname('/zh/learn/compiler/effect-system/');
  if (effect !== '/zh/learn/compiler/effect-system/') {
    errors.push(`expected zh effect-system to self-canonicalize, got ${effect}`);
  }
  const units = canonicalPathname('/zh-hk/learn/language/units-and-uncertainty/');
  if (units !== '/learn/language/units-and-uncertainty/') {
    errors.push(`expected zh-hk units page to canonicalize to English, got ${units}`);
  }
  const effectCodes = hreflangLinks('/zh/learn/compiler/effect-system/').map((link) => link.hreflang);
  if (!effectCodes.includes('zh-CN') || !effectCodes.includes('x-default') || effectCodes.includes('zh')) {
    errors.push(`effect-system hreflang codes drifted: ${effectCodes.join(',')}`);
  }
  const packageCodes = hreflangLinks('/es/packages/').map((link) => link.hreflang);
  if (packageCodes.includes('es') || packageCodes.includes('pt-BR')) {
    errors.push(`packages hreflang still lists untranslated locales: ${packageCodes.join(',')}`);
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

    if (page.head.noindex) {
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
    if (isRedirectPath(new URL(entry.loc).pathname)) errors.push(`sitemap lists redirect ${entry.loc}`);
    const expected = hreflangLinks(new URL(entry.loc).pathname).map((link) => `${link.hreflang} ${link.href}`);
    const actual = entry.alternates.map((link) => `${link.hreflang} ${link.href}`);
    if (!sameSet(actual, expected)) {
      errors.push(`sitemap alternates for ${entry.loc} mismatch\n  actual: ${actual.join(' | ')}\n  expected: ${expected.join(' | ')}`);
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
