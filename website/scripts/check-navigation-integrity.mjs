import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd());
const distDir = path.join(root, 'dist');

const ASSET_EXT_RE = /\.(?:png|jpe?g|webp|avif|gif|svg|ico|css|js|mjs|map|txt|xml|json|pdf|woff2?|ttf|otf|webmanifest)$/i;
const HASH_HREF_RE = /^#(.+)$/;
const EXTERNAL_RE = /^(?:[a-z][a-z0-9+.-]*:)?\/\//i;
const SCHEME_RE = /^(?:mailto:|tel:|javascript:|data:)/i;

function normalizeRoute(p) {
  if (!p) return '/';
  const noQuery = p.split('?')[0]?.split('#')[0] || '/';
  if (noQuery === '/') return '/';
  return noQuery.replace(/\/+$/, '');
}

function fileToRoute(relFile) {
  const posix = relFile.split(path.sep).join('/');

  if (posix === 'index.html') return '/';
  if (posix.endsWith('/index.html')) {
    const base = posix.slice(0, -'/index.html'.length);
    return `/${base}`;
  }
  if (posix.endsWith('.html')) {
    return `/${posix.slice(0, -'.html'.length)}`;
  }
  return null;
}

function routeToFile(route) {
  const normalized = normalizeRoute(route);
  if (normalized === '/') return 'index.html';
  return `${normalized.slice(1)}/index.html`;
}

function looksLikeAssetPath(urlPath) {
  return (
    ASSET_EXT_RE.test(urlPath) ||
    urlPath.startsWith('/_astro/') ||
    urlPath.startsWith('/assets/') ||
    urlPath.startsWith('/pagefind/')
  );
}

function extractHrefs(html) {
  const hrefs = [];
  const re = /\bhref=(["'])(.*?)\1/gi;
  let match;
  while ((match = re.exec(html)) !== null) {
    hrefs.push(match[2]);
  }
  return hrefs;
}

async function* walk(dir) {
  const entries = await readdir(dir);
  for (const entry of entries) {
    const abs = path.join(dir, entry);
    const info = await stat(abs);
    if (info.isDirectory()) {
      yield* walk(abs);
      continue;
    }
    yield abs;
  }
}

function resolveHrefToPath(href, currentRoute) {
  const base = new URL(
    currentRoute === '/' ? 'https://example.local/' : `https://example.local${currentRoute}/`
  );
  const resolved = new URL(href, base);
  return resolved.pathname;
}

function idExistsInHtml(html, id) {
  const escaped = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`\\sid=(["'])${escaped}\\1`);
  return re.test(html);
}

async function run() {
  const routeSet = new Set();
  const htmlFiles = [];

  for await (const abs of walk(distDir)) {
    if (!abs.endsWith('.html')) continue;
    const rel = path.relative(distDir, abs);
    const route = fileToRoute(rel);
    if (route) routeSet.add(route);
    htmlFiles.push(abs);
  }

  const errors = [];
  let checkedLinks = 0;

  for (const abs of htmlFiles) {
    const rel = path.relative(distDir, abs);
    const currentRoute = fileToRoute(rel);
    if (!currentRoute) continue;

    // Skip link checking on 404 page — it's a hosting-level page with locale
    // switcher links that legitimately point to non-existent localized 404 routes.
    if (rel === '404.html') continue;

    const html = await readFile(abs, 'utf8');
    const hrefs = extractHrefs(html);

    for (const href of hrefs) {
      if (!href || href === '#') {
        errors.push(`Empty hash href in ${rel}`);
        continue;
      }

      if (EXTERNAL_RE.test(href) || SCHEME_RE.test(href)) {
        continue;
      }

      const hashOnly = href.match(HASH_HREF_RE);
      if (hashOnly) {
        const targetId = decodeURIComponent(hashOnly[1]);
        if (!idExistsInHtml(html, targetId)) {
          errors.push(`Broken in-page hash link '${href}' in ${rel}`);
        }
        checkedLinks += 1;
        continue;
      }

      const urlPath = href.startsWith('/') ? href.split('?')[0].split('#')[0] : resolveHrefToPath(href, currentRoute);
      if (looksLikeAssetPath(urlPath)) continue;

      const normalized = normalizeRoute(urlPath);
      if (!routeSet.has(normalized)) {
        const targetFile = routeToFile(normalized);
        errors.push(`Broken internal link '${href}' in ${rel} (missing ${targetFile})`);
      }
      checkedLinks += 1;
    }
  }

  if (errors.length > 0) {
    console.error('Navigation integrity validation failed:');
    for (const err of errors.slice(0, 120)) {
      console.error(`- ${err}`);
    }
    if (errors.length > 120) {
      console.error(`- ...and ${errors.length - 120} more`);
    }
    process.exit(1);
  }

  console.log(`OK: navigation integrity validated (${checkedLinks} internal links).`);
}

await run();
