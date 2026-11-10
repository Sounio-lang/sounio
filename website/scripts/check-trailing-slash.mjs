/**
 * Canonical trailing-slash contract.
 *
 * Astro `trailingSlash: 'always'` makes built directory URLs, canonicals and
 * the dev/preview server use a trailing slash. Production routing lives in
 * vercel.json `routes`, in this order:
 *   1. /docs and /:lang/docs (with or without a slash, one or two slug
 *      segments, no file extension) 308 to /learn/ or /:lang/learn/.
 *   2. /sitemap.xml (optional trailing slash) 308 to /sitemap-index.xml.
 *      The built file stays sitemap-index.xml; robots.txt is unchanged.
 *   3. The same three routes `@vercel/routing-utils` `convertTrailingSlash(true)`
 *      emits: pass `/.well-known`, 308-add a slash on extensionless paths,
 *      308-strip a slash on paths whose last segment has an extension.
 *
 * `trailingSlash: true` is intentionally absent. Vercel installs that flag's
 * routes before user `routes` and `redirects`, so /docs would 308 to /docs/
 * instead of /learn/.
 *
 * The HTML scan fails when a built internal href, canonical, hreflang or
 * og:url points at an extensionless path without a trailing slash.
 *
 *   node scripts/check-trailing-slash.mjs
 *   node scripts/check-trailing-slash.mjs --serve [port]
 */
import { createReadStream } from 'node:fs';
import { readdir, readFile, stat } from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';

const root = path.resolve(process.cwd());
const distDir = path.join(root, 'dist');
const SITE_HOSTS = new Set(['www.souniolang.org', 'souniolang.org']);
const SKIP_PREFIXES = ['/_astro/', '/pagefind/'];

const ACCEPTANCE = [
  ['/about', 308, '/about/'],
  ['/pt/proof', 308, '/pt/proof/'],
  ['/ja/learn/language/pattern-matching', 308, '/ja/learn/language/pattern-matching/'],
  ['/docs', 308, '/learn/'],
  ['/docs/', 308, '/learn/'],
  ['/pt/docs', 308, '/pt/learn/'],
  ['/pt/docs/', 308, '/pt/learn/'],
  ['/el/docs/', 308, '/el/learn/'],
  ['/zh/docs/', 308, '/zh/learn/'],
  ['/zh-hk/docs/', 308, '/zh-hk/learn/'],
  ['/ja/docs/', 308, '/ja/learn/'],
  ['/es/docs/', 308, '/es/learn/'],
  ['/docs/compiler/effect-system', 308, '/learn/compiler/effect-system/'],
  ['/docs/compiler/effect-system/', 308, '/learn/compiler/effect-system/'],
  ['/about/', null, null],
  ['/learn/', null, null],
  ['/robots.txt', null, null],
  ['/sitemap.xml', 308, '/sitemap-index.xml'],
  ['/sitemap.xml/', 308, '/sitemap-index.xml'],
  ['/sitemap-index.xml', null, null],
  ['/feed.xml', null, null],
  ['/llms.txt', null, null],
  ['/favicon.ico', null, null],
  ['/manifest.webmanifest', null, null],
  ['/docs/assets/vancomycin-ship/check_pass.png', null, null],
  ['/feed.xml/', 308, '/feed.xml'],
  ['/robots.txt/', 308, '/robots.txt'],
  ['/.well-known/acme-challenge/token', null, null],
];

export function applyRoutes(pathname, routes) {
  for (const route of routes) {
    const match = new RegExp(route.src).exec(pathname);
    if (!match) continue;
    const locationTemplate = route.headers?.Location;
    if (!locationTemplate) return null;
    let location = locationTemplate;
    for (let index = match.length - 1; index >= 1; index -= 1) {
      location = location.split(`$${index}`).join(match[index] ?? '');
    }
    return { status: route.status ?? 308, location };
  }
  return null;
}

function selfTestRouter() {
  const errors = [];
  const routes = [
    { src: '^/docs/?$', headers: { Location: '/learn/' }, status: 308 },
    { src: '^/((?:[^/]+/)*[^/\\.]+)$', headers: { Location: '/$1/' }, status: 308 },
  ];
  const direct = applyRoutes('/docs', routes);
  if (direct?.location !== '/learn/') errors.push('self-test: /docs must win over the trailing-slash route');
  const about = applyRoutes('/about', routes.slice(1));
  if (about?.location !== '/about/') errors.push(`self-test: /about location ${about?.location}`);
  const file = applyRoutes('/robots.txt', routes);
  if (file) errors.push('self-test: /robots.txt must not redirect');
  return errors;
}

function attr(tag, name) {
  const match = new RegExp(`\\b${name}=(["'])(.*?)\\1`, 'i').exec(tag);
  return match?.[2] ?? '';
}

export function pathnameOf(href, pagePath) {
  if (!href) return null;
  if (href.startsWith('#') || /^(?:mailto:|tel:|javascript:|data:)/i.test(href)) return null;
  let url;
  try {
    const basePath = pagePath.endsWith('/') ? pagePath : `${pagePath}/`;
    url = new URL(href, `https://www.souniolang.org${basePath}`);
  } catch {
    return null;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
  if (!SITE_HOSTS.has(url.hostname)) return null;
  return url.pathname;
}

export function slashProblem(pathname) {
  if (!pathname || pathname === '/') return null;
  if (SKIP_PREFIXES.some((prefix) => pathname.startsWith(prefix))) return null;
  const last = pathname.split('/').filter(Boolean).pop() ?? '';
  if (last.includes('.')) return null;
  if (pathname.endsWith('/')) return null;
  return pathname;
}

export function hrefProblems(html, pagePath) {
  const problems = [];
  const hrefs = [];
  for (const match of html.matchAll(/\bhref=(["'])(.*?)\1/gi)) hrefs.push(match[2]);
  for (const href of hrefs) {
    const pathname = pathnameOf(href, pagePath);
    const problem = slashProblem(pathname);
    if (problem) problems.push(`${href} → ${problem}`);
  }
  for (const tag of html.matchAll(/<link\b[^>]*>/gi)) {
    const rel = attr(tag[0], 'rel');
    const href = attr(tag[0], 'href');
    if (rel !== 'canonical' && !(rel === 'alternate' && attr(tag[0], 'hreflang'))) continue;
    const problem = slashProblem(pathnameOf(href, pagePath));
    if (problem) problems.push(`${rel} ${href} → ${problem}`);
  }
  for (const tag of html.matchAll(/<meta\b[^>]*>/gi)) {
    if (attr(tag[0], 'property') !== 'og:url') continue;
    const content = attr(tag[0], 'content');
    const problem = slashProblem(pathnameOf(content, pagePath));
    if (problem) problems.push(`og:url ${content} → ${problem}`);
  }
  const refresh = /http-equiv="refresh"[^>]*content="[^"]*?url=([^";\s]+)/i.exec(html)
    ?? /content="[^"]*?url=([^";\s]+)"[^>]*http-equiv="refresh"/i.exec(html);
  if (refresh) {
    const problem = slashProblem(pathnameOf(refresh[1], pagePath));
    if (problem) problems.push(`refresh ${refresh[1]} → ${problem}`);
  }
  return [...new Set(problems)];
}

function selfTestHrefs() {
  const errors = [];
  const bad = hrefProblems('<a href="/about">', '/');
  if (bad.length !== 1) errors.push(`self-test: /about href was not rejected (${bad.join('; ')})`);
  const docs = hrefProblems('<a href="/docs">', '/');
  if (docs.length !== 1) errors.push('self-test: /docs href was not rejected');
  const good = hrefProblems(
    '<a href="/about/"><a href="/feed.xml"><a href="/robots.txt"><a href="/sitemap-index.xml"><a href="/llms.txt"><a href="/docs/assets/vancomycin-ship/check_pass.png"><a href="https://github.com/sounio-lang/sounio/blob/main/docs/guide/x.md"><a href="#x"><link rel="canonical" href="https://www.souniolang.org/about/"><link rel="alternate" hreflang="en" href="https://www.souniolang.org/learn/">',
    '/about/',
  );
  if (good.length) errors.push(`self-test: clean hrefs were rejected: ${good.join('; ')}`);
  const canonical = hrefProblems('<link rel="canonical" href="https://www.souniolang.org/learn">', '/learn/');
  if (canonical.length === 0) errors.push('self-test: slashless canonical was not rejected');
  return errors;
}

function fileToPage(relFile) {
  const posix = relFile.split(path.sep).join('/');
  if (posix === 'index.html') return '/';
  if (posix.endsWith('/index.html')) return `/${posix.slice(0, -'index.html'.length)}`;
  if (posix.endsWith('.html')) return `/${posix.slice(0, -'.html'.length)}`;
  return null;
}

async function* walk(dir) {
  const entries = await readdir(dir);
  for (const entry of entries) {
    if (entry === 'pagefind') continue;
    const full = path.join(dir, entry);
    const info = await stat(full);
    if (info.isDirectory()) yield* walk(full);
    else yield full;
  }
}

async function readConfig() {
  const errors = [];
  const astro = await readFile(path.join(root, 'astro.config.mjs'), 'utf8');
  if (!astro.includes("trailingSlash: 'always'")) {
    errors.push("astro.config.mjs must set trailingSlash: 'always'");
  }
  const robots = await readFile(path.join(root, 'public', 'robots.txt'), 'utf8');
  if (!robots.includes('Sitemap: https://www.souniolang.org/sitemap-index.xml')) {
    errors.push('robots.txt must keep Sitemap pointed at /sitemap-index.xml');
  }
  if (/^Sitemap:.*\/sitemap\.xml\s*$/m.test(robots)) {
    errors.push('robots.txt must not advertise /sitemap.xml; that path 308s to /sitemap-index.xml');
  }
  const vercel = JSON.parse(await readFile(path.join(root, 'vercel.json'), 'utf8'));
  if (vercel.trailingSlash === true) {
    errors.push('vercel.json trailingSlash: true runs before routes and would 308 /docs to /docs/ instead of /learn/');
  }
  if (!Array.isArray(vercel.routes) || vercel.routes.length === 0) {
    errors.push('vercel.json routes are missing');
    return { errors, routes: [] };
  }
  for (const [pathname, status, location] of ACCEPTANCE) {
    const hit = applyRoutes(pathname, vercel.routes);
    if (status === null) {
      if (hit) errors.push(`${pathname} redirected to ${hit.location}; expected no redirect`);
      continue;
    }
    if (!hit || hit.status !== status || hit.location !== location) {
      errors.push(`${pathname} → ${hit ? `${hit.status} ${hit.location}` : 'no redirect'}; expected ${status} ${location}`);
    }
  }
  return { errors, routes: vercel.routes };
}

async function scanDist() {
  const errors = [];
  let pages = 0;
  let hrefs = 0;
  try {
    await stat(distDir);
  } catch {
    return { errors: ['dist/ is missing; run the website build first'], pages, hrefs };
  }
  for await (const file of walk(distDir)) {
    if (!file.endsWith('.html')) continue;
    const rel = path.relative(distDir, file);
    const posix = rel.split(path.sep).join('/');
    if (/(^|\/)docs(\/|\.html$)/.test(posix)) {
      errors.push(`meta-refresh docs page still built: ${posix}`);
      continue;
    }
    const pagePath = fileToPage(rel);
    if (!pagePath) continue;
    pages += 1;
    const html = await readFile(file, 'utf8');
    hrefs += [...html.matchAll(/\bhref=/gi)].length;
    for (const problem of hrefProblems(html, pagePath)) {
      errors.push(`${posix}: ${problem}`);
    }
  }
  const indexPath = path.join(distDir, 'sitemap-index.xml');
  try {
    const index = await readFile(indexPath, 'utf8');
    const children = [...index.matchAll(/<loc>([^<]+)<\/loc>/g)].map((match) => match[1]);
    for (const child of children) {
      const name = new URL(child).pathname.split('/').pop();
      const xml = await readFile(path.join(distDir, name), 'utf8');
      for (const loc of xml.matchAll(/<loc>([^<]+)<\/loc>/g)) {
        const problem = slashProblem(new URL(loc[1]).pathname);
        if (problem) errors.push(`sitemap loc ${loc[1]}`);
      }
      for (const href of xml.matchAll(/\bhref="([^"]+)"/g)) {
        const problem = slashProblem(new URL(href[1]).pathname);
        if (problem) errors.push(`sitemap alternate ${href[1]}`);
      }
    }
  } catch (error) {
    errors.push(`sitemap read failed: ${error.message}`);
  }
  return { errors, pages, hrefs };
}

function contentType(file) {
  const ext = path.extname(file).toLowerCase();
  const types = {
    '.html': 'text/html; charset=utf-8',
    '.xml': 'application/xml',
    '.txt': 'text/plain; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
    '.ico': 'image/x-icon',
    '.css': 'text/css',
    '.js': 'text/javascript',
    '.json': 'application/json',
    '.wasm': 'application/wasm',
    '.webmanifest': 'application/manifest+json',
  };
  return types[ext] ?? 'application/octet-stream';
}

async function fileFor(pathname) {
  const rel = decodeURIComponent(pathname);
  if (rel.includes('\0') || rel.split('/').includes('..')) return null;
  const parts = rel.split('/').filter(Boolean);
  if (pathname.endsWith('/') || pathname === '/') {
    const index = path.join(distDir, ...parts, 'index.html');
    try {
      if ((await stat(index)).isFile()) return index;
    } catch { /* missing */ }
    return null;
  }
  const direct = path.join(distDir, ...parts);
  try {
    if ((await stat(direct)).isFile()) return direct;
  } catch { /* missing */ }
  return null;
}

function listen(routes, port) {
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url ?? '/', 'http://127.0.0.1');
    const hit = applyRoutes(url.pathname, routes);
    if (hit) {
      res.writeHead(hit.status, { Location: hit.location, 'Cache-Control': 'no-store' });
      res.end();
      return;
    }
    const file = await fileFor(url.pathname);
    if (!file) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('not found');
      return;
    }
    res.writeHead(200, { 'Content-Type': contentType(file) });
    if (req.method === 'HEAD') {
      res.end();
      return;
    }
    createReadStream(file).pipe(res);
  });
  server.listen(port, '127.0.0.1', () => {
    console.log(`slash preview http://127.0.0.1:${port}/`);
  });
}

const errors = [...selfTestRouter(), ...selfTestHrefs()];
const config = await readConfig();
errors.push(...config.errors);
const dist = await scanDist();
errors.push(...dist.errors);

if (errors.length > 0) {
  console.error(`Trailing-slash validation failed (${errors.length}):`);
  for (const error of errors.slice(0, 80)) console.error(`- ${error}`);
  if (errors.length > 80) console.error(`- ...and ${errors.length - 80} more`);
  process.exit(1);
}

console.log(`OK: trailing slash validated (${dist.pages} pages, ${dist.hrefs} hrefs, ${ACCEPTANCE.length} route cases).`);

const serve = process.argv.indexOf('--serve');
if (serve !== -1) {
  const port = Number(process.argv[serve + 1] ?? 8787);
  listen(config.routes, port);
}
