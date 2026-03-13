/**
 * External link checker — crawls built HTML for outbound URLs,
 * deduplicates, and HEAD-checks them with bounded concurrency.
 *
 * Usage:
 *   node scripts/check-external-links.mjs          # warn-only (default)
 *   node scripts/check-external-links.mjs --strict  # exit 1 on any broken link
 */
import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd());
const distDir = path.join(root, 'dist');
const strict = process.argv.includes('--strict');

const CONCURRENCY = 12;
const TIMEOUT_MS = 10_000;

const EXTERNAL_RE = /^(?:https?:)?\/\//i;
const HREF_RE = /\bhref=(["'])(.*?)\1/gi;
const SRC_RE = /\bsrc=(["'])(.*?)\1/gi;

// Domains known to block automated HEAD/GET, require auth, or are self-referential
const SKIP_DOMAINS = new Set([
  'fonts.googleapis.com',
  'fonts.gstatic.com',
  'localhost',
  '127.0.0.1',
  'example.com',
  'example.local',
  'www.souniolang.org',
  'souniolang.org',
]);

async function* walk(dir) {
  const entries = await readdir(dir);
  for (const entry of entries) {
    const abs = path.join(dir, entry);
    const info = await stat(abs);
    if (info.isDirectory()) {
      yield* walk(abs);
    } else {
      yield abs;
    }
  }
}

function extractExternalUrls(html) {
  const urls = new Set();
  for (const re of [HREF_RE, SRC_RE]) {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(html)) !== null) {
      const raw = m[2];
      if (EXTERNAL_RE.test(raw)) {
        // Normalize protocol-relative to https
        const url = raw.startsWith('//') ? `https:${raw}` : raw;
        try {
          const parsed = new URL(url);
          if (!SKIP_DOMAINS.has(parsed.hostname)) {
            urls.add(parsed.origin + parsed.pathname);
          }
        } catch {
          // malformed URL, skip
        }
      }
    }
  }
  return urls;
}

async function checkUrl(url) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    // Try HEAD first, fall back to GET if HEAD is rejected (405)
    let res = await fetch(url, {
      method: 'HEAD',
      signal: controller.signal,
      redirect: 'follow',
      headers: { 'User-Agent': 'SounioLinkChecker/1.0' },
    });
    if (res.status === 405) {
      res = await fetch(url, {
        method: 'GET',
        signal: controller.signal,
        redirect: 'follow',
        headers: { 'User-Agent': 'SounioLinkChecker/1.0' },
      });
    }
    return { url, status: res.status, ok: res.status < 400 };
  } catch (err) {
    return { url, status: 0, ok: false, error: err.message };
  } finally {
    clearTimeout(timer);
  }
}

async function runPool(urls, concurrency) {
  const results = [];
  const queue = [...urls];
  const workers = Array.from({ length: Math.min(concurrency, queue.length) }, async () => {
    while (queue.length > 0) {
      const url = queue.shift();
      results.push(await checkUrl(url));
    }
  });
  await Promise.all(workers);
  return results;
}

async function run() {
  // 1. Collect all external URLs from built HTML
  const urlToPages = new Map();

  for await (const abs of walk(distDir)) {
    if (!abs.endsWith('.html')) continue;
    const rel = path.relative(distDir, abs);
    const html = await readFile(abs, 'utf8');
    for (const url of extractExternalUrls(html)) {
      if (!urlToPages.has(url)) urlToPages.set(url, []);
      urlToPages.get(url).push(rel);
    }
  }

  const uniqueUrls = [...urlToPages.keys()].sort();
  console.log(`Checking ${uniqueUrls.length} unique external URLs...`);

  // 2. Check with concurrency pool
  const results = await runPool(uniqueUrls, CONCURRENCY);

  // 3. Report
  const broken = results.filter((r) => !r.ok);
  const ok = results.filter((r) => r.ok);

  if (broken.length > 0) {
    console.error(`\n${broken.length} broken external link(s):`);
    for (const r of broken) {
      const pages = urlToPages.get(r.url);
      const pageList = pages.length <= 3 ? pages.join(', ') : `${pages.slice(0, 3).join(', ')} (+${pages.length - 3} more)`;
      const detail = r.error ? r.error : `HTTP ${r.status}`;
      console.error(`  ${detail}: ${r.url}`);
      console.error(`    found in: ${pageList}`);
    }
  }

  console.log(`\n${ok.length} OK, ${broken.length} broken (${uniqueUrls.length} total).`);

  if (strict && broken.length > 0) {
    process.exit(1);
  }
}

await run();
