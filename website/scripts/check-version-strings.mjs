/**
 * Public status is exactly "Sounio 3.0 (pre-release)".
 * Built HTML must not carry another release token: pre-1.0, v1.0.0,
 * 3.0.0-rc.1, 2.1.0, 1.0.0-beta.6, 0.100.0, and the same shape.
 *
 * The Astro generator meta is the framework stamp, not a Sounio release.
 * Script, style, and svg bodies are not prose. Handbook section numbers
 * such as 7.1.26 are not release tokens (the major is outside 0–3).
 */
import assert from 'node:assert/strict';
import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';

export const STATUS_LABEL = 'Sounio 3.0 (pre-release)';

const RELEASE_TOKEN = /\b(?:pre-1\.0|v\d+(?:\.\d+){1,2}(?:[-+][0-9A-Za-z.]+)?|(?:0|1|2|3)\.\d+\.\d+(?:[-+][0-9A-Za-z.]+)?)\b/g;

const websiteRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const distDir = path.join(websiteRoot, 'dist');

export function proseForVersionScan(html) {
  return html
    .replace(/<meta\b[^>]*\bname="generator"[^>]*>/gi, '')
    .replace(/<script\b(?![^>]*type="application\/ld\+json")[^>]*>[\s\S]*?<\/script>/gi, '')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, '')
    .replace(/<svg\b[^>]*>[\s\S]*?<\/svg>/gi, '');
}

export function releaseTokens(html) {
  const prose = proseForVersionScan(html).split(STATUS_LABEL).join(' ');
  return [...prose.matchAll(RELEASE_TOKEN)].map((match) => match[0]);
}

function selfTest() {
  assert.deepEqual(releaseTokens(`<p>${STATUS_LABEL}</p>`), []);
  assert.deepEqual(releaseTokens('<title>Language Spec</title>'), []);
  assert.deepEqual(releaseTokens('<p>Abramowitz &amp; Stegun (7.1.26)</p>'), []);
  assert.deepEqual(releaseTokens('<svg><path d="M5.092.682 9.504.5"/></svg>'), []);
  assert.deepEqual(
    releaseTokens('<meta name="generator" content="Astro v5.16.15">'),
    [],
  );
  assert.deepEqual(releaseTokens('<p>Language Spec (v1.0.0)</p>'), ['v1.0.0']);
  assert.deepEqual(releaseTokens('<p>research language (pre-1.0)</p>'), ['pre-1.0']);
  assert.deepEqual(releaseTokens('<p>3.0.0-rc.1 (madaros)</p>'), ['3.0.0-rc.1']);
  assert.deepEqual(releaseTokens('<p>aligned on 2.1.0</p>'), ['2.1.0']);
  assert.deepEqual(releaseTokens('<p>1.0.0-beta.6</p>'), ['1.0.0-beta.6']);
  assert.deepEqual(releaseTokens('<p>Madaros v0.80.0</p>'), ['v0.80.0']);
  assert.deepEqual(
    releaseTokens('<script type="application/ld+json">{"softwareVersion":"2.1.0"}</script>'),
    ['2.1.0'],
  );
  assert.deepEqual(releaseTokens('<script>const v = "2.1.0"</script><p>ok</p>'), []);
  assert.deepEqual(releaseTokens(`<p>${STATUS_LABEL} and also 0.100.0</p>`), ['0.100.0']);
}

async function walkHtml(dir, out = []) {
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await walkHtml(full, out);
    else if (entry.name.endsWith('.html')) out.push(full);
  }
  return out;
}

async function scanDist() {
  const files = await walkHtml(distDir);
  const errors = [];
  let labelled = 0;
  for (const file of files) {
    const html = await readFile(file, 'utf8');
    if (html.includes(STATUS_LABEL)) labelled += 1;
    const hits = releaseTokens(html);
    if (hits.length > 0) {
      const rel = path.relative(distDir, file);
      errors.push(`${rel}: ${[...new Set(hits)].join(', ')}`);
    }
  }
  if (files.length === 0) errors.push('dist has no HTML');
  if (labelled === 0) errors.push(`no built page contains ${JSON.stringify(STATUS_LABEL)}`);
  return errors;
}

selfTest();

if (process.argv.includes('--self-test')) {
  console.log('version-string self-test passed');
  process.exit(0);
}

const distErrors = await scanDist().catch((error) => {
  if (error && error.code === 'ENOENT') return ['dist/ is missing; run the website build first'];
  throw error;
});

if (distErrors.length > 0) {
  console.error(`version-string check failed (${distErrors.length}):`);
  for (const error of distErrors.slice(0, 40)) console.error(`- ${error}`);
  if (distErrors.length > 40) console.error(`- … ${distErrors.length - 40} more`);
  process.exit(1);
}

console.log(`version strings: only ${JSON.stringify(STATUS_LABEL)}`);
