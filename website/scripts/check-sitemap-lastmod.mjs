/**
 * Every sitemap URL has <lastmod> from a content date or a git commit day.
 * The build clock is not a lastmod.
 */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { isCalendarDate, lastmodForUrl, resolveLastmod } from '../src/lib/sitemap-lastmod.mjs';

const websiteRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const distSitemap = path.join(websiteRoot, 'dist/sitemap-0.xml');
const sourcePath = path.join(websiteRoot, 'src/lib/sitemap-lastmod.mjs');

function selfTest() {
  assert.equal(resolveLastmod({ contentDate: 'unvalidated', gitDate: '2026-08-01' }), '2026-08-01');
  assert.equal(resolveLastmod({ contentDate: '2026-11-06', gitDate: '2026-08-01' }), '2026-11-06');
  assert.equal(resolveLastmod({ contentDate: '2026-02-31', gitDate: '2026-08-01' }), '2026-08-01');
  assert.equal(resolveLastmod({ contentDate: '2026-08-01', gitDate: '2026-11-06' }), '2026-11-06');
  assert.equal(resolveLastmod({ contentDate: null, gitDate: null }), null);
  assert.equal(isCalendarDate('2024-02-29'), true);
  assert.equal(isCalendarDate('2026-02-29'), false);
  assert.equal(isCalendarDate('2025-02-29'), false);
  assert.equal(isCalendarDate('2026-11-20T13:00:00Z'), false);
}

/** The sitemap integration may write a calendar day as UTC midnight. The day is the source date. */
export function lastmodDay(value) {
  if (isCalendarDate(value)) return value;
  const midnight = /^(\d{4}-\d{2}-\d{2})T00:00:00(?:\.000)?Z$/.exec(value ?? '');
  if (midnight && isCalendarDate(midnight[1])) return midnight[1];
  return null;
}

export function sitemapLastmodErrors(xml, lookup) {
  const errors = [];
  const blocks = [...xml.matchAll(/<url>([\s\S]*?)<\/url>/g)].map((match) => match[1]);
  if (blocks.length === 0) errors.push('sitemap has no <url>');
  for (const block of blocks) {
    const loc = block.match(/<loc>([^<]*)<\/loc>/)?.[1] ?? '';
    const lastmod = block.match(/<lastmod>([^<]*)<\/lastmod>/)?.[1] ?? '';
    if (!loc) {
      errors.push('sitemap url without loc');
      continue;
    }
    const day = lastmodDay(lastmod);
    if (!day) {
      errors.push(`${loc}: lastmod ${JSON.stringify(lastmod)} is not a content or git day`);
      continue;
    }
    const expected = lookup(loc);
    if (expected !== day) {
      errors.push(`${loc}: lastmod ${day}, source date ${expected ?? 'missing'}`);
    }
  }
  return errors;
}

selfTest();

const source = await readFile(sourcePath, 'utf8');
if (/new Date|Date\.now/.test(source)) {
  console.error('sitemap-lastmod.mjs must not read the build clock');
  process.exit(1);
}

if (process.argv.includes('--self-test')) {
  const fixture = `<?xml version="1.0"?><urlset>
    <url><loc>https://www.souniolang.org/</loc><lastmod>2026-08-01</lastmod></url>
    <url><loc>https://www.souniolang.org/learn/</loc></url>
  </urlset>`;
  const lookup = (loc) => (loc.endsWith('/') && !loc.endsWith('/learn/') ? '2026-08-01' : '2026-11-06');
  const errors = sitemapLastmodErrors(fixture, lookup);
  assert.ok(errors.some((error) => error.includes('/learn/')));
  assert.equal(lastmodDay('2026-08-01T00:00:00.000Z'), '2026-08-01');
  assert.equal(lastmodDay('2026-08-01T13:18:41.452Z'), null);
  assert.equal(sitemapLastmodErrors(
    '<urlset><url><loc>https://www.souniolang.org/</loc><lastmod>2026-08-01</lastmod></url></urlset>',
    () => '2026-08-01',
  ).length, 0);
  console.log('sitemap lastmod self-test passed');
  process.exit(0);
}

const xml = await readFile(distSitemap, 'utf8').catch(() => '');
if (!xml) {
  console.error('dist/sitemap-0.xml is missing; run the website build first');
  process.exit(1);
}

const errors = sitemapLastmodErrors(xml, lastmodForUrl);
if (errors.length > 0) {
  console.error(`sitemap lastmod failed (${errors.length}):`);
  for (const error of errors.slice(0, 30)) console.error(`- ${error}`);
  if (errors.length > 30) console.error(`- … ${errors.length - 30} more`);
  process.exit(1);
}

const count = xml.match(/<lastmod>/g)?.length ?? 0;
console.log(`sitemap lastmod: ${count} urls, from content or git dates`);
