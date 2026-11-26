/**
 * Every sitemap URL has <lastmod> from a content date or a git commit day.
 * The build clock is not a lastmod.
 */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { isCalendarDate, lastmodForUrl, repositoryIsShallow, resolveLastmod } from '../src/lib/sitemap-lastmod.mjs';

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

export function sitemapLastmodErrors(xml, lookup, { allowMissing = false } = {}) {
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
    const expected = lookup(loc);
    if (!day) {
      if (allowMissing && !expected) continue;
      errors.push(`${loc}: lastmod ${JSON.stringify(lastmod)} is not a content or git day`);
      continue;
    }
    if (expected !== day) {
      errors.push(`${loc}: lastmod ${day}, source date ${expected ?? 'missing'}`);
    }
  }
  return errors;
}

/** Distinct calendar days in <lastmod>, and how many URLs share the most common one. */
export function lastmodConcentration(xml) {
  const blocks = [...xml.matchAll(/<url>([\s\S]*?)<\/url>/g)].map((match) => match[1]);
  const counts = new Map();
  for (const block of blocks) {
    const raw = block.match(/<lastmod>([^<]*)<\/lastmod>/)?.[1] ?? '';
    const day = lastmodDay(raw);
    if (!day) continue;
    counts.set(day, (counts.get(day) ?? 0) + 1);
  }
  let mode = 0;
  for (const count of counts.values()) if (count > mode) mode = count;
  return {
    urls: blocks.length,
    distinct: counts.size,
    mode,
    share: blocks.length === 0 ? 0 : mode / blocks.length,
  };
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
  const omitted = '<urlset><url><loc>https://www.souniolang.org/learn/</loc></url></urlset>';
  assert.equal(sitemapLastmodErrors(omitted, () => null, { allowMissing: true }).length, 0);
  assert.ok(sitemapLastmodErrors(omitted, () => null).some((error) => error.includes('/learn/')));
  const sameDay = Array.from({ length: 10 }, () => '<url><loc>https://www.souniolang.org/a/</loc><lastmod>2026-08-01</lastmod></url>').join('');
  const concentrated = lastmodConcentration(`<urlset>${sameDay}</urlset>`);
  assert.equal(concentrated.urls, 10);
  assert.equal(concentrated.distinct, 1);
  assert.ok(concentrated.share > 0.9);
  const mixed = lastmodConcentration('<urlset><url><loc>https://www.souniolang.org/a/</loc><lastmod>2026-08-01</lastmod></url><url><loc>https://www.souniolang.org/b/</loc><lastmod>2026-08-02T00:00:00.000Z</lastmod></url></urlset>');
  assert.equal(mixed.distinct, 2);
  assert.ok(mixed.share <= 0.9);
  console.log('sitemap lastmod self-test passed');
  process.exit(0);
}

const xml = await readFile(distSitemap, 'utf8').catch(() => '');
if (!xml) {
  console.error('dist/sitemap-0.xml is missing; run the website build first');
  process.exit(1);
}

const shallow = repositoryIsShallow();
const errors = sitemapLastmodErrors(xml, lastmodForUrl, { allowMissing: shallow });
if (errors.length > 0) {
  console.error(`sitemap lastmod failed (${errors.length}):`);
  for (const error of errors.slice(0, 30)) console.error(`- ${error}`);
  if (errors.length > 30) console.error(`- … ${errors.length - 30} more`);
  process.exit(1);
}

const concentration = lastmodConcentration(xml);
console.log(`sitemap lastmod: ${concentration.urls} urls, ${concentration.distinct} distinct values, from content or git dates`);
// A shallow checkout has already refused to invent a deploy-day lastmod.
// A full history that still stamps one day on more than 90% of URLs is the production bug.
if (!shallow && concentration.urls > 0 && concentration.share > 0.9) {
  console.error(`sitemap lastmod failed: ${concentration.mode} of ${concentration.urls} urls share one lastmod (${concentration.distinct} distinct values)`);
  process.exit(1);
}
