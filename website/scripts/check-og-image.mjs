/**
 * Open Graph and Twitter cards use a PNG, 1200×630.
 * SVG is not a card image for X or for most scrapers.
 */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

export const OG_PATH = '/assets/og/sounio-1200x630.png';
export const OG_WIDTH = 1200;
export const OG_HEIGHT = 630;

const KEY_PAGES = [
  '/index.html',
  '/learn/index.html',
  '/learn/epistemic/index.html',
  '/learn/getting-started/index.html',
  '/learn/epistemic/uncertainty-tools-compared/index.html',
  '/learn/epistemic/sounio-vs-mp-units-vs-physure/index.html',
];

const websiteRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const distDir = path.join(websiteRoot, 'dist');
const pngPath = path.join(websiteRoot, 'public', OG_PATH);

export function pngSize(buffer) {
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (buffer.length < 24 || !buffer.subarray(0, 8).equals(sig)) return null;
  if (buffer.subarray(12, 16).toString('ascii') !== 'IHDR') return null;
  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
}

function meta(html, key) {
  const tags = [...html.matchAll(/<meta\b[^>]*>/gi)].map((match) => match[0]);
  for (const tag of tags) {
    const prop = tag.match(/\b(?:property|name)="([^"]*)"/)?.[1];
    if (prop !== key) continue;
    return tag.match(/\bcontent="([^"]*)"/)?.[1] ?? '';
  }
  return '';
}

export function ogErrors(html, page) {
  const errors = [];
  const image = meta(html, 'og:image');
  const twitter = meta(html, 'twitter:image');
  if (!image.endsWith(OG_PATH)) errors.push(`${page}: og:image is ${image || 'missing'}`);
  if (image.endsWith('.svg') || twitter.endsWith('.svg')) errors.push(`${page}: card image is SVG`);
  if (meta(html, 'og:image:type') !== 'image/png') errors.push(`${page}: og:image:type`);
  if (meta(html, 'og:image:width') !== String(OG_WIDTH)) errors.push(`${page}: og:image:width`);
  if (meta(html, 'og:image:height') !== String(OG_HEIGHT)) errors.push(`${page}: og:image:height`);
  if (!meta(html, 'og:image:alt')) errors.push(`${page}: og:image:alt`);
  if (meta(html, 'twitter:card') !== 'summary_large_image') errors.push(`${page}: twitter:card`);
  if (twitter !== image) errors.push(`${page}: twitter:image does not match og:image`);
  if (!meta(html, 'twitter:image:alt')) errors.push(`${page}: twitter:image:alt`);
  return errors;
}

function selfTest() {
  const good = `<head>
    <meta property="og:image" content="https://www.souniolang.org${OG_PATH}" />
    <meta property="og:image:type" content="image/png" />
    <meta property="og:image:width" content="1200" />
    <meta property="og:image:height" content="630" />
    <meta property="og:image:alt" content="Sounio: units, effects, and uncertainty types" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:image" content="https://www.souniolang.org${OG_PATH}" />
    <meta name="twitter:image:alt" content="Sounio: units, effects, and uncertainty types" />
  </head>`;
  assert.deepEqual(ogErrors(good, '/'), []);
  const svg = good.replaceAll(OG_PATH, '/assets/lockups/lockup_horizontal_light.svg');
  assert.ok(ogErrors(svg, '/').some((error) => error.includes('SVG') || error.includes('og:image is')));
  assert.equal(pngSize(Buffer.from('not a png')), null);
  const header = Buffer.alloc(24);
  header.set(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
  header.write('IHDR', 12, 'ascii');
  header.writeUInt32BE(1200, 16);
  header.writeUInt32BE(630, 20);
  assert.deepEqual(pngSize(header), { width: 1200, height: 630 });
}

selfTest();

if (process.argv.includes('--self-test')) {
  console.log('og-image self-test passed');
  process.exit(0);
}

const errors = [];
const png = await readFile(pngPath).catch(() => null);
const size = png ? pngSize(png) : null;
if (!size || size.width !== OG_WIDTH || size.height !== OG_HEIGHT) {
  errors.push(`${OG_PATH} is ${size ? `${size.width}x${size.height}` : 'missing or not a PNG'}`);
}

for (const page of KEY_PAGES) {
  const html = await readFile(path.join(distDir, page), 'utf8').catch(() => '');
  if (!html) {
    errors.push(`${page}: missing from dist`);
    continue;
  }
  errors.push(...ogErrors(html, page));
}

if (errors.length > 0) {
  console.error(`og-image check failed (${errors.length}):`);
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

console.log(`og:image ${OG_WIDTH}x${OG_HEIGHT} PNG on ${KEY_PAGES.length} key pages`);
