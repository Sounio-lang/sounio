/**
 * Vercel Web Analytics, no cookies. Clicks are counted from
 * data-analytics-event, so a later CTA is tracked without a new listener.
 *
 * Documented names:
 *   github_click
 *   getting_started_click
 *   codespaces_click
 *   install_click
 *   install_copy
 */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

export const ANALYTICS_EVENTS = [
  'github_click',
  'getting_started_click',
  'codespaces_click',
  'install_click',
  'install_copy',
];

const EVENT_NAME = /^[a-z][a-z0-9_]{0,40}$/;
const websiteRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const distDir = path.join(websiteRoot, 'dist');

export function analyticsErrors(html, page) {
  const errors = [];
  if (!html.includes('src="/_vercel/insights/script.js"')) {
    errors.push(`${page}: missing Vercel Web Analytics script`);
  }
  if (!html.includes('data-analytics-event')) {
    errors.push(`${page}: click listener is missing`);
  }
  if (!html.includes("window.va('event'")) errors.push(`${page}: event call is missing`);
  return errors;
}

export function missingEvents(html, names) {
  return names.filter((name) => !html.includes(`data-analytics-event="${name}"`));
}

function selfTest() {
  for (const name of ANALYTICS_EVENTS) assert.equal(EVENT_NAME.test(name), true, name);
  assert.equal(EVENT_NAME.test('GitHub'), false);
  assert.equal(EVENT_NAME.test(''), false);
  const page = `<script src="/_vercel/insights/script.js"></script>
    <script>window.va('event', { name: name })</script>
    data-analytics-event
    <a data-analytics-event="github_click"></a>
    <a data-analytics-event="getting_started_click"></a>`;
  assert.deepEqual(analyticsErrors(page, '/'), []);
  assert.deepEqual(missingEvents(page, ['github_click', 'getting_started_click']), []);
  assert.deepEqual(missingEvents(page, ['codespaces_click']), ['codespaces_click']);
  assert.ok(analyticsErrors('<html></html>', '/').length >= 2);
}

selfTest();

if (process.argv.includes('--self-test')) {
  console.log('analytics self-test passed');
  process.exit(0);
}

const home = await readFile(path.join(distDir, 'index.html'), 'utf8').catch(() => '');
const learn = await readFile(path.join(distDir, 'learn/index.html'), 'utf8').catch(() => '');
const errors = [];
if (!home) errors.push('dist/index.html is missing');
else {
  errors.push(...analyticsErrors(home, '/'));
  errors.push(...missingEvents(home, ['github_click', 'getting_started_click']).map((name) => `/: missing ${name}`));
}
if (!learn) errors.push('dist/learn/index.html is missing');
else {
  errors.push(...missingEvents(learn, ['getting_started_click', 'github_click']).map((name) => `/learn/: missing ${name}`));
}

if (errors.length > 0) {
  console.error(`analytics check failed (${errors.length}):`);
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

console.log(`analytics: ${ANALYTICS_EVENTS.join(', ')}`);
