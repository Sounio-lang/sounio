import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequestGuard, searchResultsForLocale, urlMatchesLocale } from '../src/lib/search-state.mjs';
import { HREFLANG, htmlLang } from '../src/lib/seo-locales.mjs';
import { htmlVisibleText } from './html-visible-text.mjs';

const website = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repository = path.dirname(website);
const locales = ['en', 'pt', 'el', 'es', 'zh', 'zh-hk', 'ja'];
const args = process.argv.slice(2);
const siteIndex = args.indexOf('--site');
const site = siteIndex < 0 ? path.join(website, 'dist') : path.resolve(args[siteIndex + 1] ?? '');
const copy = JSON.parse(await readFile(path.join(website, 'src/data/publicSurface.json'), 'utf8'));
const manifest = JSON.parse(await readFile(path.join(repository, 'docs/llm-guide/example-manifest.json'), 'utf8'));
const footerSource = await readFile(path.join(website, 'src/components/common/Footer.astro'), 'utf8');
const hash = source => createHash('sha256').update(source).digest('hex');
const prefix = locale => locale === 'en' ? '' : `/${locale}`;

function visibleText(html) {
  return htmlVisibleText(html, { collapseWhitespace: true });
}

function homepageIssues(html, locale) {
  const errors = [];
  const prose = visibleText(html);
  const expected = copy[locale];
  if ((html.match(/<h1\b[^>]*>/g) ?? []).length !== 1) errors.push('expected one primary heading');
  for (const field of ['title', 'description', 'start', 'note', 'scope']) if (!prose.includes(expected[field])) errors.push(`missing localized ${field}`);
  if (!html.includes(`href="${prefix(locale)}/learn/getting-started/"`)) errors.push('missing first-program entry');
  if (!html.includes('class="language-picker"')) errors.push('missing header language control');
  if (!html.includes(`<html lang="${htmlLang(locale)}"`)) errors.push('wrong document language');
  for (const alternate of locales) {
    if (!html.includes(`rel="alternate" hreflang="${HREFLANG[alternate]}"`)) errors.push(`missing language alternate: ${HREFLANG[alternate]}`);
  }
  if (/100\s*%|automatically tracks measurement error|cannot lie about uncertainty|patient safety at compile.time/i.test(prose)) errors.push('unscoped public claim');
  const schemas = [];
  for (const match of html.matchAll(/<script\b[^>]*type="application\/ld\+json"[^>]*>([\s\S]*?)<\/script>/g)) {
    try { schemas.push(JSON.parse(match[1])); } catch { errors.push('invalid structured metadata'); }
  }
  const software = schemas.find(schema => schema['@type'] === 'SoftwareApplication');
  if (!software || 'softwareVersion' in software) errors.push('unverified release metadata');
  if (software?.author?.name !== 'Demetrios Chiuratto Agourakis') errors.push('incorrect project authorship');
  if (!software?.operatingSystem?.includes('macOS arm64 (source-built)')) errors.push('missing native platform boundary');
  return errors;
}

function fencedBlocks(source) {
  const blocks = [];
  let fence = null;
  for (const line of source.replaceAll('\r\n', '\n').split('\n')) {
    if (fence) {
      const closing = /^\s*(`{3,}|~{3,})\s*$/.exec(line);
      if (closing && closing[1][0] === fence.delimiter[0] && closing[1].length >= fence.delimiter.length) {
        blocks.push({ language: fence.language, source: `${fence.lines.join('\n')}\n` });
        fence = null;
      } else fence.lines.push(line.slice(Math.min(fence.indent, /^ */.exec(line)[0].length)));
    } else {
      const opening = /^( *)(`{3,}|~{3,})(\w*)\s*$/.exec(line);
      if (opening) fence = { delimiter: opening[2], language: opening[3], indent: opening[1].length, lines: [] };
    }
  }
  if (fence) throw new Error('Unterminated onboarding fence');
  return blocks;
}

function hubIssues(html, locale) {
  const prose = visibleText(html);
  return ['journey', 'journeyIntro', 'start', 'scope'].filter(field => !prose.includes(copy[locale][field])).map(field => `untranslated learning hub: ${field}`);
}

function exampleIssues(source, expectedId) {
  const errors = [];
  const blocks = fencedBlocks(source);
  const programs = blocks.filter(block => ['sio', 'sounio'].includes(block.language));
  if (programs.length !== 1) errors.push('expected one complete reference program');
  const example = manifest.examples.find(entry => entry.id === expectedId);
  if (!example || example.kind !== 'positive' || example.verification.state !== 'run-passed') errors.push('reference is not an eligible historical example');
  if (programs.some(program => hash(program.source) !== example?.source_sha256)) errors.push('reference source drift');
  if (blocks.some(block => ['sh', 'bash'].includes(block.language) && /^\s*(?:fn|struct)\s+\w+/m.test(block.source))) errors.push('Sounio source in shell fence');
  return errors;
}

async function linkIssues(html) {
  const errors = [];
  for (const match of html.matchAll(/<a\b[^>]*href="([^"]+)"/g)) {
    const href = match[1];
    let target;
    if (href.startsWith('/') && !href.startsWith('//')) {
      const route = href.split(/[?#]/)[0];
      target = path.join(site, route, path.extname(route) ? '' : 'index.html');
    } else {
      const source = /^https:\/\/github\.com\/Sounio-lang\/sounio\/blob\/main\/([^#?]+)/i.exec(href);
      if (source) target = path.join(repository, decodeURIComponent(source[1]));
    }
    if (target) {
      try { await access(target); } catch { errors.push(`missing destination: ${href}`); }
    }
  }
  return errors;
}

const errors = [];
if (/label:\s*['"]Playground['"]/.test(footerSource)) errors.push('footer has hardcoded Playground label');
const expectedKeys = Object.keys(copy.en).sort();
const expectedChrome = Object.keys(copy.en.chrome).sort();
for (const locale of locales) {
  assert.deepEqual(Object.keys(copy[locale]).sort(), expectedKeys, `${locale} copy keys`);
  assert.deepEqual(Object.keys(copy[locale].chrome).sort(), expectedChrome, `${locale} chrome keys`);
  assert.ok(Object.values(copy[locale]).every(value => typeof value === 'object' || typeof value === 'string' && value.trim()), `${locale} empty copy`);
  assert.ok(Object.values(copy[locale].chrome).every(value => typeof value === 'string' && value.trim()), `${locale} empty chrome`);
  let html;
  try { html = await readFile(path.join(site, prefix(locale), 'index.html'), 'utf8'); }
  catch { errors.push(`${locale}: missing built homepage`); continue; }
  errors.push(...homepageIssues(html, locale).map(error => `${locale}: ${error}`));
  errors.push(...(await linkIssues(html)).map(error => `${locale}: ${error}`));
  const hub = await readFile(path.join(site, prefix(locale), 'learn/index.html'), 'utf8');
  errors.push(...hubIssues(hub, locale).map(error => `${locale}: ${error}`));
  errors.push(...(await linkIssues(hub)).map(error => `${locale} hub: ${error}`));
}

let onboardingPages = 0;
for (const locale of ['en', 'pt', 'el']) {
  for (const [page, example] of [['hello-world', 'hello'], ['first-program', 'struct']]) {
    const file = `src/content/docs/${locale}/getting-started/${page}.mdx`;
    const source = await readFile(path.join(website, file), 'utf8');
    errors.push(...exampleIssues(source, example).map(error => `${file}: ${error}`));
    const html = await readFile(path.join(site, prefix(locale), 'learn/getting-started', page, 'index.html'), 'utf8');
    if ((html.match(/<h1\b/g) ?? []).length !== 1) errors.push(`${file}: duplicate primary heading`);
    onboardingPages += 1;
  }
  const file = `src/content/docs/${locale}/getting-started.mdx`;
  const source = await readFile(path.join(website, file), 'utf8');
  if (!source.includes('Linux') || !source.includes('macOS') || !source.includes('ARM64')) errors.push(`${file}: missing platform distinction`);
  const html = await readFile(path.join(site, prefix(locale), 'learn/getting-started/index.html'), 'utf8');
  if ((html.match(/<h1\b/g) ?? []).length !== 1) errors.push(`${file}: duplicate primary heading`);
  onboardingPages += 1;
}

let controls = 0;
let searchControls = 0;
if (args.includes('--self-test')) {
  for (const locale of locales) {
    assert.equal(urlMatchesLocale(`${prefix(locale)}/learn/`, locale, locales), true);
    if (locale !== 'en') assert.equal(urlMatchesLocale(`${prefix(locale)}/learn/`, 'en', locales), false);
    searchControls += 1;
  }
  assert.equal(urlMatchesLocale('https://www.souniolang.org/zh-hk/learn/', 'en', locales), false);
  assert.equal(urlMatchesLocale('/pt-br/learn/', 'pt', locales), false);
  assert.equal(urlMatchesLocale('/learn/', 'unknown', locales), false);
  const guard = createRequestGuard();
  const first = guard.begin();
  const second = guard.begin();
  assert.equal(guard.isCurrent(first), false);
  assert.equal(guard.isCurrent(second), true);
  guard.invalidate();
  assert.equal(guard.isCurrent(second), false);
  searchControls += 6;
  const mixedResults = Array.from({ length: 12 }, (_, index) => ({ data: async () => ({ url: `/learn/${index}/`, title: 'English', excerpt: '' }) }));
  mixedResults.push({ data: async () => ({ url: '/el/learn/start/', title: 'Ελληνικά', excerpt: '' }) });
  assert.deepEqual((await searchResultsForLocale(mixedResults, 'el', locales)).map(result => result.url), ['/el/learn/start/']);
  assert.equal((await searchResultsForLocale(mixedResults, 'en', locales)).length, 10);
  assert.deepEqual(await searchResultsForLocale(mixedResults, 'el', locales, () => false), []);
  let active = true;
  assert.deepEqual(await searchResultsForLocale([{ data: async () => { active = false; return { url: '/el/learn/' }; } }], 'el', locales, () => active), []);
  searchControls += 4;
  const baseline = await readFile(path.join(site, 'index.html'), 'utf8');
  for (const [mutate, expected] of [
    [html => html.replace('<h1', '<h2'), 'primary heading'],
    [html => html.replace('</h1>', '</h1><h1>Duplicate</h1>'), 'primary heading'],
    [html => html.replaceAll(copy.en.title, 'Unreviewed title'), 'localized title'],
    [html => html.replaceAll(copy.en.note, ''), 'localized note'],
    [html => html.replaceAll(copy.en.scope, ''), 'localized scope'],
    [html => html.replaceAll('hreflang="zh-HK"', 'hreflang="unknown"'), 'language alternate'],
    [html => html.replace('class="language-picker"', 'class="missing-language"'), 'language control'],
    [html => html.replace('</main>', '<p>100% translated</p></main>'), 'unscoped public claim'],
    [html => html.replace('"programmingLanguage":"Sounio"', '"programmingLanguage":"Sounio","softwareVersion":"2.0.0"'), 'release metadata'],
    [html => html.replace(/(<script[^>]*type="application\/ld\+json"[^>]*>)\{/, '$1{broken'), 'invalid structured metadata'],
  ]) {
    assert.ok(homepageIssues(mutate(baseline), 'en').some(error => error.includes(expected)), expected);
    controls += 1;
  }
  const source = await readFile(path.join(website, 'src/content/docs/en/getting-started/hello-world.mdx'), 'utf8');
  assert.ok(exampleIssues(source.replace('Hello, Sounio', 'Wrong output'), 'hello').includes('reference source drift'));
  assert.ok(exampleIssues(`${source}\n\x60\x60\x60bash\nfn main() {}\n\x60\x60\x60\n`, 'hello').includes('Sounio source in shell fence'));
  assert.equal(fencedBlocks('````text\n```sh\nfn main() {}\n```\n````')[0].language, 'text');
  controls += 3;
  assert.ok(hubIssues('<h1>Untranslated learning hub</h1>', 'el').length);
  controls += 1;
}

console.log(JSON.stringify({ schema: 'sounio.public-surface.validation.v1', locales: locales.length, learning_hubs: locales.length, onboarding_pages: onboardingPages, mutation_and_parser_controls: controls, search_controls: searchControls, compiler_invoked: false, scope: 'static-content-and-source-identity-only', errors }, null, 2));
if (errors.length) process.exitCode = 1;
