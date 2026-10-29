/**
 * English search-appearance gate.
 *
 * Warns when an English page's <title> is longer than 60 characters or its
 * meta description is longer than 155. Warnings do not fail the run.
 * The six pages in src/lib/search-appearance.mjs must match that copy.
 * Every built page must have one H1. Learn pages must carry TechArticle
 * and a BreadcrumbList taken from the visible breadcrumb. SoftwareApplication
 * stays on the home page. SoftwareSourceCode is on the home page and
 * /learn/getting-started/. /language/ carries one ComputerLanguage whose
 * alternateName and sameAs reuse the former-name constants. /learn/ and
 * /learn/getting-started/ show the research-language status line.
 */
import assert from 'node:assert/strict';
import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { chdir } from 'node:process';
import { fileURLToPath } from 'node:url';
import { parse } from 'parse5';
import { EFFECTS_PAGE, ENGLISH_SEARCH, SELF_HOSTED_PAGE } from '../src/lib/search-appearance.mjs';
import { EFFECTS_FAQ } from '../src/lib/effect-system-page.mjs';
import { rehypeDemoteH1 } from '../src/lib/rehype-demote-h1.mjs';
import { isoDateOrNothing, CODE_REPOSITORY, COMPUTER_LANGUAGE_ID, FORMER_LANGUAGE_NAME, FORMER_RELEASE_URL, computerLanguage } from '../src/lib/structured-data.mjs';

// seo-locales resolves content from cwd. Astro and `npm --prefix website`
// already start there. This command is also run from the repo root.
const website = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
chdir(website);
const { barePath, localeFromPath, isRedirectPath, absoluteUrl, htmlLang, canonicalPathname } = await import('../src/lib/seo-locales.mjs');

const dist = path.join(website, 'dist');
const TITLE_MAX = 60;
const DESCRIPTION_MAX = 155;

function decodeEntity(entity) {
  if (entity === '&amp;') return '&';
  if (entity === '&lt;') return '<';
  if (entity === '&gt;') return '>';
  if (entity === '&quot;') return '"';
  if (entity === '&#39;' || entity === '&apos;') return "'";
  return entity;
}

// One pass. Decoding `&amp;` in a separate earlier replace would turn
// `&amp;lt;` into `&lt;` and then into `<`.
function decode(text) {
  return text.replace(/&(?:amp|lt|gt|quot|#39|apos);/g, decodeEntity);
}

function webPath(relFile) {
  const clean = relFile.split(path.sep).join('/');
  if (clean === 'index.html') return '/';
  if (clean.endsWith('/index.html')) return `/${clean.slice(0, -'index.html'.length)}`;
  if (clean.endsWith('.html')) return `/${clean.slice(0, -'.html'.length)}`;
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

function jsonLd(html) {
  const blocks = [];
  for (const match of html.matchAll(/<script\b[^>]*type="application\/ld\+json"[^>]*>([\s\S]*?)<\/script>/gi)) {
    blocks.push(JSON.parse(match[1]));
  }
  return blocks;
}

function collectAlternateNames(value, found) {
  if (!value || typeof value !== 'object') return;
  if (Array.isArray(value)) {
    for (const item of value) collectAlternateNames(item, found);
    return;
  }
  const alt = value.alternateName;
  if (typeof alt === 'string') found.add(alt);
  else if (Array.isArray(alt)) {
    for (const item of alt) if (typeof item === 'string') found.add(item);
  }
  for (const child of Object.values(value)) collectAlternateNames(child, found);
}

const FORMER_NAME_NOTE = 'Sounio (formerly Demetrios). Sounio was previously released as Demetrios, the former name of this language.';
const FORMER_NAME_NOTE_PT = 'Sounio (anteriormente Demetrios). O Sounio foi lançado anteriormente como Demetrios, o nome anterior desta linguagem.';
const RESEARCH_STATUS = 'Status: research language (pre-1.0). APIs may change; not recommended for production use.';
const COMPILER_VERSION = /v0\.80|v0\.99|v0\.100|v1\.0/;
const RESERVED_SCHEMA_IDS = new Set([
  'https://www.souniolang.org/about/#demetrios',
  'https://www.souniolang.org/about/#dionisio',
  'https://www.souniolang.org/pt/about/#webpage',
]);
const RESEARCH_STATUS_ROUTES = new Set(['/', '/learn/', '/learn/getting-started/']);

function collectTyped(value, type, found) {
  if (!value || typeof value !== 'object') return;
  if (Array.isArray(value)) {
    for (const item of value) collectTyped(item, type, found);
    return;
  }
  if (value['@type'] === type) found.push(value);
  for (const child of Object.values(value)) collectTyped(child, type, found);
}

function collectIds(value, found) {
  if (!value || typeof value !== 'object') return;
  if (Array.isArray(value)) {
    for (const item of value) collectIds(item, found);
    return;
  }
  if (typeof value['@id'] === 'string') found.push(value['@id']);
  for (const child of Object.values(value)) collectIds(child, found);
}

// HTML5 end tags may carry whitespace or ignored attributes before `>`,
// as in `</script >`. A regexp that requires a bare `</script>` leaves
// that element body in the visible text. parse5 follows the HTML parser.
function collectVisibleText(node, parts) {
  if (!node || typeof node !== 'object') return;
  const name = node.nodeName;
  if (name === 'script' || name === 'style') return;
  if (name === '#text') {
    parts.push(node.value);
    return;
  }
  const children = node.childNodes;
  if (children) {
    for (const child of children) collectVisibleText(child, parts);
  }
  if (node.content) collectVisibleText(node.content, parts);
}

function visibleProse(html) {
  const parts = [];
  collectVisibleText(parse(html), parts);
  return parts.join(' ').replace(/\s+/g, ' ');
}

function titleOf(html) {
  const match = html.match(/<title>([\s\S]*?)<\/title>/i);
  return match ? decode(match[1].trim()) : '';
}

function descriptionOf(html) {
  const match = html.match(/<meta\b[^>]*name="description"[^>]*>/i);
  if (!match) return '';
  const quoted = match[0].match(/content="([^"]*)"/);
  return quoted ? decode(quoted[1]) : '';
}

function h1Texts(html) {
  return [...html.matchAll(/<h1\b[^>]*>([\s\S]*?)<\/h1>/gi)].map((match) =>
    decode(match[1].replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim())
  );
}

function firstParagraphAfterH1(html) {
  const h1End = html.search(/<\/h1>/i);
  if (h1End < 0) return '';
  const match = html.slice(h1End).match(/<p\b[^>]*>([\s\S]*?)<\/p>/i);
  if (!match) return '';
  return decode(match[1].replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim());
}

function attr(tag, name) {
  return tag?.match(new RegExp(`${name}="([^"]*)"`))?.[1] ?? '';
}

function visibleCrumbs(html) {
  const nav = html.match(/<nav\b[^>]*aria-label="Breadcrumb"[^>]*>([\s\S]*?)<\/nav>/i);
  if (!nav) return null;
  return [...nav[1].matchAll(/<li\b[^>]*>([\s\S]*?)<\/li>/gi)]
    .map((match) => decode(match[1].replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim()))
    .filter((text) => text && text !== '/');
}

function selfTest() {
  const tree = {
    type: 'root',
    children: [
      { type: 'element', tagName: 'h1', children: [{ type: 'text', value: 'Title' }] },
      {
        type: 'element',
        tagName: 'pre',
        children: [{ type: 'element', tagName: 'span', children: [{ type: 'text', value: '# not a heading' }] }],
      },
    ],
  };
  rehypeDemoteH1()(tree);
  assert.equal(tree.children[0].tagName, 'h2');
  assert.equal(tree.children[1].children[0].tagName, 'span');
  assert.equal(isoDateOrNothing('unvalidated'), undefined);
  assert.equal(isoDateOrNothing('2026-10-27'), '2026-10-27');
  assert.equal(isoDateOrNothing('2026-02-31'), undefined);
  assert.equal(isoDateOrNothing('2026-10-27T12:00:00Z'), undefined);
  assert.equal(decode('Sounio&#39;s'), "Sounio's");
  assert.equal(decode('&amp;'), '&');
  assert.equal(decode('&lt;h1&gt;'), '<h1>');
  assert.equal(decode('&amp;lt;'), '&lt;');
  assert.equal(decode('&amp;amp;'), '&amp;');

  const aboutSnippet = `<section><h1>About</h1><p>
        Sounio (formerly Demetrios). Sounio was previously released as Demetrios, the former name of this language.
      </p></section>`;
  assert.equal(visibleProse(aboutSnippet).includes(FORMER_NAME_NOTE), true);

  const closings = [
    '</script >',
    '</script\t>',
    '</script\n>',
    '</script foo="bar">',
    '</script\t\n bar>',
    '</SCRIPT >',
  ];
  for (const closing of closings) {
    const hidden = `<script>SECRET_TOKEN ${FORMER_NAME_NOTE}${closing}<p>visible only</p>`;
    const prose = visibleProse(hidden);
    assert.equal(prose.includes('SECRET_TOKEN'), false, closing);
    assert.equal(prose.includes(FORMER_NAME_NOTE), false, closing);
    assert.equal(prose.includes('visible only'), true, closing);
  }
  const styled = '<style>SECRET_STYLE</style ><p>visible only</p>';
  assert.equal(visibleProse(styled).includes('SECRET_STYLE'), false);
  assert.equal(visibleProse(styled).includes('visible only'), true);

  const languageNode = computerLanguage({ description: 'Authoritative tour of Sounio.' });
  assert.equal(languageNode['@type'], 'ComputerLanguage');
  assert.equal(languageNode['@id'], COMPUTER_LANGUAGE_ID);
  assert.equal(languageNode.name, 'Sounio');
  assert.equal(languageNode.alternateName, FORMER_LANGUAGE_NAME);
  assert.deepEqual(languageNode.sameAs, [FORMER_RELEASE_URL]);
  assert.equal('softwareVersion' in languageNode, false);
  assert.equal(COMPILER_VERSION.test(JSON.stringify(languageNode)), false);
  assert.equal(COMPILER_VERSION.test(RESEARCH_STATUS), false);
  assert.equal(RESERVED_SCHEMA_IDS.has(COMPUTER_LANGUAGE_ID), false);
}

selfTest();

let distStat;
try {
  distStat = await stat(dist);
} catch {
  distStat = null;
}
if (!distStat?.isDirectory()) {
  console.error('search-appearance: dist/ is missing. Run the website build first.');
  process.exit(1);
}

const warnings = [];
const errors = [];
let pages = 0;
let englishPages = 0;

const i18nDir = path.join(website, 'src/i18n');
const i18nFiles = (await readdir(i18nDir)).filter((name) => name.endsWith('.json'));
if (i18nFiles.length === 0) errors.push('i18n: no locale files');
for (const file of i18nFiles) {
  const data = JSON.parse(await readFile(path.join(i18nDir, file), 'utf8'));
  if (data.maturity_description !== RESEARCH_STATUS) {
    errors.push(`src/i18n/${file}: maturity_description drifted from the shared status line`);
  }
}

for await (const file of walk(dist)) {
  if (!file.endsWith('.html')) continue;
  const rel = path.relative(dist, file);
  const route = webPath(rel);
  if (!route) continue;
  if (isRedirectPath(route)) continue;
  const html = await readFile(file, 'utf8');
  const bare = barePath(route);
  const locale = localeFromPath(route);
  pages += 1;

  const headings = h1Texts(html);
  if (headings.length !== 1) errors.push(`${route}: expected 1 H1, found ${headings.length}`);

  let blocks = [];
  try {
    blocks = jsonLd(html);
  } catch (error) {
    errors.push(`${route}: invalid JSON-LD (${error.message})`);
    continue;
  }
  const software = blocks.filter((block) => block['@type'] === 'SoftwareApplication');
  const source = blocks.filter((block) => block['@type'] === 'SoftwareSourceCode');
  if (bare === '/' || bare === '/about/' || bare === '/language/') {
    const names = new Set();
    for (const block of blocks) collectAlternateNames(block, names);
    if (!names.has(FORMER_LANGUAGE_NAME)) {
      errors.push(`${route}: JSON-LD missing alternateName ${FORMER_LANGUAGE_NAME}`);
    }
    const serialized = JSON.stringify(blocks);
    if (!serialized.includes(FORMER_RELEASE_URL)) {
      errors.push(`${route}: JSON-LD missing former-release sameAs ${FORMER_RELEASE_URL}`);
    }
  }
  if (RESEARCH_STATUS_ROUTES.has(bare)) {
    if (!visibleProse(html).includes(RESEARCH_STATUS)) {
      errors.push(`${route}: research-language status line is missing`);
    }
    if (headings.includes(RESEARCH_STATUS)) {
      errors.push(`${route}: research-language status line is a heading`);
    }
  }
  const schemaIds = [];
  collectIds(blocks, schemaIds);
  if (bare === '/language/') {
    const languages = [];
    collectTyped(blocks, 'ComputerLanguage', languages);
    if (languages.length !== 1) {
      errors.push(`${route}: expected one ComputerLanguage, found ${languages.length}`);
    }
    const language = languages[0];
    if (language) {
      if (language.name !== 'Sounio') errors.push(`${route}: ComputerLanguage.name`);
      if (language.alternateName !== FORMER_LANGUAGE_NAME) {
        errors.push(`${route}: ComputerLanguage.alternateName`);
      }
      const sameAs = Array.isArray(language.sameAs) ? language.sameAs : [language.sameAs];
      if (!sameAs.includes(FORMER_RELEASE_URL)) errors.push(`${route}: ComputerLanguage.sameAs`);
      if (language['@id'] !== COMPUTER_LANGUAGE_ID) errors.push(`${route}: ComputerLanguage.@id`);
      if ('softwareVersion' in language) errors.push(`${route}: ComputerLanguage.softwareVersion`);
      if (COMPILER_VERSION.test(JSON.stringify(language))) {
        errors.push(`${route}: ComputerLanguage names a compiler version`);
      }
    }
    const idCounts = new Map();
    for (const id of schemaIds) idCounts.set(id, (idCounts.get(id) ?? 0) + 1);
    for (const [id, count] of idCounts) {
      if (count > 1) errors.push(`${route}: duplicate @id ${id}`);
      if (RESERVED_SCHEMA_IDS.has(id)) errors.push(`${route}: reuses @id ${id}`);
    }
    const nestedSource = [];
    collectTyped(blocks, 'SoftwareSourceCode', nestedSource);
    if (nestedSource.length !== 0) {
      errors.push(`${route}: ComputerLanguage must not duplicate SoftwareSourceCode`);
    }
  } else if (schemaIds.includes(COMPUTER_LANGUAGE_ID)) {
    errors.push(`${route}: reuses ComputerLanguage @id ${COMPUTER_LANGUAGE_ID}`);
  }
  if (bare === '/about/') {
    const expectedNote = locale === 'pt' ? FORMER_NAME_NOTE_PT : FORMER_NAME_NOTE;
    if (!visibleProse(html).includes(expectedNote)) {
      errors.push(`${route}: visible former-name note is missing`);
    }
  }
  const articles = blocks.filter((block) => block['@type'] === 'TechArticle');
  const crumbs = blocks.filter((block) => block['@type'] === 'BreadcrumbList');
  if (bare === '/language/' && crumbs.length !== 1) {
    errors.push(`${route}: language page must keep one BreadcrumbList`);
  }

  if (bare === '/') {
    if (software.length !== 1) errors.push(`${route}: home must have one SoftwareApplication`);
    if (source.length !== 1) errors.push(`${route}: home must have one SoftwareSourceCode`);
    if (articles.length !== 0) errors.push(`${route}: home must not be a TechArticle`);
  } else if (software.length !== 0) {
    errors.push(`${route}: SoftwareApplication is only for the home page`);
  }

  if (bare === '/learn/getting-started/') {
    if (source.length !== 1) errors.push(`${route}: getting-started must have one SoftwareSourceCode`);
  } else if (bare !== '/' && source.length !== 0) {
    errors.push(`${route}: SoftwareSourceCode is only for the home page and getting-started`);
  }

  for (const block of source) {
    if (block.codeRepository !== CODE_REPOSITORY) errors.push(`${route}: SoftwareSourceCode.codeRepository`);
    if (block.programmingLanguage !== 'Sounio') errors.push(`${route}: SoftwareSourceCode.programmingLanguage`);
  }
  for (const block of software) {
    if ('softwareVersion' in block) errors.push(`${route}: unverified softwareVersion`);
    if (block.author?.name !== 'Demetrios Chiuratto Agourakis') errors.push(`${route}: software author`);
  }

  if (bare.startsWith('/learn/')) {
    if (articles.length !== 1) errors.push(`${route}: learn page must have one TechArticle`);
    if (crumbs.length !== 1) errors.push(`${route}: learn page must have one BreadcrumbList`);
    const article = articles[0];
    if (article) {
      for (const field of ['headline', 'description', 'inLanguage', 'url']) {
        if (typeof article[field] !== 'string' || article[field].length === 0) {
          errors.push(`${route}: TechArticle.${field} missing`);
        }
      }
      if ('dateModified' in article && article.dateModified !== isoDateOrNothing(article.dateModified)) {
        errors.push(`${route}: TechArticle.dateModified is not an available calendar date`);
      }
      const canonical = canonicalPathname(route);
      if (article.url !== absoluteUrl(canonical)) errors.push(`${route}: TechArticle url ${article.url}`);
      if (article.inLanguage !== htmlLang(localeFromPath(canonical))) {
        errors.push(`${route}: TechArticle inLanguage ${article.inLanguage}`);
      }
    }
    const visible = visibleCrumbs(html);
    const list = crumbs[0]?.itemListElement;
    if (visible && list) {
      const names = list.map((item) => item.name);
      if (names.join('\n') !== visible.join('\n')) {
        errors.push(`${route}: BreadcrumbList names do not match the visible breadcrumb`);
      }
      list.forEach((item, index) => {
        if (item['@type'] !== 'ListItem' || item.position !== index + 1) {
          errors.push(`${route}: BreadcrumbList position ${index + 1}`);
        }
        if (typeof item.item !== 'string' || !item.item.startsWith('https://www.souniolang.org/')) {
          errors.push(`${route}: BreadcrumbList item URL`);
        }
      });
    }
  }

  if (locale !== 'en') continue;
  englishPages += 1;
  const title = titleOf(html);
  const description = descriptionOf(html);
  if (title.length > TITLE_MAX) warnings.push(`${route}: title ${title.length} > ${TITLE_MAX}: ${title}`);
  if (description.length > DESCRIPTION_MAX) warnings.push(`${route}: description ${description.length} > ${DESCRIPTION_MAX}`);

  const expected = ENGLISH_SEARCH[bare];
  if (!expected) continue;
  if (title !== expected.title) errors.push(`${route}: title is ${JSON.stringify(title)}, expected ${JSON.stringify(expected.title)}`);
  if (description !== expected.description) errors.push(`${route}: description does not match the recommended copy`);
  if (headings.length === 1 && headings[0] === expected.title) {
    errors.push(`${route}: the visible H1 was replaced by the search title`);
  }
  const article = articles[0];
  if (article) {
    const isSelfHostedPage = bare === '/learn/compiler/self-hosted/';
    const isEffectsPage = bare === '/learn/effects/';
    const headline = isSelfHostedPage
      ? SELF_HOSTED_PAGE.h1
      : isEffectsPage
        ? EFFECTS_PAGE.h1
        : expected.title;
    if (article.headline !== headline) errors.push(`${route}: TechArticle headline does not match ${JSON.stringify(headline)}`);
    if (article.description !== expected.description) errors.push(`${route}: TechArticle description does not match the meta description`);
    if (article.inLanguage !== 'en') errors.push(`${route}: TechArticle inLanguage`);
    if (article.url !== absoluteUrl(bare)) errors.push(`${route}: TechArticle url ${article.url}`);
    if (isEffectsPage) {
      if (article.dateModified !== EFFECTS_PAGE.dateModified) {
        errors.push(`${route}: dateModified is ${JSON.stringify(article.dateModified)}`);
      }
    } else if ('dateModified' in article) {
      errors.push(`${route}: dateModified was emitted without a calendar date in the source`);
    }
  }
  if (bare === '/learn/compiler/self-hosted/') {
    if (headings[0] !== SELF_HOSTED_PAGE.h1) {
      errors.push(`${route}: H1 is ${JSON.stringify(headings[0])}, expected ${JSON.stringify(SELF_HOSTED_PAGE.h1)}`);
    }
    const lede = firstParagraphAfterH1(html);
    if (lede !== SELF_HOSTED_PAGE.lede) errors.push(`${route}: first paragraph after H1 does not match the answer`);
    const versionHit = html.match(/v0\.80|v0\.99|v0\.100|v1\.0/);
    if (versionHit) errors.push(`${route}: compiler version ${versionHit[0]} appears in the HTML`);
    if (/\bphysician\b|\bPhD\b|\bDr\.|\bMD\b|honorificPrefix/.test(html)) {
      errors.push(`${route}: forbidden credential or honorificPrefix in the HTML`);
    }
    const canonicalTag = html.match(/<link\b[^>]*rel="canonical"[^>]*>/i)?.[0] ?? '';
    if (attr(canonicalTag, 'href') !== 'https://www.souniolang.org/learn/compiler/self-hosted/') {
      errors.push(`${route}: canonical is ${attr(canonicalTag, 'href')}`);
    }
    if (!html.includes('hreflang="en"') || !html.includes('hreflang="x-default"')) {
      errors.push(`${route}: hreflang alternates are missing`);
    }
    const names = crumbs[0]?.itemListElement?.map((item) => item.name) ?? [];
    if (names.join('\n') !== SELF_HOSTED_PAGE.crumbs.join('\n')) {
      errors.push(`${route}: BreadcrumbList is ${names.join(' / ')}`);
    }
    const author = article?.author;
    if (JSON.stringify(author) !== JSON.stringify(SELF_HOSTED_PAGE.author)) {
      errors.push(`${route}: TechArticle author ${JSON.stringify(author)}`);
    }
    if (author && ('honorificPrefix' in author || 'honorificSuffix' in author)) {
      errors.push(`${route}: TechArticle author carries an honorific`);
    }
    if (JSON.stringify(article?.about) !== JSON.stringify(SELF_HOSTED_PAGE.about)) {
      errors.push(`${route}: TechArticle about ${JSON.stringify(article?.about)}`);
    }
  }
  if (bare === '/learn/effects/') {
    if (headings[0] !== EFFECTS_PAGE.h1) {
      errors.push(`${route}: H1 is ${JSON.stringify(headings[0])}, expected ${JSON.stringify(EFFECTS_PAGE.h1)}`);
    }
    const lede = firstParagraphAfterH1(html);
    if (lede !== EFFECTS_PAGE.lede) errors.push(`${route}: first paragraph after H1 does not match the answer`);
    const versionHit = html.match(/v0\.80|v0\.99|v0\.100|v1\.0/);
    if (versionHit) errors.push(`${route}: compiler version ${versionHit[0]} appears in the HTML`);
    if (/\bAsync\b/.test(html)) errors.push(`${route}: Async appears in the HTML`);
    if (/handle&lt;(?!\/)|handle<(?!\/)/.test(html)) {
      errors.push(`${route}: handle< appears in the HTML`);
    }
    if (/\bphysician\b|\bPhD\b|\bDr\.|\bMD\b|honorificPrefix/.test(html)) {
      errors.push(`${route}: forbidden credential or honorificPrefix in the HTML`);
    }
    if ('softwareVersion' in (article ?? {})) errors.push(`${route}: softwareVersion on TechArticle`);
    if (JSON.stringify(article?.author) !== JSON.stringify(EFFECTS_PAGE.author)) {
      errors.push(`${route}: TechArticle author ${JSON.stringify(article?.author)}`);
    }
    if (JSON.stringify(article?.about) !== JSON.stringify(EFFECTS_PAGE.about)) {
      errors.push(`${route}: TechArticle about ${JSON.stringify(article?.about)}`);
    }
    if (JSON.stringify(article?.isPartOf) !== JSON.stringify(EFFECTS_PAGE.isPartOf)) {
      errors.push(`${route}: TechArticle isPartOf ${JSON.stringify(article?.isPartOf)}`);
    }
    if (article?.proficiencyLevel !== EFFECTS_PAGE.proficiencyLevel) {
      errors.push(`${route}: TechArticle proficiencyLevel`);
    }
    if (JSON.stringify(article?.dependencies) !== '[]') {
      errors.push(`${route}: TechArticle dependencies ${JSON.stringify(article?.dependencies)}`);
    }
    const faq = blocks.find((block) => block['@type'] === 'FAQPage');
    if (!faq) errors.push(`${route}: FAQPage missing`);
    const entities = faq?.mainEntity ?? [];
    if (entities.length !== EFFECTS_FAQ.length) {
      errors.push(`${route}: FAQPage has ${entities.length} questions`);
    }
    const visibleHtml = visibleProse(html);
    EFFECTS_FAQ.forEach((item, index) => {
      const entity = entities[index];
      if (entity?.name !== item.question || entity?.acceptedAnswer?.text !== item.answer) {
        errors.push(`${route}: FAQPage item ${index} does not match the source strings`);
      }
      if (!visibleHtml.includes(item.question) || !visibleHtml.includes(item.answer)) {
        errors.push(`${route}: FAQ item ${index} is not visible`);
      }
    });
    for (const rel of [
      'tests/compile-fail/effect_io_in_pure.sio',
      'tests/compile-fail/effects_unhandled.sio',
      'tests/run-pass/effect_handler_basic.sio',
    ]) {
      if (!html.includes(rel)) errors.push(`${route}: missing test path ${rel}`);
    }
  }
  if (bare === '/') {
    const app = software[0];
    if (app && app.description !== expected.description) errors.push(`${route}: SoftwareApplication description drifted from the meta description`);
  }
}

for (const page of Object.keys(ENGLISH_SEARCH)) {
  const suffix = page === '/' ? 'index.html' : `${page.slice(1)}index.html`;
  const file = path.join(dist, suffix);
  try {
    await stat(file);
  } catch {
    errors.push(`${page}: missing built page ${suffix}`);
  }
}

if (pages === 0) errors.push('no HTML pages checked');

const readmePath = path.join(website, '..', 'README.md');
const readmeHead = (await readFile(readmePath, 'utf8')).split('\n').slice(0, 25).join('\n');
if (!readmeHead.includes('Sounio (formerly Demetrios)') || !readmeHead.includes('previously released as Demetrios, the former name of this language.')) {
  errors.push('README.md: former-name note is missing from the top of the file');
}

for (const warning of warnings) console.warn(`search-appearance: WARN ${warning}`);
if (errors.length) {
  for (const error of errors) console.error(`search-appearance: ${error}`);
  process.exitCode = 1;
}
console.log(JSON.stringify({
  schema: 'sounio.search-appearance.v1',
  pages,
  englishPages,
  warnings: warnings.length,
  errors: errors.length,
  priorityPages: Object.keys(ENGLISH_SEARCH),
}, null, 2));
