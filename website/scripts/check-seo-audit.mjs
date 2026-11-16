/**
 * Acceptance checks for the 2026-11-16 technical SEO audit.
 *
 *   node scripts/check-seo-audit.mjs
 *
 * Reads website/dist after `npm run build`. Exits 1 on any failure.
 */
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd());
const distDir = path.join(root, 'dist');
const LOCALES = new Set(['pt', 'el', 'zh', 'zh-hk', 'ja', 'es']);
const STUBS = 'climate|neuroimaging|causal|graphics|scientific-computing|quantum|pharma|genomics|gpu';
const stubHref = new RegExp(
  `href="(?:/(?:pt|el|zh-hk|zh|ja|es))?/showcases/(?:${STUBS})/?"`,
  'g',
);
const scienceHref = new RegExp(
  `href="(?:/(?:pt|el|zh-hk|zh|ja|es))?/science/(?:${STUBS})/"`,
  'g',
);
const insightsHref = /href="(?:\/(?:pt|el|zh-hk|zh|ja|es))?\/insights\/"/;

function attr(tag, name) {
  const match = new RegExp(`\\b${name}=(["'])(.*?)\\1`, 'i').exec(tag);
  return match?.[2] ?? '';
}

function localeOf(pathname) {
  const first = pathname.split('/').filter(Boolean)[0];
  return LOCALES.has(first) ? first : 'en';
}

function fileFor(pathname) {
  if (pathname === '/') return path.join(distDir, 'index.html');
  const parts = pathname.split('/').filter(Boolean);
  return path.join(distDir, ...parts, 'index.html');
}

async function sitemapLocs() {
  const index = await readFile(path.join(distDir, 'sitemap-index.xml'), 'utf8');
  const children = [...index.matchAll(/<loc>([^<]+)<\/loc>/g)].map((match) => match[1]);
  const locs = [];
  for (const child of children) {
    const name = new URL(child).pathname.split('/').pop();
    const xml = await readFile(path.join(distDir, name), 'utf8');
    for (const url of xml.split(/<url\b/)) {
      const loc = url.match(/<loc>([^<]+)<\/loc>/);
      if (loc) locs.push(loc[1]);
    }
  }
  return locs;
}

function jsonLdBlocks(html) {
  const blocks = [];
  const re = /<script\b([^>]*)>([\s\S]*?)<\/script>/gi;
  let match;
  while ((match = re.exec(html)) !== null) {
    if (!/\btype=(["'])application\/ld\+json\1/i.test(match[1])) continue;
    blocks.push(match[2].trim());
  }
  return blocks;
}

function titleOf(html) {
  return html.match(/<title>([\s\S]*?)<\/title>/i)?.[1]?.replace(/\s+/g, ' ').trim() ?? '';
}

function descriptionOf(html) {
  const tags = html.match(/<meta\b[^>]*>/gi) ?? [];
  for (const tag of tags) {
    if (attr(tag, 'name') !== 'description') continue;
    return attr(tag, 'content').replace(/\s+/g, ' ').trim();
  }
  return '';
}

function footersOf(html) {
  return html.match(/<footer\b[\s\S]*?<\/footer>/gi) ?? [];
}

async function walkHtml(dir, out = []) {
  const { readdir } = await import('node:fs/promises');
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name === 'pagefind') continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await walkHtml(full, out);
    else if (entry.name.endsWith('.html')) out.push(full);
  }
  return out;
}

async function run() {
  const errors = [];
  try {
    await stat(distDir);
  } catch {
    console.error('dist/ is missing; run npm run build first');
    process.exit(1);
  }

  const locs = await sitemapLocs();
  let withJsonLd = 0;
  let missingJsonLd = 0;
  const titles = new Map();
  const descriptions = new Map();

  for (const loc of locs) {
    const url = new URL(loc);
    const pathname = url.pathname;
    const file = fileFor(pathname);
    let html;
    try {
      html = await readFile(file, 'utf8');
    } catch {
      errors.push(`sitemap URL has no HTML: ${pathname}`);
      missingJsonLd += 1;
      continue;
    }
    const blocks = jsonLdBlocks(html);
    let parsed = 0;
    for (const block of blocks) {
      try {
        JSON.parse(block);
        parsed += 1;
      } catch (error) {
        errors.push(`${pathname}: JSON-LD did not parse (${error.message})`);
      }
    }
    if (parsed === 0) {
      missingJsonLd += 1;
      errors.push(`${pathname}: no parseable application/ld+json`);
    } else {
      withJsonLd += 1;
    }
    const locale = localeOf(pathname);
    const title = titleOf(html);
    const description = descriptionOf(html);
    if (!titles.has(locale)) titles.set(locale, new Map());
    if (!descriptions.has(locale)) descriptions.set(locale, new Map());
    const titleBucket = titles.get(locale);
    const descriptionBucket = descriptions.get(locale);
    if (!titleBucket.has(title)) titleBucket.set(title, []);
    titleBucket.get(title).push(pathname);
    if (description) {
      if (!descriptionBucket.has(description)) descriptionBucket.set(description, []);
      descriptionBucket.get(description).push(pathname);
    }
    if (!footersOf(html).some((footer) => insightsHref.test(footer))) {
      errors.push(`${pathname}: footer does not link to an insights hub`);
    }
  }

  for (const [locale, bucket] of titles) {
    for (const [title, pages] of bucket) {
      if (pages.length > 1) errors.push(`${locale} duplicate title ${JSON.stringify(title)}: ${pages.join(', ')}`);
    }
  }
  for (const [locale, bucket] of descriptions) {
    for (const [description, pages] of bucket) {
      if (pages.length > 1) {
        errors.push(`${locale} duplicate description ${JSON.stringify(description)}: ${pages.join(', ')}`);
      }
    }
  }

  const htmlFiles = await walkHtml(distDir);
  for (const file of htmlFiles) {
    const rel = path.relative(distDir, file).split(path.sep).join('/');
    if (/^(?:[^/]+\/)?showcases\/[^/]+\/index\.html$/.test(rel)) {
      const html = await readFile(file, 'utf8');
      if (/http-equiv=["']refresh["']/i.test(html)) {
        errors.push(`${rel}: still contains a meta refresh`);
      } else {
        errors.push(`${rel}: showcase stub page still exists`);
      }
    }
  }

  const showcaseIndexes = htmlFiles.filter((file) => /(?:^|\/)showcases\/index\.html$/.test(path.relative(distDir, file).split(path.sep).join('/')));
  if (showcaseIndexes.length === 0) errors.push('no /showcases/ index was built');
  for (const file of showcaseIndexes) {
    const html = await readFile(file, 'utf8');
    const rel = path.relative(distDir, file);
    const stubs = html.match(stubHref) ?? [];
    if (stubs.length) errors.push(`${rel}: still links to showcase stubs ${stubs.join(', ')}`);
    const science = html.match(scienceHref) ?? [];
    if (science.length < 9) errors.push(`${rel}: expected 9 /science/<slug>/ card links, found ${science.length}`);
  }

  try {
    const llms = await readFile(path.join(distDir, 'llms.txt'), 'utf8');
    if (!llms.startsWith('# Sounio\n')) errors.push('dist/llms.txt does not start with "# Sounio"');
  } catch {
    errors.push('dist/llms.txt is missing');
  }

  const vercel = JSON.parse(await readFile(path.join(root, 'vercel.json'), 'utf8'));
  const routes = vercel.routes ?? [];
  const slashAt = routes.findIndex((route) => route.src === '^/((?:[^/]+/)*[^/\\.]+)$');
  const showcaseRoutes = routes.filter((route) => String(route.src).includes('showcases/'));
  if (showcaseRoutes.length < 2) errors.push('vercel.json is missing the showcase 308 routes');
  for (const route of showcaseRoutes) {
    const index = routes.indexOf(route);
    if (slashAt !== -1 && index > slashAt) errors.push(`showcase route is after the trailing-slash route: ${route.src}`);
    if (route.status !== 308) errors.push(`showcase route is not 308: ${route.src}`);
  }

  console.log(`sitemap URLs: ${locs.length}`);
  console.log(`with parseable JSON-LD: ${withJsonLd}`);
  console.log(`missing JSON-LD: ${missingJsonLd}`);
  console.log(`html files scanned for stubs: ${htmlFiles.length}`);
  console.log(`showcase indexes: ${showcaseIndexes.length}`);

  if (errors.length) {
    console.error(`SEO audit checks failed (${errors.length}):`);
    for (const error of errors.slice(0, 80)) console.error(`- ${error}`);
    if (errors.length > 80) console.error(`- ...and ${errors.length - 80} more`);
    process.exit(1);
  }
  console.log('OK: SEO audit checks passed.');
}

await run();
