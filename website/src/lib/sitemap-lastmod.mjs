/**
 * Sitemap <lastmod> from a content date or the file's last git commit.
 * Build time is not a date. A missing date is null, not today.
 */
import { execFileSync } from 'node:child_process';
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';

const LOCALES = ['pt', 'el', 'zh', 'ja', 'es', 'zh-hk'];
const CONTENT_LOCALES = new Set(['en', ...LOCALES]);

export function isCalendarDate(value) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value ?? '');
  if (!match) return false;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  if (month < 1 || month > 12 || day < 1) return false;
  const lengths = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  if (month === 2 && leap) return day <= 29;
  return day <= lengths[month];
}

export function laterDate(left, right) {
  const a = isCalendarDate(left) ? left : null;
  const b = isCalendarDate(right) ? right : null;
  if (!a) return b;
  if (!b) return a;
  return a > b ? a : b;
}

/** Content date when it is a real calendar day, otherwise the git day. Never the build clock. */
export function resolveLastmod({ contentDate, gitDate }) {
  return laterDate(isCalendarDate(contentDate) ? contentDate : null, isCalendarDate(gitDate) ? gitDate : null);
}

function websiteRoot() {
  return process.cwd();
}

function repoRoot() {
  return path.resolve(websiteRoot(), '..');
}

function contentDateFromFrontmatter(frontmatter) {
  let found = null;
  for (const key of ['last_validated', 'updatedAt', 'publishedAt', 'date']) {
    const match = new RegExp(`^${key}:\\s*['"]?([^'"\\s]+)`, 'm').exec(frontmatter);
    if (!match) continue;
    const day = match[1].slice(0, 10);
    if (isCalendarDate(day)) found = laterDate(found, day);
  }
  return found;
}

function walk(dir, out = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, out);
    else out.push(full);
  }
  return out;
}

let gitDayByFile = null;

function loadGitDays() {
  if (gitDayByFile) return gitDayByFile;
  gitDayByFile = new Map();
  const output = execFileSync(
    'git',
    ['log', '--format=%cs', '--name-only', '--', 'website/src', 'website/public'],
    { cwd: repoRoot(), encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 },
  );
  let day = null;
  for (const line of output.split('\n')) {
    if (isCalendarDate(line.trim())) {
      day = line.trim();
      continue;
    }
    const file = line.trim();
    if (!day || !file || gitDayByFile.has(file)) continue;
    gitDayByFile.set(file, day);
  }
  return gitDayByFile;
}

function gitDay(absPath) {
  const rel = path.relative(repoRoot(), absPath).split(path.sep).join('/');
  return loadGitDays().get(rel) ?? null;
}

function normalizePath(pathname) {
  if (!pathname.startsWith('/')) pathname = `/${pathname}`;
  if (pathname !== '/' && !pathname.endsWith('/')) pathname = `${pathname}/`;
  return pathname.replace(/\/{2,}/g, '/');
}

let table = null;

function bump(map, pathname, date) {
  if (!date) return;
  const key = normalizePath(pathname);
  map.set(key, laterDate(map.get(key) ?? null, date));
}

function registerFile(map, pathname, absPath, contentDate) {
  bump(map, pathname, resolveLastmod({ contentDate, gitDate: gitDay(absPath) }));
}

function splitFrontmatter(text) {
  if (!text.startsWith('---')) return { frontmatter: '', body: text };
  const end = text.indexOf('\n---', 3);
  if (end < 0) return { frontmatter: '', body: text };
  return { frontmatter: text.slice(3, end), body: text.slice(end + 4) };
}

function seriesOf(frontmatter) {
  const match = /^series:\s*['"]?([^'"\n]+)/m.exec(frontmatter);
  return match ? match[1].trim() : '';
}

function buildTable() {
  if (table) return table;
  const map = new Map();
  const root = websiteRoot();
  const pagesRoot = path.join(root, 'src/pages');

  for (const abs of walk(pagesRoot)) {
    if (!/\.(astro|ts|mjs)$/.test(abs)) continue;
    const rel = path.relative(pagesRoot, abs).split(path.sep).join('/');
    if (rel.includes('[') && !rel.includes('[lang]')) continue;
    let route = rel.replace(/\.(astro|ts|mjs)$/, '');
    if (route.endsWith('/index')) route = route.slice(0, -'/index'.length);
    else if (route === 'index') route = '';
    const locales = route.split('/').includes('[lang]') ? LOCALES : [''];
    const bare = route.replace(/\/?\[lang\]\/?/, '/').replace(/^\/+|\/+$/g, '');
    for (const locale of locales) {
      const parts = [locale, bare].filter(Boolean);
      registerFile(map, `/${parts.join('/')}/`, abs, null);
    }
  }

  function addCollection(dirName, urlFor) {
    const dir = path.join(root, 'src/content', dirName);
    for (const abs of walk(dir)) {
      if (!/\.mdx?$/.test(abs)) continue;
      const rel = path.relative(dir, abs).split(path.sep);
      let locale = 'en';
      if (CONTENT_LOCALES.has(rel[0])) {
        locale = rel.shift();
      }
      const text = readFileSync(abs, 'utf8');
      const { frontmatter } = splitFrontmatter(text);
      let slug = rel.join('/').replace(/\.mdx?$/, '');
      if (slug.endsWith('/index')) slug = slug.slice(0, -'/index'.length);
      const target = urlFor({ locale, slug, frontmatter });
      if (!target) continue;
      const contentDate = contentDateFromFrontmatter(frontmatter);
      registerFile(map, target, abs, contentDate);
      if (target.startsWith('/tutorials/') || (locale !== 'en' && target.includes('/tutorials/'))) {
        const series = seriesOf(frontmatter);
        if (series) {
          const prefix = locale === 'en' ? '' : `/${locale}`;
          bump(map, `${prefix}/tutorials/${series}/`, resolveLastmod({ contentDate, gitDate: gitDay(abs) }));
          bump(map, `${prefix}/tutorials/`, resolveLastmod({ contentDate, gitDate: gitDay(abs) }));
        }
      }
    }
  }

  addCollection('docs', ({ locale, slug }) => {
    const prefix = locale === 'en' ? '' : `/${locale}`;
    return `${prefix}/learn/${slug}/`;
  });
  addCollection('showcases', ({ locale, slug }) => {
    const prefix = locale === 'en' ? '' : `/${locale}`;
    return `${prefix}/science/${slug}/`;
  });
  addCollection('blog', ({ locale, slug }) => {
    const prefix = locale === 'en' ? '' : `/${locale}`;
    return `${prefix}/insights/${slug}/`;
  });
  addCollection('tutorials', ({ locale, slug, frontmatter }) => {
    const series = seriesOf(frontmatter);
    if (!series || !slug) return '';
    const prefix = locale === 'en' ? '' : `/${locale}`;
    return `${prefix}/tutorials/${series}/${slug}/`;
  });

  const packages = path.join(root, 'src/content/registry/packages.json');
  try {
    const parsed = JSON.parse(readFileSync(packages, 'utf8'));
    for (const pkg of parsed.packages ?? []) {
      for (const locale of ['', ...LOCALES]) {
        const prefix = locale ? `/${locale}` : '';
        registerFile(map, `${prefix}/packages/${pkg.name}/`, packages, null);
      }
    }
  } catch {
    // The package index page still has its own astro-file date.
  }

  table = map;
  return table;
}

export function lastmodForPathname(pathname) {
  return buildTable().get(normalizePath(pathname)) ?? null;
}

export function lastmodForUrl(urlString) {
  const { pathname } = new URL(urlString);
  return lastmodForPathname(pathname);
}
