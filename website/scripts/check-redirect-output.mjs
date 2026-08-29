import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd());
const distDir = path.join(root, 'dist');

const localeSet = new Set(['pt', 'el', 'zh', 'ja', 'es', 'zh-hk']);

function normalizeWebPathFromFile(relFile) {
  const clean = relFile.split(path.sep).join('/');
  if (clean.endsWith('/index.html')) {
    return `/${clean.slice(0, -'/index.html'.length)}` || '/';
  }
  if (clean.endsWith('.html')) {
    return `/${clean.slice(0, -'.html'.length)}`;
  }
  return null;
}

function mapLegacyPathToTarget(webPath) {
  const parts = webPath.split('/').filter(Boolean);

  let localePrefix = '';
  let basePath = webPath;

  if (parts.length > 0 && localeSet.has(parts[0])) {
    localePrefix = `/${parts[0]}`;
    basePath = `/${parts.slice(1).join('/')}`;
    if (basePath === '/') basePath = '/';
  }

  const matchers = [
    {
      test: /^\/docs(?:\/(.+))?$/,
      map: (m) => (m[1] ? `/learn/${m[1]}` : '/learn'),
    },
    {
      test: /^\/showcases\/(.+)$/,
      map: (m) => `/science/${m[1]}`,
    },
    {
      test: /^\/registry(?:\/(.+))?$/,
      map: (m) => (m[1] ? `/packages/${m[1]}` : '/packages'),
    },
    {
      test: /^\/blog$/,
      map: () => '/insights',
    },
    {
      test: /^\/manifesto$/,
      map: () => '/about/vision',
    },
    {
      test: /^\/roadmap$/,
      map: () => '/about/roadmap',
    },
    {
      test: /^\/changelog$/,
      map: () => '/releases',
    },
  ];

  for (const matcher of matchers) {
    const match = basePath.match(matcher.test);
    if (!match) continue;
    return `${localePrefix}${matcher.map(match)}`;
  }

  return null;
}

async function* walk(dir) {
  const entries = await readdir(dir);
  for (const entry of entries) {
    const full = path.join(dir, entry);
    const info = await stat(full);
    if (info.isDirectory()) {
      yield* walk(full);
      continue;
    }
    yield full;
  }
}

async function run() {
  const errors = [];
  let checked = 0;

  for await (const file of walk(distDir)) {
    if (!file.endsWith('.html')) continue;

    const rel = path.relative(distDir, file);
    const webPath = normalizeWebPathFromFile(rel);
    if (!webPath) continue;

    const expected = mapLegacyPathToTarget(webPath);
    if (!expected) continue;

    checked += 1;

    const html = await readFile(file, 'utf8');
    if (!html.includes(expected)) {
      errors.push(`Redirect target mismatch for ${webPath} (expected target: ${expected})`);
      continue;
    }

    if (html.includes(`url=${webPath}`)) {
      errors.push(`Redirect loop detected at ${webPath}`);
    }
  }

  if (checked === 0) {
    errors.push('No legacy redirect outputs were checked in dist/.');
  }

  if (errors.length > 0) {
    console.error('Redirect output validation failed:');
    for (const err of errors) {
      console.error(`- ${err}`);
    }
    process.exit(1);
  }

  console.log(`OK: redirect outputs validated (${checked} legacy pages).`);
}

await run();
