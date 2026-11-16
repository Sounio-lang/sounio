import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd());
const distDir = path.join(root, 'dist');

const localeSet = new Set(['pt', 'el', 'zh', 'ja', 'es', 'zh-hk']);

// Vercel serves vercel.json routes before the filesystem, so a legacy path
// whose route answers 308 never reaches its meta-refresh stub in dist/.
const vercelRoutes = JSON.parse(await readFile(path.join(root, 'vercel.json'), 'utf8')).routes ?? [];

function vercelLocationFor(webPath) {
  for (const route of vercelRoutes) {
    const match = webPath.match(new RegExp(route.src));
    if (!match) continue;
    const location = route.headers?.Location;
    if (!location) return null;
    return { status: route.status, location: location.replace(/\$(\d)/g, (_, n) => match[Number(n)] ?? '') };
  }
  return null;
}

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
      test: /^\/platform$/,
      map: () => '/language',
    },
    {
      test: /^\/changelog$/,
      map: () => '/releases',
    },
  ];

  for (const matcher of matchers) {
    const match = basePath.match(matcher.test);
    if (!match) continue;
    const target = `${localePrefix}${matcher.map(match)}`;
    if (target === '/' || target.endsWith('/')) return target;
    return `${target}/`;
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

    // The legacy path, with and without its trailing slash, must get one
    // 308 hop from vercel.json to the same target as the stub.
    for (const requestPath of [`${webPath}/`, webPath]) {
      const route = vercelLocationFor(requestPath);
      if (!route || route.status !== 308 || route.location !== expected) {
        errors.push(
          `vercel.json does not 308 ${requestPath} to ${expected} (got ${route ? `${route.status} ${route.location}` : 'no redirect'})`,
        );
      }
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
