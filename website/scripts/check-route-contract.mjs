import { readFile } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd());

const requiredRoutes = [
  'src/pages/index.astro',
  'src/pages/language.astro',
  'src/pages/science/index.astro',
  'src/pages/science/[slug].astro',
  'src/pages/learn/index.astro',
  'src/pages/learn/[...slug].astro',
  'src/pages/packages/index.astro',
  'src/pages/packages/[name].astro',
  'src/pages/insights/index.astro',
  'src/pages/insights/[slug].astro',
  'src/pages/about/index.astro',
  'src/pages/about/vision.astro',
  'src/pages/about/roadmap.astro',
  'src/pages/releases/index.astro',
];

// Legacy paths are permanent redirects in vercel.json, not Astro meta-refresh stubs.
const localePrefixes = ['', '/pt', '/el', '/zh', '/ja', '/es', '/zh-hk'];
const requiredVercelRedirects = localePrefixes.flatMap((prefix) => [
  [`${prefix}/manifesto`, `${prefix}/about/vision/`],
  [`${prefix}/roadmap`, `${prefix}/about/roadmap/`],
  [`${prefix}/platform`, `${prefix}/language/`],
  [`${prefix}/changelog`, `${prefix}/releases/`],
  [`${prefix}/registry`, `${prefix}/packages/`],
  [`${prefix}/registry/:name`, `${prefix}/packages/`],
  [`${prefix}/docs/:path*`, `${prefix}/learn/:path*/`],
  [`${prefix}/showcases`, `${prefix}/science/`],
  [`${prefix}/showcases/:slug`, `${prefix}/science/:slug/`],
  [`${prefix}/blog`, `${prefix}/insights/`],
]);

const redirectContracts = [
  { file: 'src/pages/registry/[name].astro', needle: '/packages/' },
  { file: 'src/pages/[lang]/registry/[name].astro', needle: '/packages/' },
];

async function exists(rel) {
  try {
    await readFile(path.join(root, rel), 'utf8');
    return true;
  } catch {
    return false;
  }
}

async function run() {
  const errors = [];

  for (const route of requiredRoutes) {
    if (!(await exists(route))) {
      errors.push(`Missing required V2 route file: ${route}`);
    }
  }

  for (const contract of redirectContracts) {
    try {
      const content = await readFile(path.join(root, contract.file), 'utf8');
      if (!content.includes(contract.needle) || !content.includes('Astro.redirect')) {
        errors.push(`Redirect contract mismatch in ${contract.file}`);
      }
    } catch {
      errors.push(`Unable to read redirect contract file: ${contract.file}`);
    }
  }

  try {
    const vercel = JSON.parse(await readFile(path.join(root, 'vercel.json'), 'utf8'));
    const pairs = new Set(
      (vercel.redirects ?? []).map((entry) => `${entry.source} -> ${entry.destination}`)
    );
    for (const [source, destination] of requiredVercelRedirects) {
      if (!pairs.has(`${source} -> ${destination}`)) {
        errors.push(`Missing permanent redirect in vercel.json: ${source} -> ${destination}`);
      }
    }
    for (const entry of vercel.redirects ?? []) {
      if (entry.permanent !== true) {
        errors.push(`Redirect is not permanent: ${entry.source}`);
      }
    }
  } catch (error) {
    errors.push(`Unable to read vercel.json redirects: ${error instanceof Error ? error.message : error}`);
  }

  if (errors.length > 0) {
    console.error('Route contract validation failed:');
    for (const err of errors) {
      console.error(`- ${err}`);
    }
    process.exit(1);
  }

  console.log('OK: route contract validated.');
}

await run();
await import('./check-unguarded-numerals.mjs');
