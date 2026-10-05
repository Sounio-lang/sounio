import { readFile } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd());

const requiredRoutes = [
  'src/pages/index.astro',
  'src/pages/language.astro',
  'src/pages/platform.astro',
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

const redirectContracts = [
  { file: 'src/pages/docs/index.astro', needle: '/learn' },
  { file: 'src/pages/docs/[...slug].astro', needle: '/learn/' },
  { file: 'src/pages/showcases/[slug].astro', needle: '/science/' },
  { file: 'src/pages/registry/index.astro', needle: '/packages' },
  { file: 'src/pages/registry/[name].astro', needle: '/packages/' },
  { file: 'src/pages/blog/index.astro', needle: '/insights' },
  { file: 'src/pages/manifesto.astro', needle: '/about/vision' },
  { file: 'src/pages/roadmap.astro', needle: '/about/roadmap' },
  { file: 'src/pages/changelog.astro', needle: '/releases' },
  { file: 'src/pages/[lang]/docs/index.astro', needle: '/learn' },
  { file: 'src/pages/[lang]/docs/[...slug].astro', needle: '/learn/' },
  { file: 'src/pages/[lang]/showcases/[slug].astro', needle: '/science/' },
  { file: 'src/pages/[lang]/registry/index.astro', needle: '/packages' },
  { file: 'src/pages/[lang]/registry/[name].astro', needle: '/packages/' },
  { file: 'src/pages/[lang]/blog/index.astro', needle: '/insights' },
  { file: 'src/pages/[lang]/manifesto.astro', needle: '/about/vision' },
  { file: 'src/pages/[lang]/roadmap.astro', needle: '/about/roadmap' },
  { file: 'src/pages/[lang]/changelog.astro', needle: '/releases' },
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
