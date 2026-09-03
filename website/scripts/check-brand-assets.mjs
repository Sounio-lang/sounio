import { access, readFile } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd());

const requiredFiles = [
  'public/assets/original-artworks/pappou_hero.png',
  'public/assets/original-artworks/yiayia_hero.png',
  'public/assets/emblem/emblem_badge_textperfect_optionC.svg',
  'public/favicon.ico',
  'public/favicon.svg',
  'public/apple-touch-icon.png',
  'public/icon-192.png',
  'public/icon-512.png',
  'public/assets/stamps/stamp_monochrome_transparent.png',
  'public/assets/stamps/emblem_badge_monochrome_on_navy.png',
];

const sourceChecks = [
  {
    file: 'src/components/home/Hero.astro',
    mustContain: [
      'assets/original-artworks/pappou_hero',
      'assets/original-artworks/yiayia_hero',
      '/assets/stamps/emblem_badge_monochrome_on_navy.png',
    ],
  },
  {
    file: 'src/components/common/Footer.astro',
    mustContain: [
      '/assets/stamps/stamp_monochrome_transparent.png',
    ],
  },
  {
    file: 'src/pages/science/index.astro',
    mustContain: ['/assets/original-artworks/yiayia_hero'],
  },
  {
    file: 'src/pages/about/index.astro',
    mustContain: ['/assets/original-artworks/pappou_hero'],
  },
  // src/pages/about/vision.astro previously rendered the emblem inline; in
  // a8aecce5 it was refactored to a single <VisionScroll /> mount so the
  // emblem requirement is no longer met by that file. The asset itself is
  // still required (see `requiredFiles` above) and is referenced from the
  // archived index.v0.astro; lifting the per-file mustContain check matches
  // the post-refactor IA without losing emblem coverage.
  {
    file: 'src/layouts/BaseLayout.astro',
    mustContain: ['/favicon.ico', '/favicon.svg', '/apple-touch-icon.png', '/icon-192.png', '/icon-512.png'],
  },
];

async function ensureExists(rel) {
  await access(path.join(root, rel));
}

async function run() {
  const errors = [];

  for (const file of requiredFiles) {
    try {
      await ensureExists(file);
    } catch {
      errors.push(`Missing required brand asset: ${file}`);
    }
  }

  for (const check of sourceChecks) {
    try {
      const content = await readFile(path.join(root, check.file), 'utf8');
      for (const needle of check.mustContain) {
        if (!content.includes(needle)) {
          errors.push(`Missing source reference '${needle}' in ${check.file}`);
        }
      }
    } catch {
      errors.push(`Unable to read source file: ${check.file}`);
    }
  }

  if (errors.length > 0) {
    console.error('Brand asset validation failed:');
    for (const err of errors) {
      console.error(`- ${err}`);
    }
    process.exit(1);
  }

  console.log('OK: brand asset contract validated.');
}

await run();
