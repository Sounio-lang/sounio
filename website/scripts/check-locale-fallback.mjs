import { readFile } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd());

const locales = ['pt', 'el', 'zh', 'ja', 'es'];
const iaRoutes = ['/', '/language', '/platform', '/science', '/learn', '/packages', '/insights', '/about', '/releases'];
const fallbackNotice = 'Localized V2 rewrite for this language is in progress. Showing English-first content for now.';

function routeToDistFile(route, locale) {
  if (locale === 'en') {
    return route === '/' ? 'dist/index.html' : `dist${route}/index.html`;
  }
  return route === '/' ? `dist/${locale}/index.html` : `dist/${locale}${route}/index.html`;
}

async function readDist(route, locale) {
  const rel = routeToDistFile(route, locale);
  const abs = path.join(root, rel);
  return { rel, html: await readFile(abs, 'utf8') };
}

async function run() {
  const errors = [];
  let checked = 0;

  for (const route of iaRoutes) {
    try {
      const en = await readDist(route, 'en');
      checked += 1;

      if (en.html.includes(fallbackNotice)) {
        errors.push(`Unexpected fallback notice on English route ${en.rel}`);
      }
      if (!en.html.includes('lang="en"')) {
        errors.push(`Missing lang="en" on ${en.rel}`);
      }
    } catch {
      errors.push(`Missing English IA route output: ${routeToDistFile(route, 'en')}`);
    }

    for (const locale of locales) {
      try {
        const localized = await readDist(route, locale);
        checked += 1;

        if (!localized.html.includes(fallbackNotice)) {
          errors.push(`Missing fallback notice on localized route ${localized.rel}`);
        }
        if (!localized.html.includes(`lang="${locale}"`)) {
          errors.push(`Missing lang="${locale}" on ${localized.rel}`);
        }
      } catch {
        errors.push(`Missing localized IA route output: ${routeToDistFile(route, locale)}`);
      }
    }
  }

  if (errors.length > 0) {
    console.error('Locale fallback validation failed:');
    for (const err of errors) {
      console.error(`- ${err}`);
    }
    process.exit(1);
  }

  console.log(`OK: locale fallback validated (${checked} IA pages).`);
}

await run();
