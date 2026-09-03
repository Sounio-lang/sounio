import { readFile } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd());

const locales = ['pt', 'el', 'zh', 'ja', 'es', 'zh-hk'];
// Marketing / shell pages: no translation banner (UI may still be localized)
const shellRoutesNoBanner = ['/', '/language', '/science', '/packages', '/insights', '/about', '/releases'];
// Long-form MDX: banner must appear for non-fully-localized locales
const localizedDocsRoutes = ['/learn', '/learn/feature-status', '/learn/vancomycin-uncertainty', '/tutorials'];
const fallbackNoticeMarker = 'class="locale-notice"';

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

  for (const route of shellRoutesNoBanner) {
    try {
      const en = await readDist(route, 'en');
      checked += 1;

      if (en.html.includes(fallbackNoticeMarker)) {
        errors.push(`Unexpected translation notice on English route ${en.rel}`);
      }
      if (!en.html.includes('lang="en"')) {
        errors.push(`Missing lang="en" on ${en.rel}`);
      }
    } catch {
      errors.push(`Missing English route output: ${routeToDistFile(route, 'en')}`);
    }

    for (const locale of locales) {
      try {
        const localized = await readDist(route, locale);
        checked += 1;

        if (localized.html.includes(fallbackNoticeMarker)) {
          errors.push(`Unexpected translation notice on shell route ${localized.rel} (policy: banner only on /learn and /tutorials)`);
        }
        if (!localized.html.includes(`lang="${locale}"`)) {
          errors.push(`Missing lang="${locale}" on ${localized.rel}`);
        }
      } catch {
        errors.push(`Missing localized route output: ${routeToDistFile(route, locale)}`);
      }
    }
  }

  for (const route of localizedDocsRoutes) {
    try {
      const en = await readDist(route, 'en');
      checked += 1;

      if (en.html.includes(fallbackNoticeMarker)) {
        errors.push(`Unexpected translation notice on English docs route ${en.rel}`);
      }
      if (!en.html.includes('lang="en"')) {
        errors.push(`Missing lang="en" on ${en.rel}`);
      }
    } catch {
      errors.push(`Missing English docs route output: ${routeToDistFile(route, 'en')}`);
    }

    for (const locale of locales) {
      try {
        const localized = await readDist(route, locale);
        checked += 1;

        if (!localized.html.includes(fallbackNoticeMarker)) {
          errors.push(`Missing translation notice on localized docs route ${localized.rel}`);
        }
        if (!localized.html.includes(`lang="${locale}"`)) {
          errors.push(`Missing lang="${locale}" on ${localized.rel}`);
        }
      } catch {
        errors.push(`Missing localized docs route output: ${routeToDistFile(route, locale)}`);
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

  console.log(`OK: locale fallback validated (${checked} pages).`);
}

await run();
