# Website i18n policy (souniolang.org)

## Locales

Configured in [astro.config.mjs](../astro.config.mjs) and [src/lib/i18n.ts](../src/lib/i18n.ts): `en` (default, unprefixed), `pt`, `el`, `zh`, `ja`, `es`, `zh-hk`.

## "Fully localized" flag

`fullyLocalizedLocales` in [src/lib/i18n.ts](../src/lib/i18n.ts) is the single switch for treating a locale as complete. Today only `en` is listed. When a locale passes human review (MDX parity with English + UI string drift check), add it to that set; the translation banner stops automatically for that locale.

The public URL prefix is intentionally short (`/pt`, `/zh`, `/zh-hk`). `<html lang>`, `<link rel="alternate" hreflang>`, and sitemap `xhtml:link` all use one BCP 47 tag per prefix, the tags the sitemap already published:

| URL prefix | hreflang / html lang | Open Graph |
|---|---|---|
| (none) | `en` | `en_US` |
| `/pt` | `pt-BR` | `pt_BR` |
| `/el` | `el` | `el` |
| `/zh` | `zh-CN` | `zh_CN` |
| `/ja` | `ja` | `ja` |
| `/es` | `es` | `es` |
| `/zh-hk` | `zh-HK` | `zh_HK` |

`x-default` is the English URL. Split `pt-BR` / `pt-PT` only when both variants have reviewed copy. The map lives in [src/lib/seo-locales.mjs](../src/lib/seo-locales.mjs).

## Translation banner scope

**Problem we solved:** Showing the "not yet translated / English is authoritative" banner on every non-English page (including the homepage) made the product feel broken even when UI strings were translated.

**Current rule:** The banner appears only on routes whose primary content is long-form documentation that may lag English:

- Paths under `/learn/**`
- Paths under `/tutorials/**`

(Including localized prefixes, e.g. `/pt/learn/getting-started`.)

**Banner does not appear on:** Home (`/`, `/pt`, etc.), `/language`, `/proof`, `/manifesto`, `/about`, `/blog`, `/insights`, `/science`, `/packages`, `/playground`, `/changelog`, etc. Those pages may still be English-first in body copy; the UI chrome follows the selected locale.

## Content sync

Translated MDX under `src/content/docs/<locale>/` must be refreshed when English changes. Track drift via `npm run check:docs-parity` and the checklist in [i18n/pt/TRANSLATION_TODO.md](../i18n/pt/TRANSLATION_TODO.md) (other locales: same pattern).

## Canonical and hreflang

A localized URL is an indexable translation only when its **main document** is not the English source. Indexable translations self-canonicalize. Every other localized URL canonicalizes to the English URL for the same path, and that URL is omitted from every hreflang cluster and from the sitemap. Pointing hreflang at a URL that canonicalizes elsewhere is invalid, and an English query should not land on a near-duplicate.

What counts as a translation:

- **Content documents** (`/learn/<slug>/`, `/science/<slug>/`, `/insights/<slug>/`, `/tutorials/<series>/<lesson>/`): a locale file exists and its prose is not the English prose. A missing file is the English fallback. Today that is `zh-hk` for `/learn/language/units-and-uncertainty/` (no `zh-hk` source). `/zh/learn/compiler/effect-system/` has its own prose, so it self-canonicalizes.
- **Listing hubs** whose cards are those documents (`/science/`, `/tutorials/`, `/tutorials/<series>/`) follow the same locales. `/learn/` also follows [publicSurface.json](../src/data/publicSurface.json).
- **Home** (`/`) follows `publicSurface.json` (every current locale differs from English).
- **`/language/` and `/compiler/`** branch the article for Greek only.
- **`/scientist/`** follows `scientistCopy.json` (`pt`, `el`).
- **`/honesty/`** has a full Portuguese article; other locales render the English article.
- **`/about/`** has a full Portuguese article; other locales render the English article. `/about/vision/` and `/about/roadmap/` stay English-only.
- **`/releases/`** has a distinct notes array for every locale, including `zh-hk`.
- **Everything else** is English-only even when the chrome or the `<title>` is translated. That includes `/packages/` (the Spanish URL is the English package index), `/proof/` (a few Portuguese labels inside an English article), `/playground/`, `/plan/`, `/credits/`, `/dissertation/`, the insights index (it lists English posts only), and the showcases index.

Redirects (`/blog/`, `/docs/**`, `/showcases/<slug>/`, `/registry/**`, `/changelog/`, `/manifesto/`, `/roadmap/`, `/platform/`) are not in the sitemap and are not hreflang targets.

Clusters are reciprocal: each self-canonical page emits one `hreflang` per indexable locale (including itself), absolute `https://www.souniolang.org/.../` URLs, plus `x-default` to English. A fallback page emits that same cluster and does not list itself. `<html lang>` is the BCP 47 tag of the URL locale even when the canonical is English.

## Ops / validation

`npm run check:locale-fallback` asserts HTML output: marketing shell pages **omit** the banner for non-English locales; docs routes under `/learn` and `/tutorials` **include** it until the locale is promoted to `fullyLocalizedLocales`.

`npm run check:hreflang` asserts the built HTML and sitemap: reciprocal hreflang, self-canonical translations, English canonicals for fallbacks, trailing-slash absolute URLs, and sitemap `xhtml:link` rows (including `x-default`) matching the page clusters.
