# Website i18n policy (souniolang.org)

## Locales

Configured in [astro.config.mjs](../astro.config.mjs) and [src/lib/i18n.ts](../src/lib/i18n.ts): `en` (default, unprefixed), `pt`, `el`, `zh`, `ja`, `es`, `zh-hk`.

## "Fully localized" flag

`fullyLocalizedLocales` in [src/lib/i18n.ts](../src/lib/i18n.ts) is the single switch for treating a locale as complete. Today only `en` is listed. When a locale passes human review (MDX parity with English + UI string drift check), add it to that set; the translation banner stops automatically for that locale.

The public URL prefix is intentionally short (`/pt`), but sitemap and Open Graph locale metadata currently map it to `pt-BR` to match [astro.config.mjs](../astro.config.mjs). Split this into separate `pt-BR` / `pt-PT` locales only when both variants have reviewed copy.

## Translation banner scope

**Problem we solved:** Showing the "not yet translated / English is authoritative" banner on every non-English page (including the homepage) made the product feel broken even when UI strings were translated.

**Current rule:** The banner appears only on routes whose primary content is long-form documentation that may lag English:

- Paths under `/learn/**`
- Paths under `/tutorials/**`

(Including localized prefixes, e.g. `/pt/learn/getting-started`.)

**Banner does not appear on:** Home (`/`, `/pt`, etc.), `/language`, `/proof`, `/manifesto`, `/about`, `/blog`, `/insights`, `/science`, `/packages`, `/playground`, `/changelog`, etc. Those pages may still be English-first in body copy; the UI chrome follows the selected locale.

## Content sync

Translated MDX under `src/content/docs/<locale>/` must be refreshed when English changes. Track drift via `npm run check:docs-parity` and the checklist in [i18n/pt/TRANSLATION_TODO.md](../i18n/pt/TRANSLATION_TODO.md) (other locales: same pattern).

## Ops / validation

`npm run check:locale-fallback` asserts HTML output: marketing shell pages **omit** the banner for non-English locales; docs routes under `/learn` and `/tutorials` **include** it until the locale is promoted to `fullyLocalizedLocales`.
