# Sounio Website

Astro site for `sounio-lang.org` documentation, tutorials, releases, and multilingual navigation.

## Requirements

- Node.js 20+
- npm 10+

## Local Development

```bash
npm install
npm run dev
```

Dev server defaults to `http://localhost:4321`.

## Build and Quality Gates

Run from `website/`:

```bash
npm run build
npm run check:quality
```

`check:quality` is release-blocking and includes:

- docs topic registry parity
- i18n key parity
- Astro build + Pagefind indexing
- route/redirect contracts
- brand asset checks
- navigation integrity (no broken internal links)
- locale fallback and syntax highlight checks

## Route Notes

- Canonical docs and tutorials live under `/learn/**` and `/tutorials/**`.
- Legacy docs routes (`/docs/**`) and selected localized routes redirect to canonical surfaces.
- Localized language prefixes (`/pt`, `/el`, `/zh`, `/ja`, `/es`) are generated under `src/pages/[lang]/`.

## Useful Commands

```bash
npm run dev
npm run build
npm run preview
npm run check:quality
npm run check:nav
```

## Source Layout

- `src/pages/`: Route definitions.
- `src/content/`: Documentation/tutorial content collections.
- `src/layouts/` and `src/components/`: Shared UI and page scaffolding.
- `scripts/`: Site-specific validation scripts used by CI.
