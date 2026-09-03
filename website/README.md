# Sounio Website

Astro 5 site for **https://www.souniolang.org** — documentation, tutorials, blog, and multilingual navigation.

## Requirements

- Node.js **22.12+** (see `package.json` `engines`)
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
- **i18n policy** (which routes show the “translation may lag English” banner): [docs/I18N_POLICY.md](docs/I18N_POLICY.md).

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
- `docs/`: Maintainer-facing notes (e.g. i18n policy).

## Design System & UI Components

The site uses **Tailwind CSS v4** with design tokens in [`src/styles/global.css`](src/styles/global.css) (`@theme` block). Interactive sections use **React islands** (`client:visible` / `client:load`) where needed.

### Color & theme

Dark-first brand navy (`#0B1E3A`), gold accents (`#D6B35A`), glass surfaces — exposed as CSS custom properties and consumed across layouts.

### Typography

[`BaseLayout.astro`](src/layouts/BaseLayout.astro) loads **Google Fonts** (Inter, JetBrains Mono, EB Garamond, GFS Neohellenic) for headings, body, and code. Long-form reading uses sensible system/font stacks from `@theme`.

### Navigation

- Primary chrome: [`Header.astro`](src/components/common/Header.astro) and [`Footer.astro`](src/components/common/Footer.astro) (responsive, keyboard-accessible patterns).
- Theme toggle: [`ThemeToggle.tsx`](src/components/common/ThemeToggle.tsx).

### Other shared components (`src/components/common/`)

Examples: `CustomCursor.tsx` (desktop-only; disabled when `prefers-reduced-motion: reduce`), `KineticText.tsx`, `MCQBlock.tsx`, `AudienceSelector.tsx`, `RenderPreview.astro`.

Homepage demos live under `src/components/home/` (code examples, charts, dissertation viewer, etc.). Legacy WebGPU experiments are archived under `src/components/_archive/`.

### Contributing to styles

1. Prefer extending tokens in `src/styles/global.css` (`@theme` / utility classes).
2. Respect **`prefers-reduced-motion`** for animation-heavy UI (see existing patterns for kinetic text and canvas sections).
3. Test contrast (axe / Lighthouse) and viewports from 320px upward.

## Further Reading

- [Astro documentation](https://docs.astro.build)
- [Tailwind CSS](https://tailwindcss.com)
- [MDX](https://mdxjs.com)
- [Site i18n policy](docs/I18N_POLICY.md)
