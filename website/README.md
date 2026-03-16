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

## Design System & UI Components

The Sounio website follows a consistent design system focused on accessibility, readability, and a scientific aesthetic.

### Color Palette

- **Primary Blue**: `#1a365d` – used for headers, accents, and links.
- **Gold Accent**: `#d4af37` – highlights, buttons, and interactive elements.
- **Orange Secondary**: `#ed8936` – complementary calls‑to‑action.
- **Neutral Grayscale**: A range from `#f9fafb` (light background) to `#111827` (dark text).
- **Dark Theme**: All colors adapt automatically via CSS custom properties (see `src/styles/`).

### Typography

- **Headings**: Inter (sans‑serif) – weights 500‑600 for clear hierarchy.
- **Body Text**: Crimson Pro (serif) – optimized for long‑form reading.
- **Code**: JetBrains Mono – monospaced with syntax highlighting.

### Responsive Navigation

- Desktop: horizontal nav bar with dropdowns (if needed).
- Mobile: hamburger menu that toggles a sliding panel (implemented in `src/components/common/Navigation.astro`).
- The navigation is fully keyboard‑accessible and includes ARIA labels.

### Component Library

Reusable UI components are located in `src/components/common/`:

- `Button.astro` – primary/secondary buttons with hover states.
- `Card.astro` – content cards with shadow and border‑radius.
- `CodeBlock.astro` – syntax‑highlighted code snippets.
- `Nav.astro` – main navigation component.

### Legacy Hugo Site

A previous version of the website built with Hugo remains in the `/hugo` directory. Recent UI improvements there include:

- Enhanced CSS variables in `hugo/static/css/main.css`.
- Mobile‑first responsive breakpoints and a hamburger menu (`hugo/layouts/partials/nav.html`).
- Improved button and card styling with modern shadows and transitions.
- Dark/light theme toggle with persistent user preference.

### Contributing to the Design

When modifying styles, please:

1. Update the design tokens in `src/styles/tokens.css` (Astro) or `hugo/static/css/main.css` (Hugo).
2. Follow the existing naming convention for CSS custom properties.
3. Test color contrast with tools like axe or Lighthouse.
4. Verify responsive behavior on viewports from 320px to 1920px.

## Further Reading

- [Astro documentation](https://docs.astro.build)
- [Tailwind CSS](https://tailwindcss.com) – used for utility‑first styling.
- [MDX](https://mdxjs.com) – for content with interactive components.
