# Portuguese Translation Checklist

Maintainer: Demetrios — translate these personally in a later pass.
Current status: all pt routes show the English-first banner. When a page below is translated and verified, mark it `[x]` and, once all are done, add `pt` to `fullyLocalizedLocales` in `src/lib/i18n.ts`.

## UI strings (src/i18n/pt.json)

- [x] Navigation, footer, search, theme — translated (verified in initial pass)
- [x] Epistemic stage descriptions — translated
- [x] `locale_banner_notice` / `locale_banner_link` — added 2026-04-20
- [ ] Review all 165+ keys against the current en.json for drift

## Main pages (Astro components — currently fall back to English)

These pages have no locale-specific body content. To translate, create a `[lang]` version or add locale branching inside the component.

- [ ] `/` — homepage (index.astro)
- [ ] `/language` — language tour with 4-section TOC
- [ ] `/proof` — honest evidence page
- [ ] `/manifesto`
- [ ] `/changelog`
- [ ] `/roadmap`
- [ ] `/playground`
- [ ] `/about/`
- [ ] `/about/vision`
- [ ] `/about/roadmap`

## Learn / docs (content/docs/pt/ — translated but stale)

Files exist in `website/src/content/docs/pt/`. Stale means the English version has been updated since the pt translation was written. Review each file against the English version and update.

### getting-started/

- [ ] `hello-world.mdx` — pt shorter, missing "What this example proves" section
- [ ] `first-program.mdx` — pt shorter, verify against en

### Top-level doc pages

- [ ] `compiler.mdx`
- [ ] `effects.mdx`
- [ ] `epistemic.mdx`
- [ ] `examples.mdx`
- [ ] `feature-status.mdx`
- [ ] `getting-started.mdx`
- [ ] `gpu.mdx`
- [ ] `language.mdx`
- [ ] `spec.mdx`
- [ ] `stdlib.mdx`
- [ ] `stdlib-reference.mdx`
- [ ] `tooling/` (section)
- [ ] `units.mdx`
- [ ] `vancomycin-uncertainty.mdx`

### Section indexes

- [ ] `compiler/` (index + all files)
- [ ] `effects/` (index + all files)
- [ ] `epistemic/` (index + all files)
- [ ] `language/` (index + all files)
- [ ] `spec/` (index + all files)

## Blog, science, insights (dynamic content)

These are English-only at present. No pt content files exist.

- [ ] Decide: translate post-by-post, or mark as English-only permanently
- [ ] If translating: create `content/blog/pt/`, `content/science/pt/`, etc.

## When done

1. Mark all items above `[x]`
2. Do a final diff: `diff website/src/i18n/en.json website/src/i18n/pt.json`
3. Add `'pt'` to `fullyLocalizedLocales` in `website/src/lib/i18n.ts`
4. Build and verify the banner is gone on all pt routes
