# Credits and AI-Assistance Disclosure

This document satisfies the GAIDeT framework and ICMJE 2025 authorship guidance for AI tool disclosure.

---

## AI Tool Usage

**Tool:** Claude Code (Anthropic)
**Model:** claude-sonnet-4-6 (Sonnet 4.6)
**Operator:** Demetrios Chiuratto Agourakis (sole maintainer)
**Session dates:** [maintainer to fill in — see git log for branch creation dates]

### Scope of AI assistance

The following website restructuring work was conducted in Claude Code sessions:

| Branch | Description |
|--------|-------------|
| `website-restructure-01-objective-fixes` | Objective corrections pass |
| `website-restructure-02-homepage` | Homepage redesign |
| `website-restructure-03-proof` | Evidence/proof page |
| `website-restructure-04-language-platform-merge` | Merged `/platform` into `/language` with TOC and 308 redirect |
| `website-restructure-05-i18n-audit` | i18n coverage audit; locale banners for 5 non-English locales |
| `website-restructure-06-disclosure-polish` | GAIDeT disclosure, footer link, build polish |

### Nature of AI involvement

- **Drafting and editing:** AI drafted page copy, component markup, and configuration changes under maintainer direction.
- **Code generation:** AI wrote Astro components, CSS, and TypeScript utilities. All changes were reviewed and committed by the maintainer.
- **Translation strings:** AI generated UI string translations for pt, zh, ja, es. Greek (el) strings were deliberately left untranslated pending native speaker review.
- **No autonomous publication:** No content was published without maintainer review and explicit `git push`.

### What AI did not do

- AI did not author the Sounio language specification, compiler, standard library, or scientific claims.
- AI did not generate or modify content in `content/docs/`, `content/blog/`, `content/science/`, or `content/insights/`.
- AI did not make architectural decisions about the language semantics or epistemic type system.

---

## Human authorship

All language design, compiler implementation, scientific research, and strategic direction are solely the work of Demetrios Chiuratto Agourakis.

---

## Third-party assets

| Asset | Source | License |
|-------|--------|---------|
| Sounio stamp / seal images | Original artwork, Sounio project | Apache-2.0 |
| Inter typeface | Rasmus Andersson / Google Fonts | OFL-1.1 |
| JetBrains Mono | JetBrains | OFL-1.1 |
| GitHub mark (SVG) | GitHub, Inc. | [GitHub Logos and Usage](https://github.com/logos) |

---

*Last updated: 2026-04-20*
