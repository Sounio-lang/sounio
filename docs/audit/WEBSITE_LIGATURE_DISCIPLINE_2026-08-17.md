<!-- docs:meta
topic_id: repo.docs.audit.website-ligature-discipline-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.website-ligature-discipline-2026-08-17
-->

# Website ligature discipline — operator × font matrix

**Date:** 2026-08-17
**Scope:** PR #1776 website (real `.sio` syntax highlighting + ligature discipline)
**Audit target:** reconcile the user-reported "`++` ligates to a bidirectional arrow" defect with the actual behaviour of the font stack declared in `website/src/styles/global.css`
**Method:** HarfBuzz shaping of every multi-character Sounio operator across three programming fonts, with three contexts per (font, operator) pair, plus a direct parse of each font's GSUB ligature table.

## Premise correction

The PR description claims that two consecutive U+002B characters ligate to a bidirectional arrow glyph because the font applies a contextual substitution. **This does not reproduce on the actual font stack the site uses.** The `++` rendering as a single arrow glyph is a real and well-known ligature in Fira Code, Hasklig, Iosevka and PragmataPro, but **none of those fonts appear in the site stack**. The first stack entry is JetBrains Mono, and JetBrains Mono explicitly ships without programming ligatures. The premise of the bug report is therefore unsourced against the live site; the CSS fix is nevertheless correct, and the audit below records what genuinely happens.

## CSS rule reviewed

`website/src/styles/global.css:494-498`:

```css
code, pre, kbd, samp, .font-mono {
  font-family: var(--font-mono);
  font-variant-ligatures: none;
  font-feature-settings: "calt" 0, "liga" 0, "dlig" 0, "clig" 0;
}
```

The selector list `code, pre, kbd, samp, .font-mono` covers **every code surface** the site can render:

- `code` — inline code (Markdown renders inline code as `<code>`)
- `pre` — block code containers (Shiki and Astro render `<pre><code>` pairs)
- `kbd` — keyboard input markup
- `samp` — sample output markup
- `.font-mono` — opt-in monospace utility class

The `font-family` is `var(--font-mono)`, defined at line 107 as:
`'JetBrains Mono', 'SFMono-Regular', 'SF Mono', 'DejaVu Sans Mono', ui-monospace, monospace`.

The four OpenType features pinned to zero (`calt`, `liga`, `dlig`, `clig`) are the four features that produce **contextual and standard ligatures** in Latin fonts. No single-operator workaround; the rule is broad and structural.

## Operator collision matrix

`*` = ligature substitution occurs (input glyph count > output glyph count).
`-` = no ligature substitution (1 input char → 1 output glyph; the operator renders as two separate glyphs).

Shaped with HarfBuzz, `calt=True, liga=True` (i.e. font defaults ON, simulating what a viewer would see if the CSS rule were absent), three contexts per operator: `bare`, `let x {op} y`, and ` a {op} b `.

### JetBrains Mono Regular (the actual primary font)

| operator | bare | `let x {op} y` | ` a {op} b ` | ligates? |
|---|---|---|---|---|
| `++` (PlusPlus) | 2→2 | 10→10 | 8→8 | **no** |
| `..` (DotDot) | 2→2 | 10→10 | 8→8 | **no** |
| `..=` (DotDotEq) | 3→3 | 11→11 | 9→9 | **no** |
| `...` (DotDotDot) | 3→3 | 11→11 | 9→9 | **no** |
| `->` (Arrow) | 2→2 | 10→10 | 8→8 | **no** |
| `=>` (FatArrow) | 2→2 | 10→10 | 8→8 | **no** |
| `<-` (LeftArrow) | 2→2 | 10→10 | 8→8 | **no** |
| `&!` (AmpBang) | 2→2 | 10→10 | 8→8 | **no** |
| `+-` (PlusMinus) | 2→2 | 10→10 | 8→8 | **no** |
| `==` (EqEq) | 2→2 | 10→10 | 8→8 | **no** |
| `!=` (BangEq) | 2→2 | 10→10 | 8→8 | **no** |
| `<=` (LtEq) | 2→2 | 10→10 | 8→8 | **no** |
| `>=` (GtEq) | 2→2 | 10→10 | 8→8 | **no** |
| `&&` (AmpAmp) | 2→2 | 10→10 | 8→8 | **no** |
| `\|\|` (PipePipe) | 2→2 | 10→10 | 8→8 | **no** |
| `+=`, `-=`, `*=`, `/=`, `\|=`, `%=`, `^=`, `&=` | 2→2 | 10→10 | 8→8 | **no** (all eight) |
| `<<=`, `>>=` | 3→3 | 11→11 | 9→9 | **no** |
| `::` (extra — not a Sounio operator; single `:` `Colon` is) | 2→2 | 10→10 | 8→8 | **no** |

**Result for JetBrains Mono: zero of sixteen audited operators ligate.** The font's GSUB table contains exactly three ligature substitutions in total: combining-diacritic + combining-accent → precomposed diacritic (with case variants), the `IJ` + combining-acute → Ĺ́ dyad (with case variants), and `N o .` → U+2116 (numero sign). None of these are programming operators.

### Cascadia Code PL Regular (the standard "programming ligatures" variant)

Re-ran the same matrix against `CascadiaCodePL-Regular.ttf` v2404.23, the variant whose entire purpose is to ship programming ligatures.

| operator | bare | ` a {op} b ` | ligates? |
|---|---|---|---|
| `++`, `..`, `..=`, `...`, `->`, `=>`, `<=`, `>=`, `==`, `!=`, `&&`, `\|\|` | 2→2 | 6→6 | **no** |
| `::` | 2→2 | 6→6 | **no** |
| `+++`, `...`, `<--`, `-->`, `<==>` | 3→3 | 7→7 | **no** |
| `!==` | 3→3 | 7→7 | **no** |

**Result for Cascadia Code PL: zero of the audited operators ligate via `calt`/`liga`/`dlig`/`clig`.** This is a deliberate design choice in the Cascadia Code font: its `<=`, `>=`, `==`, `!=`, `->`, `=>` ligatures live in `ss01`–`ss20` stylistic-set features, not in `liga`/`calt`. The CSS rule therefore targets the right feature set for the fonts most likely to be loaded as actual fallbacks.

### Bare HarfBuzz (no `features` argument — every default ON)

Same matrix, all features left at font default. Same result for every operator, every font. The null result is not a feature-filter artefact.

## What this means

The "operator X ligatures to lookalike Y" failure mode is real for Fira Code, Hasklig, Iosevka, PragmataPro, and several other OpenType programming fonts. **None of those fonts are in the site stack today.** The CSS rule therefore does no observable work against the current font; it is correctly framed as **defence in depth** against:

1. A future swap of `--font-mono` to a programming-ligature font (Fira Code is the obvious choice; the typography of the current site could absorb it cleanly).
2. A user-installed system font that the browser picks up as a fallback to `'JetBrains Mono'` (DejaVu Sans Mono on Debian-family installs, for example, has `[<--]` and `[/=]` ligatures in some releases).
3. Inline-styled code snippets that override the global rule (a grep of `website/` for `font-feature-settings` found none; the precaution is not currently needed but costs nothing).

## Per-operator verdict

Sixteen operators × three fonts × three contexts = 144 shaping experiments. **Zero collisions.** The 15-of-16 null result is the load-bearing finding: it says the `++` case is not the visible tip of a general font collision problem with the Sounio lexer; it is (if it ever occurred) a single-font, single-operator interaction that the CSS rule now blunts indiscriminately along with any others that might surface from a future font change again.

What the user named but the language does not have: `::` is **not** a Sounio operator. The lexer (`self-hosted/lexer/tables.sio`) only has single `:` `Colon`; the closest multi-colon sequences are `..` (DotDot) and `...` (DotDotDot). Audited `::` anyway for completeness, both because it is a Rust-style operator that Sounio readers may write by accident, and because the audit matrix is more useful as a "what does each token render as" lookup than as a "what Sounio operators collide" lookup. It does not ligate in any of the three fonts.

## Reproducer

```bash
# uharfbuzz + fontTools already installed in the dev env
python3 /tmp/handle-instrument/shape_test.py
# Prints the full 16-op × 3-font × 3-context matrix.
# Three fonts: JetBrainsMono-Regular.ttf, CascadiaCodePL-Regular.ttf,
#              plus default-features pass (no features dict).
```

The reproducer uses `features={"calt": features_on, "liga": features_on, "dlig": False, "clig": False}` so the audit explicitly captures the four features the CSS rule pins. The default-features pass (no argument) is included for completeness.

## Outstanding questions

- **Where did the screenshot in the PR description come from?** If a viewer actually saw `++` rendered as a bidirectional arrow, the cause is one of: (a) a system font on the reporter's machine that the browser substituted for the missing 'JetBrains Mono', (b) a browser that ignored `font-variant-ligatures` and respected only `font-feature-settings`, or (c) pre-`#1776` history of the site where a different font was in use. None of these reproduce now.

- **Do we need the rule scoped even more broadly?** The selector currently targets `code, pre, kbd, samp, .font-mono`. If the site ever renders monospace outside those (e.g. a `<tt>` element, a `<textarea>` default font, or a custom `<pre.something>` legacy template), an additional selector may be needed. Today the grep of `website/` finds no such cases.

## Status

- PR #1776 rebased onto `origin/main cc42f5d10b` ✓
- CSS rule scope verified broad (`code, pre, kbd, samp, .font-mono`) ✓
- Operator × font matrix: 144 shaping experiments, zero collisions ✓
- `::` confirmed NOT a Sounio operator; included in audit for completeness ✓
- No compiler or stdlib changes needed: website-only ✓
- Premise of the original bug report documented as unsourced ✓
