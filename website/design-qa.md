# Living Observatory Design QA

## Comparison Target

- Source visual truth: `/workspace/.home/openvscode-server/.codex/generated_images/019f5998-3207-7113-9727-d7e02fa9c09e/exec-f4089adb-b74b-4a55-8687-73747a3660c5.png`
- Implementation screenshot: `/tmp/sounio-living-observatory-qa/desktop-v4.png`
- Mobile implementation screenshot: `/tmp/sounio-living-observatory-qa/mobile-pt-final.png`
- Power Atlas desktop screenshot:
  `/tmp/sounio-living-observatory-qa/desktop-power-atlas-focused.png`
- Power Atlas mobile screenshot:
  `/tmp/sounio-living-observatory-qa/mobile-pt-power-atlas-final.png`
- Viewports: `1440 x 1024` desktop and `390 x 844` mobile
- State: dark theme, homepage, first Zero Event selected
- Browser-rendered route: local Astro dev server homepage

## Full-View Comparison Evidence

The source and final desktop screenshot were opened together at original
resolution. The implementation preserves the source hierarchy: compact brand
header, literal Sounio H1, left-side introduction, unframed provenance trace,
converged `0.0`, execution receipt, and a visible hint of the next light section.

The implementation intentionally replaces generated mock values with live
checkout truth: Madaros `v0.80.0`, conformance `16 / 16`, commit `635b4cc95`,
verification date `2026-07-13`, and the seven actual Zero Event origins.

No focused crop was required because the 1440 x 1024 source and implementation
made the header, typography, trace labels, receipt, and next-section transition
readable at original resolution. The mobile screenshot was inspected separately
at original resolution for text fit and responsive composition.

## Required Fidelity Surfaces

- Fonts and typography: the implementation uses the existing Iowan/Palatino
  display stack for the Sounio H1 and the existing system mono stack for
  evidence labels. The final 100 px desktop H1 matches the source hierarchy;
  responsive sizes are explicit breakpoints rather than viewport-scaled type.
- Spacing and layout rhythm: desktop hero height is 820 px, leaving 204 px of
  the next section visible at 1024 px. Mobile hero height is 824 px, leaving a
  visible next-section edge at 844 px. The canvas and receipt remain aligned
  without horizontal page overflow.
- Colors and visual tokens: deep brand navy, restrained gold, teal active state,
  blue research states, and warm off-white atlas surface track the source. No
  purple-dominant or single-hue treatment was introduced.
- Image quality and asset fidelity: the header uses the checked-in Sounio logo.
  The source's central visual was implemented as a device-pixel-aware interactive
  canvas with real data rather than flattened into a raster or approximated as
  decorative CSS art. Canvas inspection found 22,482 nontransparent pixels.
- Copy and content: the product name remains the H1. Supporting copy is shorter
  than the source mock and avoids invented claims. English and Portuguese hero
  content fit at the tested breakpoints.
- Power Atlas: the added second-pass section exposes repository scale,
  capability maturity, and recent receipts as an evidence surface rather than
  a marketing grid. It uses the same navy, teal, gold, mono, and display
  hierarchy as the new hero and keeps frontier states visible.

## Comparison History

### Iteration 1

- P1: the seven visible provenance controls were assigned `role=listitem`, which
  suppressed their native button semantics in the accessibility snapshot.
- Fix: changed the structure to a semantic `ul`/`li` containing native buttons.
- Post-fix evidence: browser snapshot exposed all seven buttons with names and
  pressed state.

- P2: on mobile, the visualization retained a 910 px intrinsic width because
  the desktop `align-self:center` rule remained active in the flex layout.
- Fix: set the visual to `width:100%` and `align-self:stretch` below 820 px;
  converted the seven selectors into a horizontal control rail and the receipt
  into a horizontal evidence strip below 520 px.
- Post-fix evidence: `scrollWidth` equals `clientWidth` at 382 px; the visual and
  canvas render at 334 px; the next section begins at y=824 in an 844 px viewport.

### Iteration 2

- P2: the first implementation's 88 px H1 and 789 px hero made the source feel
  less monumental and revealed too much of the next section.
- Fix: increased the desktop H1 to 100 px and stabilized the hero at 820 px.
- Post-fix evidence: `/tmp/sounio-living-observatory-qa/desktop-v4.png` matches
  the source's first-viewport hierarchy while retaining the required next-section
  hint.

### Iteration 3

- P2: the new `#power-atlas` anchor landed partially under the fixed header on
  mobile, hiding the section kicker.
- Fix: added `scroll-margin-top` to the Power Atlas section.
- Post-fix evidence: mobile measurement reports `powerTop: 66` at `390 x 844`.

- P2: the first mobile metric treatment used a horizontal strip, hiding two of
  the four scale numbers until sideways scroll.
- Fix: retained the existing 2x2 metric grid at mobile widths.
- Post-fix evidence: mobile screenshot shows `6,144`, `1,317`, `888`, and `223`
  in the first Power Atlas view, with `scrollWidth` equal to `clientWidth`.

### Iteration 4

- MAJOR from xAI offload review: `real execution` and `every promise` risked
  overclaiming because the public playground remains simulator-backed and some
  surfaces are explicitly preview/frontier.
- Fix: changed hero copy to `Madaros-gated execution receipts`, softened
  `executable science foundry` to `evidence-carrying science foundry`, and
  changed Power Atlas copy from `every promise` to `each listed capability`.
- Post-fix evidence: `npm run check:astro`, `npm run check:brand`, and
  `git diff --check` passed after the wording changes.

## Interaction And Browser Verification

- Page loaded with meaningful content and no framework error overlay.
- Browser console contained no errors.
- All seven provenance controls render as accessible buttons.
- Selecting `Confidence gated` changed `aria-pressed`, the witness to `ZE_GATED`,
  the explanatory text, and the canvas highlight.
- The primary CTA navigated to `#system-atlas` with the atlas aligned at the top.
- Desktop and Portuguese mobile routes were rendered and inspected.
- Desktop and mobile page widths had no horizontal document overflow.
- The animated canvas rendered nonblank at desktop and mobile sizes.
- Power Atlas desktop route rendered without horizontal overflow:
  `scrollWidth: 1432`, `clientWidth: 1432`.
- Power Atlas Portuguese mobile route rendered without horizontal overflow:
  `scrollWidth: 382`, `clientWidth: 382`.
- LLM offload public-copy review: `xai/grok-4.3` returned no BLOCKER and two
  MAJOR wording risks; both were addressed. `deepseek` and `gemini` returned
  provider errors for this fan-out run.

## Findings

No actionable P0, P1, or P2 differences remain.

## Follow-Up Polish

- P3: the source mock has decorative receipt icons; the implementation keeps a
  quieter text-first receipt because the current website has no matching icon
  set enabled for this surface.
- P3: translated hero copy currently falls back to English outside Portuguese;
  full seven-locale editorial translation can follow without changing layout.

## Implementation Checklist

- [x] Match selected first-viewport hierarchy.
- [x] Replace mock values with current repo evidence.
- [x] Implement real interactive Zero Event trace.
- [x] Preserve brand navigation and responsive behavior.
- [x] Keep a hint of the next section visible on desktop and mobile.
- [x] Verify keyboard-readable controls and selected state.
- [x] Check browser console and canvas pixels.
- [x] Pass Astro, route, navigation, locale, docs, and brand checks.

final result: passed
