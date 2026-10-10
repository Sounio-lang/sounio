# Epistemic design system (website)

React components for epistemic honesty surfaces on the public site. They consume
only shared CSS tokens from `website/src/styles/global.css` — no ad hoc hex
colours in component CSS.

## Token triad

Each epistemic verdict maps to a four-part token set:

| Verdict | Tokens |
|---------|--------|
| `TRUSTWORTHY` | `--color-epistemic-verified`, `-surface`, `-border`, `-text` |
| `UNBOUNDED` | `--color-epistemic-uncertain`, `-surface`, `-border`, `-text` |
| `REFUSED` | `--color-epistemic-refused`, `-surface`, `-border`, `-text` |

Shared layout tokens: `--color-surface-primary`, `--color-text-primary`,
`--radius-xl`, `--font-mono`, `--font-sans`.

## EpistemicGateCard

Gate ladder **E0 → E5** with a per-card **ceiling** and **verdict**:

- E0 Syntax · E1 HIR Type · E2 Effect Lattice · E3 GUM Variance Bounds ·
  E4 Lean 4 Formal Proof · E5 Closed-form Theorem

Verdict tokens bind via `data-verdict` on the root — no inline colour styles.

### Sizes

| `size` | Use |
|--------|-----|
| `default` | Playground and standalone exhibits (`/proof`) |
| `compact` | Dense panels (`HonestyArgument`) — hides step labels visually, keeps `aria-label` on each step |

### Accessibility

- Screen-reader summary + `aria-labelledby` on the card
- Ladder steps: `aria-current="step"` on the ceiling gate
- Unreached steps: dashed border (not colour alone)
- External link: `aria-label` includes “opens in new tab”

Used on `/proof` (`EpistemicGateExhibit`) and `/honesty` (`HonestyArgument`).

## Sibling components

- `DiagnosticCodeBlock` — compiler diagnostic with refused styling
- `EffectLatticeBadge` — effect-set chip
- `FabricatedDatum` — fifth epistemic state witness
- `KnowledgeRefusalMeter` — refusal band meter

## Lane

`lane/cursor-1/design-system-epistemic-gate-card-20260824` — token hygiene,
a11y, and exhibit integration for `EpistemicGateCard`.
