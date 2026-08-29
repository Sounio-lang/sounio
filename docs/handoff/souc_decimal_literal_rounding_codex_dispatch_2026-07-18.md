<!-- docs:meta
topic_id: repo.docs.handoff.souc-decimal-literal-rounding-codex-dispatch-2026-07-18
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.souc-decimal-literal-rounding-codex-dispatch-2026-07-18
-->

# Dispatch to CODEX-2 — souc decimal-literal parser is not correctly-rounded (drifts up to ~17 ULP)

**Date:** 2026-07-18
**Owner:** CODEX-2 (compiler front-end / lexer-numeric conversion; `self-hosted/`)
**Author:** data-science lane (surfaced while building the C4 Arrow bridge)
**Status:** confirmed defect, minimal repro included — front-end, low blast radius, correctness-relevant

---

## TL;DR

Sounio's **decimal-literal → f64** conversion in the front-end is **not correctly-rounded** (not
round-to-nearest-even against the exact decimal value). It is exact for short, normal-magnitude
literals, but drifts by **+1 to +17 ULP** for literals with long mantissas (>~17 significant digits)
or extreme exponents (near `DBL_MAX` / near the subnormal floor). A correctly-rounded parser (big-
integer / Grisu / Ryū-style) would return the IEEE-nearest double for every literal.

This matters because it is upstream of *every* float constant in the language: for a correctness-first
data stack, `let x = 1.23456789012345678` should be the nearest double, and it currently is not.

## Repro (exact, reproducible now)

`f64_to_bits(<literal>)` compiled under either engine, compared to CPython's correctly-rounded
`struct.pack('<d', float(s))`:

| literal | souc bits | CPython (IEEE-nearest) bits | drift |
|---|---|---|---|
| `0.1` | 4591870180066957722 | 4591870180066957722 | **0 (exact)** |
| `3.14` | 4614253070214989087 | 4614253070214989087 | **0 (exact)** |
| `0.333333333333333` | 4599676419421066575 | 4599676419421066575 | **0 (exact)** |
| `2.718281828459045` | 4613303445314885481 | 4613303445314885481 | **0 (exact)** |
| `1.23456789012345678` | 4608238818662570492 | 4608238818662570491 | **+1 ULP** |
| `6.022e23` | 4962933069378480660 | 4962933069378480659 | **+1 ULP** |
| `1.7976931348623157e308` | 9218868437227405305 | 9218868437227405311 | **−6 ULP** |
| `5e-300` | 129137261445328960 | 129137261445328943 | **+17 ULP** |

Repro program (single file, either engine):
```
fn line(name: string, x: f64) with IO, Mut, Panic, Div { print(name); print(" "); print(f64_to_bits(x)); print("\n") }
fn main() -> i32 with IO, Mut, Panic, Div {
    line("1.23456789012345678", 1.23456789012345678)
    line("1.7976931348623157e308", 1.7976931348623157e308)
    line("5e-300", 5e-300)
    return 0
}
```
Oracle (Python): `import struct; struct.unpack('<q', struct.pack('<d', float(s)))[0]`.

## Root-cause hypothesis

The pattern — exact for short values, drifting for many-digit mantissas and extreme exponents — is the
signature of a **naive `mantissa * 10^exp` (or repeated `*10` / `/10`) float evaluation** in the
lexer's numeric path, where each floating multiply rounds, and the errors accumulate/compound for long
digit strings and large `|exp|`. A correctly-rounded conversion parses the full decimal into an exact
big integer (or uses a proven shortest-round-trip algorithm) and rounds once, at the end.

CODEX-2 to confirm the exact conversion routine (front-end lexer / literal folding, likely in the
tokenizer or a `parse_float`-style helper) — this dispatch does not touch `self-hosted/`.

## The ask

Replace the decimal-literal → f64 conversion with a **correctly-rounded** one (round-to-nearest-even
against the exact decimal): a big-integer scaling approach is the simplest provably-correct option, or
adopt a known algorithm (Clinger/Gay, Grisu3, Ryū). Same for `f32` if it shares the path.

## Acceptance

- The eight literals above all match CPython's `struct.pack('<d')` bits exactly (0 ULP).
- A fuzz set of random decimal strings (varied digit counts and exponents) round-trips to the
  IEEE-nearest double for 100% of cases.

## Scope / impact

- **Low blast radius:** front-end only; does not affect the runtime `f64_to_bits` intrinsic (which is
  a correct IEEE reinterpret) nor the round-trip fidelity of already-parsed values.
- **Correctness relevance:** this is the front-end analogue of the exact-arithmetic lane's guarantees —
  `data::bigrat::bigrat_from_decimal` already parses decimal *strings* to exact rationals correctly; the
  *language literal* path should be at least IEEE-correctly-rounded to match.
- Normal-magnitude literals (the common case) are already exact, so this is a precision-edge fix, not a
  pervasive breakage.

## Pointers
- Repro + oracle: this file's table; `scratchpad/parsedrift.sio`.
- Related: `stdlib/data/bigrat.sio::bigrat_from_decimal` (exact decimal→rational, the correct-by-
  construction reference), `stdlib/data/arrow_bridge.sio` (where the drift was noticed — the SIO1
  round-trip is bit-exact for the *runtime* double; this defect is upstream, at parse time).
