<!-- docs:meta
topic_id: repo.docs.research.2026-07-19-special-scipy-parity
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.2026-07-19-special-scipy-parity
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# SciPy↔Sounio special-function parity — `stdlib/special`

**Status:** complete (2026-07-19)
**Reference:** mpmath 1.3.0 at `mp.dps = 30` — the arbitrary-precision ground
truth that `scipy.special` is itself validated against (scipy is not installed;
mpmath is a *stronger* reference than scipy's doubles).
**Spec / plan:** `docs/superpowers/specs/2026-07-19-special-scipy-parity-design.md`,
`docs/superpowers/plans/2026-07-19-special-scipy-parity.md`.

## What this is

An honest, reproducible per-function accuracy map of Sounio's special-function
library against arbitrary-precision ground truth. **48 functions across 9
families**, all measured to their genuine achieved relative error vs mpmath, all
green under a deterministic gate. Along the way the parity screen found and
fixed **5 real defects** in `stdlib/special` (see below) — the point of a parity
vertical is exactly this: it turns "we have special functions" into "we know, to
30 digits, how good each one is."

## Method

- **Bit-exact bridge (no precision loss).** A Sounio emitter compiled under
  `lean_single` evaluates each function at a point set and prints, per
  evaluation, the raw IEEE-754 bit patterns (`f64_to_bits`, a compiler builtin)
  of the arguments and the result as signed i64 integers — never `f64 as string`
  (the fragile Madaros path). Wire: `<fn> <nargs> <arg_bits...> <val_bits>`.
- **mpmath comparator** (`scripts/parity/special_parity_ref.py`) reconstructs
  each f64 exactly (`struct.unpack('<d', struct.pack('<q', bits))`), computes the
  mpmath reference at the *exact* same double, and records
  `max_rel_err = max |value − ref| / max(|ref|, 1e-300)` per function.
- **Engine:** the whole vertical is `lean_single`-locked — glob-import into the
  non-`pub` `special::*` families and `f64_to_bits` both require it on default
  Madaros (a compiler-surface artifact, not a math issue).
- **Gate:** `scripts/special_scipy_parity_gate.sh` compiles+runs every family
  emitter, pipes to the comparator with `--require-all` (a coverage assertion:
  any function with no emitted data FAILs, so a missing emitter can't silently
  drop a function from the map), and emits `SPECIAL_SCIPY_PARITY_GATE_OK`.
  Verified **deterministic** (identical numbers across runs).

## The parity map (max relative error vs mpmath, dps=30)

| function | pts | max rel err | | function | pts | max rel err |
|---|---:|---:|---|---|---:|---:|
| erf | 10 | 4.41e-07 | | airy_ai | 6 | 3.01e-14 |
| erfc | 10 | **6.63e-04** | | airy_bi | 6 | 1.60e-15 |
| erfinv | 7 | 2.79e-06 | | airy_ai_prime | 6 | 4.20e-07 |
| normal_cdf | 10 | 3.04e-06 | | airy_bi_prime | 6 | 1.86e-09 |
| normal_quantile | 7 | 6.03e-07 | | riemann_zeta | 6 | 3.69e-16 |
| gamma | 9 | 1.48e-15 | | hurwitz_zeta | 4 | 5.67e-06 |
| lgamma | 5 | 4.16e-16 | | dirichlet_eta | 3 | 4.93e-16 |
| digamma | 11 | 1.71e-11 | | elliptic_k | 5 | 3.90e-16 |
| beta | 5 | 4.79e-15 | | elliptic_e | 5 | 1.90e-16 |
| lbeta | 5 | 1.05e-15 | | hyperg_0f1 | 4 | 4.19e-16 |
| ibeta | 5 | 9.29e-16 | | hyperg_1f1 | 4 | 4.24e-16 |
| ibeta_inv | 3 | 1.30e-15 | | hyperg_u | 3 | 3.91e-06 |
| igamma_lower | 10 | 8.28e-13 | | hyperg_2f1 | 4 | 6.58e-15 |
| igamma_upper | 10 | 1.42e-12 | | legendre_p | 15 | 1.04e-15 |
| chi2_cdf | 6 | 8.28e-13 | | chebyshev_t | 15 | 7.38e-16 |
| bessel_j0 | 5 | 1.58e-08 | | chebyshev_u | 15 | 1.76e-15 |
| bessel_j1 | 5 | 3.92e-09 | | hermite_h | 12 | 0.00e+00 |
| bessel_y0 | 5 | 8.14e-08 | | hermite_he | 12 | 0.00e+00 |
| bessel_y1 | 5 | 2.37e-08 | | laguerre_l | 12 | 1.67e-16 |
| bessel_i0/i1 | 5 | ~2e-08 | | laguerre_l_assoc | 12 | 9.52e-16 |
| bessel_k0/k1 | 5 | ~5e-08 | | jacobi_p | 18 | 0.00e+00 |
| bessel_jn/yn/in/kn | 2-4 | ~1e-08 | | | | |

**Accuracy tiers:** ~half the catalog is at machine precision (1e-15/1e-16, and
the polynomial families hermite/jacobi are bit-exact 0). The incomplete-gamma
family and digamma land ~1e-11/1e-13. The erf family, the bessel functions,
`airy_ai_prime`, `hurwitz_zeta`, and `hyperg_u` are ~1e-6/1e-8 — genuine
finite-approximation accuracy, comfortably inside the 1e-2 gross bar.

**Least-accurate function: `erfc` at 6.6e-04** (~4 correct digits). It passes,
but its approximation is the weakest in the catalog — the natural next
improvement target (a follow-up, not a defect).

## Defects found and fixed (the headline)

All five were root-caused against mpmath and fixed in `stdlib/special`, each in
its own `fix(special): …` commit, each re-verified to pass with no regression in
existing `tests/stdlib/special/**`:

1. **`erfinv`** — the Acklam tail (`|x|>0.9515`) computed `t = sqrt(−ln(p))` but
   the algorithm (and the function's own comment) require `sqrt(−2·ln(p))`.
   Missing factor of 2. **1.33e-2 → 2.79e-6.**
2. **`bessel_y1`** — the x≤8 rational-approximation coefficients were miscopied
   from Numerical Recipes (wrong digits/exponents from the 3rd numerator term).
   `bessel_yn` failed downstream (recurses off Y0/Y1). **~6.4e+2 → 2.37e-8.**
3. **`bessel_in`** — Miller downward-recurrence seeded with `bi = 0.0`, so the
   recurrence stayed identically zero and the final normalization divided by zero
   → **NaN**. Seed `bi = 1.0` (renormalized away). **NaN → 2.35e-8.**
4. **`elliptic_e`** — the AGM correction term `c_{n+1}=(a_n−b_n)/2` used the
   already-updated `b_{n+1}` instead of the pre-update `b_n` (b was overwritten a
   line above). Error grew with k: **~1.0e-1 at k=0.9 → 1.90e-16.** (`elliptic_k`
   never reads `c`, so it was unaffected.)
5. **`hg_gamma` / `hyperg_u`** — `hg_gamma` saturated *any* `x ≤ 0` to a poison
   constant, including negative non-integers where Γ is finite, corrupting
   `hyperg_u`'s connection formula (off ~65% on a normal point, diverging to
   ~5.3e+201 at a pole). Fixed with the reflection formula
   `Γ(x)=π/(sin(πx)·Γ(1−x))` for non-integer `x≤0`, and generalized `hyperg_u`'s
   removable-pole limit from `b==1` to all integer `b`. **~5.3e+201 → 3.91e-6.**

## Convention notes (verified empirically, matched in the reference)

- **elliptic:** Sounio `elliptic_k(k)`/`elliptic_e(k)` take the **modulus k**;
  mpmath `ellipk(m)`/`ellipe(m)` take the **parameter m = k²** → ref uses `k²`.
- **Hermite:** `hermite_h` = physicists' Hₙ (`mp.hermite`); `hermite_he` =
  probabilists' Heₙ = `2^(−n/2)·Hₙ(x/√2)`. (Checked: H₂(1)=2, He₂(1)=0.)
- **Regularized forms:** `ibeta` is the regularized I_x; `igamma_lower/upper`
  are the regularized P/Q; `chi2_cdf(x,k)=P(k/2, x/2)`. Refs match.

## Out of scope

- `caputo.sio`, `mittag_leffler*` (fractional calculus — no standard reference).
- `legendre_q`, `legendre_p_assoc` (not required; all 8 targeted orthopoly
  functions passed — a small extension if wanted later).
- Nothing here ships into the prebuilt compiler; the 5 stdlib fixes are ordinary
  source changes that reach users on the normal build path.

## Reproduce

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
bash scripts/special_scipy_parity_gate.sh    # → SPECIAL_SCIPY_PARITY_GATE_OK
```
Requires `mpmath` (`python3 -c 'import mpmath'`); the gate SKIPs cleanly if it's
absent. Dev-tier — not wired into `ci.yml` (needs mpmath + lean_single).
