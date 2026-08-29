<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-19-stats-dist-parity-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-19-stats-dist-parity-design
-->

# stats↔mpmath distribution parity vertical — `stdlib/stats`

**Status:** design (approved 2026-07-19)
**Owner:** Data & Science verticals
**Reuses:** the special-function parity harness merged in PR #1210
(`stdlib/parity/emit.sio`, the bit-exact bridge, the gate/comparator pattern).
See `docs/research/2026-07-19-special-scipy-parity.md`.

## 1. Goal

An honest, reproducible per-function accuracy map of Sounio's **probability
distribution** functions (`stdlib/stats/densities.sio`, `distributions.sio`)
against an arbitrary-precision **mpmath (dps=30)** reference, computed from each
distribution's closed-form definition. Same philosophy as the special vertical:
measure the genuine achieved accuracy vs ground truth, gate on it, and fix any
root-caused defects the map surfaces. mpmath is the reference `scipy.stats` is
itself validated against (scipy is not installed).

## 2. Scope — the catalog (~18–20 functions)

| Distribution | File | Functions | mpmath reference (formula) |
|---|---|---|---|
| normal | densities/distributions | `normal_pdf(x,μ,σ)`, `normal_cdf`/`normal_cdf_at` | pdf `exp(−z²/2)/(σ√(2π))`; cdf `mp.ncdf((x−μ)/σ)` |
| normal quantile | distributions | `inverse_standard_normal_cdf(p)` | `mp.sqrt(2)·mp.erfinv(2p−1)` |
| exponential | densities | `exponential_pdf(x,λ)`, `exponential_cdf` | pdf `λe^{−λx}`; cdf `1−e^{−λx}` |
| gamma | densities | `gamma_pdf(x,k,θ)`, `gamma_cdf` | pdf via `mp.gamma`; cdf `mp.gammainc(k,0,x/θ,regularized=True)` |
| beta | densities/distributions | `beta_pdf(x,a,b)`, `beta_cdf` | pdf via `mp.beta`; cdf `mp.betainc(a,b,0,x,regularized=True)` |
| lognormal | densities | `lognormal_pdf(x,μ,σ)`, `lognormal_cdf` | pdf `exp(−(ln x−μ)²/2σ²)/(xσ√(2π))`; cdf `mp.ncdf((ln x−μ)/σ)` |
| uniform | densities | `uniform_pdf(x,a,b)`, `uniform_cdf` | pdf `1/(b−a)` on [a,b]; cdf `(x−a)/(b−a)` clamped |
| poisson | densities | `poisson_pmf(k,λ)`, `poisson_cdf` | pmf `λ^k e^{−λ}/k!`; cdf `mp.gammainc(k+1,λ,∞,regularized=True)` |
| binomial | densities | `binomial_pmf(k,n,p)`, `binomial_cdf` | pmf `C(n,k)p^k(1−p)^{n−k}`; cdf `mp.betainc(n−k,k+1,0,1−p,regularized=True)` |
| geometric | densities | `geometric_pmf(k,p)` | pmf `(1−p)^{k−1}p` (verify k-convention: from-1 vs from-0) |

Exact names/signatures/parameter conventions are confirmed per file during
implementation (`git grep -nE '^\s*(pub )?fn ' stdlib/stats/<file>.sio`). The
functions in these files are `pub`, but the vertical still runs under
`lean_single` (the bit-exact `f64_to_bits` bridge requires it).

**Out of scope:** hypothesis tests, descriptive stats, survival/hazard,
Bayesian, correlation — the ~430 other stats pubs. Only the distribution
`pdf/cdf/pmf/quantile` surface. `t`/`F`/`χ²` distribution CDFs are not in
`stdlib/stats` (χ² lives in `stdlib/special` as `chi2_cdf`, already mapped in
#1210); if any turn up during implementation, add them, else note their absence.

## 3. Convention checks (WILL bite — verify empirically before trusting a ref)

- **Parameterization:** gamma `(k,θ)` shape-scale vs `(α,β)` shape-rate; exponential
  `λ` rate vs mean; normal/lognormal `σ` std vs variance. Read each Sounio fn and
  match the mpmath formula to ITS convention (compare one known value).
- **Discrete support:** geometric `k` from 1 (`(1−p)^{k−1}p`) vs from 0
  (`(1−p)^k p`); binomial/poisson cdf inclusive of `k`.
- **CDF edge clamping:** uniform/beta outside support → 0/1.

Each mismatch between the Sounio fn and the chosen mpmath ref is a REFERENCE fix
(match the code's convention), NOT a Sounio bug — as in the special vertical.

## 4. Architecture — reuse the special harness

1. **Bit-exact bridge (unchanged):** the Sounio emitter prints IEEE-754 bit
   patterns (`f64_to_bits`, `print_int` builtins) as i64; wire
   `<fn> <nargs> <arg_bits...> <val_bits>`. Bit f64 on LOCALS in `main`, i64 to
   `emit1..emit4` (already in `stdlib/parity/emit.sio` from #1210).
2. **Comparator** `scripts/parity/stats_parity_ref.py` — a parallel copy of the
   special comparator's structure with a distributions `REF` (fn → mpmath-formula
   lambda + gross threshold), the whole-stream tokenizer, `--require-all`
   coverage assertion, and a `--selftest`. mpmath only; no scipy.
3. **Emitters** `tests/parity/stats_parity_<group>.sio` (one per small group of
   distributions to keep imports clean), `use stats::densities::*` etc.
4. **Gate** `scripts/stats_dist_parity_gate.sh` — mirrors
   `special_scipy_parity_gate.sh`: `SOUNIO_SOUC_ENGINE=lean_single`, compile+run
   emitters, pipe to the comparator with `--require-all`, emit
   `STATS_DIST_PARITY_GATE_OK`. Deterministic; dev-tier (needs mpmath).
5. **Report** `docs/research/2026-07-19-stats-dist-parity.md` — the full
   per-function accuracy map, convention notes, any defects found+fixed.

## 5. Tolerance philosophy (same as special)

Measure the genuine `max_rel_err` per function; fail loudly only on gross error
(> 1e-2, likely a bug); calibrate a function's threshold to its real accuracy
only after confirming it's an approximation, never to hide a bug; exact anchors
(e.g. `normal_cdf(μ)=0.5`, `uniform_cdf` midpoints, `poisson_pmf(0,λ)=e^{−λ}`)
held tight. **Fix-as-found:** root-caused ≈1-line defects are fixed in
`stdlib/stats/*.sio` (separate `fix(stats):` commits, re-verified, no regression
in existing `tests/stdlib/stats/**`); non-trivial ones are flagged in the report.

## 6. Point sets

Per function, representative points across the parameter domain: interior
typical, near support boundaries (documented margin), a couple parameter sets per
distribution (e.g. gamma at (k,θ) ∈ {(2,1),(0.5,2),(5,1)}), and known exact
anchors. Discrete: k over 0..n and around the mean. ~8–20 points per function.
Arguments chosen so bit-exact round-trip is exact (any f64 works via `f64_to_bits`).

## 7. Non-goals

- Not the non-distribution stats surface (tests, descriptive, survival, Bayesian).
- Not installing scipy (mpmath is the reference).
- Nothing ships into the prebuilt compiler; stdlib fixes reach users on the
  normal build path.

## 8. Verification of the vertical

- Phase-0 sanity: the reused bridge/emit path still works (a 1-point normal_cdf
  emitter round-trips) under lean_single.
- Gate emits `STATS_DIST_PARITY_GATE_OK`; coverage-complete (`--require-all`);
  deterministic across runs.
- ≥3 exact anchors pass at 1e-12.
- Existing `tests/stdlib/stats/**` for any fix-touched file stay green.
