<!-- docs:meta
topic_id: repo.docs.research.2026-07-19-stats-dist-parity
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.2026-07-19-stats-dist-parity
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# stats↔mpmath distribution parity — `stdlib/stats`

**Status:** complete (2026-07-19)
**Reference:** mpmath 1.3.0 at `mp.dps = 30`, each distribution computed from its
closed-form definition — the arbitrary-precision ground truth `scipy.stats` is
itself validated against (scipy not installed).
**Reuses:** the special-function parity harness (PR #1210) — same bit-exact
`f64_to_bits` bridge, same `stdlib/parity/emit.sio`, same lean_single-locked
gate/comparator pattern. **Spec/plan:** `docs/superpowers/{specs,plans}/2026-07-19-stats-dist-parity*.md`.

## What this is

A per-function accuracy map of Sounio's probability-distribution functions
(`stdlib/stats/densities.sio`, `distributions.sio`) vs arbitrary-precision ground
truth. **19 functions across 9 distributions**, all green under a deterministic,
coverage-complete gate. The screen found and fixed **1 real defect** and surfaced
**1 compiler-infra bug** (worked around).

## The parity map (max relative error vs mpmath, dps=30)

| function | pts | max rel err | | function | pts | max rel err |
|---|---:|---:|---|---|---:|---:|
| normal_pdf | 10 | 3.86e-16 | | uniform_pdf | 10 | 0.00e+00 |
| normal_cdf_at | 5 | 7.07e-08 | | uniform_cdf | 10 | 0.00e+00 |
| standard_normal_cdf | 8 | 3.04e-06 | | poisson_pmf | 10 | 3.34e-15 |
| inverse_standard_normal_cdf | 7 | 2.62e-08 | | poisson_cdf | 10 | 1.42e-12 |
| exponential_pdf | 5 | 5.58e-16 | | binomial_pmf | 8 | 8.45e-15 |
| exponential_cdf | 5 | 0.00e+00 | | binomial_cdf | 8 | 6.26e-15 |
| gamma_pdf | 16 | 2.06e-15 | | geometric_pmf | 8 | 1.11e-15 |
| gamma_cdf | 16 | 8.28e-13 | | lognormal_pdf | 12 | 6.04e-16 |
| beta_pdf | 15 | 5.21e-15 | | lognormal_cdf | 12 | 6.91e-07 |
| beta_cdf | 15 | 8.38e-15 | | | | |

Most functions are at machine precision. The normal-family cdfs
(`standard_normal_cdf` 3e-6, `normal_cdf_at`/`lognormal_cdf` ~7e-8/7e-7) inherit
the accuracy of the underlying erf/Φ approximation — genuine finite-approximation
error, comfortably inside the 1e-2 gross bar (and consistent with the special
vertical's erf-family numbers).

## Defect found and fixed

**`standard_normal_cdf` (`stdlib/stats/distributions.sio`)** applied the
Abramowitz & Stegun 7.1.26 rational erf approximation directly to `|z|` instead of
`|z|/√2` — computing `0.5·(1+erf(|z|))` instead of `Φ(z)=0.5·(1+erf(z/√2))`. This
made the standard normal CDF **~90 % wrong** (max_rel_err **8.97e-01**), a
load-bearing function used throughout the stats module. One-line fix (scale by
1/√2 before the approximation) → **3.04e-06**; no regression in
`tests/stdlib/stats/test_distributions.sio` or `densities.sio`'s inline tests.
Separate commit `fix(stats): scale by 1/sqrt(2) before A&S 7.1.26 erf approx …`.

## Compiler-infra finding (worked around, not a distribution bug)

The **lean_single bundler has a flat symbol table**: compiling a unit that imports
BOTH `stats::densities` and `stats::distributions` breaks, because each module
defines same-named, different-arity `pub fn`s (`normal_pdf`, `normal_cdf`,
`beta_pdf`) and the bundler mis-resolves them across modules (fully-qualified
`module::fn(...)` call syntax is also unsupported). Worked around by splitting the
emitter so each imports a single module (`stats_parity_continuous1.sio` = densities,
`stats_parity_stdnormal.sio` = distributions). No stdlib API change. **This is a
real compiler defect worth a separate fix** — any code needing two modules that
share a bare function name will hit it.

## Convention notes (verified empirically in Phase 0, matched in the reference)

- **gamma is RATE-parameterized:** `gamma_pdf/cdf(x, shape, rate)` = `P(shape, rate·x)`
  (not shape-scale). Confirmed: `gamma_cdf(2,2,2)` = 0.9084 (rate) not 0.2642 (scale).
- **geometric is FROM-0:** `geometric_pmf(k,p) = (1−p)^k·p`, support k=0,1,2,…
  Confirmed: `geometric_pmf(2,0.3)` = 0.147 (from-0) not 0.21 (from-1).
- normal/lognormal `σ` is std-dev; exponential parameter is rate λ; uniform
  clamps pdf→0 outside [a,b], cdf→0/1 — all match the reference.

## Reproduce

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
bash scripts/stats_dist_parity_gate.sh     # → STATS_DIST_PARITY_GATE_OK
```
Requires `mpmath`; SKIPs cleanly if absent. lean_single-locked, `--require-all`
coverage assertion, deterministic. Dev-tier — not wired into `ci.yml`.

## Out of scope

The ~430 non-distribution stats pubs (hypothesis tests, descriptive, survival,
Bayesian, correlation). `t`/`F` distribution cdfs are not in `stdlib/stats` (χ²
lives in `stdlib/special` as `chi2_cdf`, mapped in #1210). The struct-based
`distributions.sio` variants (`normal_pdf(NormalDist,…)`) were skipped in favor of
the scalar `densities.sio` forms. Nothing ships into the prebuilt; the
`standard_normal_cdf` fix reaches users on the normal build path.
