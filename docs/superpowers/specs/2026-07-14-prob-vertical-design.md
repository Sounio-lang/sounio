<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-14-prob-vertical-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-14-prob-vertical-design
-->

# Design — Harden the prob (distributions) vertical

**Status:** approved design, pre-implementation
**Date:** 2026-07-14
**Constraint:** No compiler changes (Madaros owned by CODEX-2). Work in `stdlib/`, `examples/`, `tests/`, `scripts/`.
**Orthography:** EN-UK.

## 1. Why
Fifth application of the playbook (GUM #860, units #873, linalg #892, negative-display fix #900): make an
importable stdlib module run-proven against known values and gated. `prob::distributions` is the healthiest
importable probability API and complements GUM (coverage factors) and stats.

## 2. Verified starting state
- `stdlib/prob/distributions.sio` (397 lines, 18 pub fns, type-checks green) covers **normal**
  (`dist_normal_pdf`, `normal_cdf_full`), **gamma** (`gamma_log_pdf/cdf/mean/variance`), **exponential**
  (`exponential_log_pdf/cdf/mean/quantile`), **uniform** (`uniform_log_pdf/cdf/mean/quantile`), **poisson**
  (`poisson_log_pmf/mean/variance`), **dirichlet** (`dirichlet_log_pdf_2`). It imports `special::gamma`,
  `special::igamma`, `special::erf`.
- **Runs correctly** — verified: pdf(0)=0.398942, cdf(0)=0.5, cdf(1.96)=0.975002, exp_mean(2)=0.5,
  unif_mean=5, pois_var(3)=3 — all textbook.
- **Engine caveat (compiler bug, escalated):** the default Madaros native engine **cannot compile** the
  ~210-function multi-module graph (`imported_simple_ir_emit_failed` → thin-link fail); the **`lean_single`
  engine compiles it** and it runs. Filed as `docs/audit/MADAROS_NATIVE_MULTIMODULE_SCALE_2026-07-14.md`.
  So this vertical's build/gate use `SOUNIO_SOUC_ENGINE=lean_single` (a supported engine, the bootstrap
  seed) + `chmod +x` (lean_single doesn't set the exec bit).
- Other prob files: `epistemic.sio` (5 pub fns, green); most others are self-test/private or fail check —
  left untouched (additive; scope is `distributions.sio`).

## 3. Goal
A program can `use prob::distributions::*`, evaluate pdf/cdf/mean/variance/quantile for the supported
distributions, and get correct values — proven by compile-and-run (under `lean_single`) against
first-principles/known values, gated.

## 4. Scope
### In
1. **Verify + document** the import idiom + the `lean_single` build requirement in the module header.
2. **Run-proof driver** — asserts normal pdf/cdf, exponential mean/cdf, uniform mean/quantile, poisson
   variance against known values.
3. **Consumer example** — a short distribution report.
4. **Gate** using `lean_single` (+ `chmod +x`).
5. **Escalate** the Madaros native multi-module scale bug (audit + issue).
### Out
- No new distributions or algorithms; expose/harden what exists.
- No display helper (distributions return usable scalars; the run-proof + engine-doc is the value).
- No fix to the broken prob files or to `special::*`.
- No compiler edits; output stdout only.
- Math-review of the distribution identities is run and logged (distributions are math).

## 5. Design
### 5.1 Import + engine (item 1)
Header note: `use prob::distributions::*`; **native compile requires
`SOUNIO_SOUC_ENGINE=lean_single`** (Madaros native scale limit, see audit) + `chmod +x`; print floats with
`print`/`println` not `print_f64`; inline logic into `main`.

### 5.2 Run-proof (items 2, 4)
`tests/stdlib/prob/test_prob_stdlib.sio`, inline in `main`, asserts (tol 1e-6; 1e-5 pdf; 1e-3 cdf tail):
- `dist_normal_pdf(0,0,1)` = 1/√(2π) ≈ 0.398942 ; `normal_cdf_full(0,0,1)` = 0.5 ;
  `normal_cdf_full(1.96,0,1)` ≈ 0.975.
- `exponential_mean(2)` = 0.5 ; `exponential_cdf(ln2,1)` = 0.5.
- `uniform_mean(0,10)` = 5 ; `uniform_quantile(0.5,0,10)` = 5.
- `poisson_variance(3)` = 3.
Then `PROB_STDLIB_OK`.

## 6. Module layout
```
stdlib/prob/distributions.sio               (modify: header note only)
tests/stdlib/prob/test_prob_stdlib.sio      (new: run-proof driver)
examples/prob/distribution_report.sio       (new: consumer example)
scripts/prob_gate.sh                         (new: lean_single compile+run gate)
docs/audit/MADAROS_NATIVE_MULTIMODULE_SCALE_2026-07-14.md  (new: dispatch)
```

## 7. Verification
- `souc check stdlib/prob/distributions.sio` green.
- `SOUNIO_SOUC_ENGINE=lean_single souc compile … -o out && chmod +x out && ./out` for driver + example.
- `scripts/prob_gate.sh` → `PROB_GATE_OK`.
- Math-review of the distribution identities logged.

## 8. Success criteria
1. A program `use`s `prob::distributions`, compiles under lean_single, runs, and returns correct values.
2. Run-proof asserts known values and passes.
3. Gate green.
4. No compiler files touched; Madaros native-scale bug escalated.

## 9. Risks
| Risk | Mitigation |
|---|---|
| lean_single output not executable | `chmod +x` in the gate/docs (documented). |
| Reader assumes default Madaros builds it | Header + gate both state the `lean_single` requirement; audit filed. |
| `println` newline differs under lean_single | Run-proof asserts on f64 values + greps a sentinel (newline-agnostic). |
