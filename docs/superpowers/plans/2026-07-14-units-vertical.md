<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-14-units-vertical
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-14-units-vertical
-->

# Units / Dimensional-Analysis Hardening — Implementation Plan

> **For agentic workers:** execute task-by-task; compile-and-run is the gate (`check:OK` ≠ runs).

**Goal:** Make `stdlib/units/lib.sio` importable + run-proven with a unit-symbol renderer, so a program can do dimension-safe arithmetic with uncertainty and print quantities with their units. No compiler changes.

**Compiler workarounds (dispatched):** wildcard imports (`use units::lib::*`); print floats with `print`/`println` not `print_f64`; inline logic into `main` in importing programs.

**Preamble:**
```bash
cd <worktree>; export SOUNIO_STDLIB_PATH="$PWD/stdlib"; SOUC=./bin/souc
SCRATCH=/tmp/claude-1000/-workspace-sounio/def94f67-8a00-442d-991e-327034e5bf67/scratchpad
```

**Ground rules:** never touch `self-hosted/`/`bootstrap/`; no change to existing `pub` signatures; additive only (leave the 7 satellite files); EN-UK; atomic commits; no AI attribution.

Spec: `docs/superpowers/specs/2026-07-14-units-vertical-design.md`.

## Task 1 — Header note + verify import/run
- [ ] Add a usage note to the top comment block of `stdlib/units/lib.sio` (wildcard import; `print`/`println` not `print_f64`; inline in importing mains; ref the multimodule audit doc).
- [ ] Prove import+run: a scratch `use units::lib::*` main that adds two masses and `println`s the value → exit 0.
- [ ] `$SOUC check stdlib/units/lib.sio` stays green. Commit.

## Task 2 — `dim_show` + `quantity_show`
- [ ] Append to `lib.sio` (after the accessors/conversions, before any `main`):
  - `dim_show(d: UnitDim) with IO` — match `d` (via `dim_eq`) against `dim_force`→`N`, `dim_energy`→`J`, `dim_power`→`W`, `dim_pressure`→`Pa`, `dim_velocity`→`m/s`, `dim_acceleration`→`m/s^2`; else print base symbols with exponents: for each of `(mass,"kg")`,`(length,"m")`,`(time,"s")`,`(temperature,"K")`,`(amount,"mol")`,`(current,"A")`,`(luminosity,"cd")` print positives first then negatives, `sym` for exp 1 and `sym` then `^` then `(exp as i64)` otherwise, a space between factors, omit zero exponents.
  - `quantity_show(label: string, q: Quantity) with IO, Mut, Div, Panic` — `print(label); print(" = "); print(q.value); print(" ± "); print(q.uncertainty); print(" "); dim_show(q.dim); print("\n")`.
- [ ] `$SOUC check stdlib/units/lib.sio` green.
- [ ] Smoke via importing driver: show 19.6 N and 5 kg → compile+run, exit 0. Commit.

## Task 3 — Run-proof driver
- [ ] Create `tests/stdlib/units/test_units_stdlib.sio` (all inline in `main`, wildcard import). Assert first-principles (tol 1e-6 unless noted):
  - add (3±0.4 kg)+(2±0.3 kg): value 5.0, u 0.5, `dim_eq(dim, dim_mass())`.
  - mul (2±0.1 kg)·(9.8 m/s²): value 19.6, u 0.98, `dim_eq(dim, dim_force())`.
  - div (10 m)/(2 s): value 5.0, `dim_eq(dim, dim_velocity())`.
  - scale 2·(3 kg): value 6.0.
  - `quantity_is_compatible(mass, length)` == false.
  - `convert_kg_to_g(2.0)` == 2000.0 ; `convert_celsius_to_kelvin(0.0)` == 273.15 (tol 1e-4).
  - call `quantity_show` once; then `print("UNITS_STDLIB_OK\n")`, return 0.
- [ ] Compile+run → `UNITS_STDLIB_OK`, exit 0. Do NOT retrofit tolerances. Commit.

## Task 4 — Consumer example
- [ ] `examples/units/dimensional_report.sio` — a short dose/rate computation (e.g. mass/time) using `quantity_*` and `quantity_show`, all inline in `main`. Compile+run, exit 0. Commit.

## Task 5 — Gate
- [ ] `scripts/units_gate.sh` — check `lib.sio`; compile+run driver (grep `UNITS_STDLIB_OK`); compile+run example; end `UNITS_GATE_OK`. Run it. Commit.

## Task 6 — Math-review + PR
- [ ] `bin/llm-offload -t math-review -p xai` on the uncertainty/dimension arithmetic + renderer recognition; append to `.claude/llm_offload_log.md`.
- [ ] Register docs: `node scripts/docs/sync_governance_metadata.mjs`; commit governance + new docs.
- [ ] Push; open PR to `main`; ensure `Contracts`/`CI Decision` green; merge.
