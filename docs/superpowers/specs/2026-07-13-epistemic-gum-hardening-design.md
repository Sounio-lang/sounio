# Design — Harden the epistemic / GUM vertical

**Status:** approved design, pre-implementation
**Date:** 2026-07-13
**Constraint:** No compiler changes (Madaros owned by CODEX-2). All work in `stdlib/`, `examples/`, `tests/`, `scripts/`.
**Orthography:** EN-UK.

## 1. Why

The prior "Data & Science I/O" lane was blocked at the compiler (file-I/O builtins broken —
`docs/audit/DATA_IO_LANE_COMPILER_BLOCKER_2026-07-13.md`). We pivoted to a lane that lives entirely
inside the *verified* working surface of current Madaros: `print`/`print_f64` to stdout, the `string`
primitive + `str_*` helpers, fixed-capacity slabs (no heap), structs + `impl`, and pure compute.

Within that surface we harden **one** vertical to genuinely-usable depth: **GUM uncertainty**
(ISO/IEC Guide 98-3) — Sounio's identity feature and dissertation-central (vancomycin PK, calibration).

## 2. Verified starting state

- `stdlib/epistemic/gum.sio` (417 lines) is green and implements GUM properly: `GUMComponent`
  (Type-A `gum_type_a`, Type-B `gum_type_b`/`_uniform`/`_triangular`/`_expanded`, `gum_with_sensitivity`),
  `GUMResult`, Welch–Satterthwaite effective dof (`ws2`), t-tables (`t95`/`t99`), combination
  (`gum_combine2`/`gum_combine3`), propagation (`gum_add`/`sub`/`mul`/`div`/`scale`), and accessors
  (`gum_value`/`gum_std_u`/`gum_u95`/`gum_u99`/`gum_dof`/`gum_k95`).
- The GUM math **compiles to native ELF and runs correctly** (verified: a full Type-A + Type-B report,
  rectangular variance a²/3 confirmed, expanded k=2).
- **Blocker:** `gum.sio` is **not importable** — `use epistemic::gum::gum_type_b_uniform` fails with
  `E137` ("use of undeclared variable"), and `use epistemic::gum::*` fails "visibility preflight". Its
  sibling `stdlib/epistemic/knightian.sio` (same directory, also no `module` decl, also has a `main`)
  imports and runs cleanly via `use epistemic::knightian::*`. So the blocker is **specific to
  `gum.sio`'s content**, not an inherent limitation — a stdlib-fixable issue.
- Consequence today: every green GUM example (`examples/real_world/04_gum_measurement_*.sio`,
  `examples/pbpk_gum_vs_montecarlo.sio`) **reinvents GUM inline** instead of reusing the stdlib.

## 3. Goal

A program can `use` the stdlib GUM module, build Type-A/Type-B components, combine and propagate through
operations, and print a proper ISO-GUM report — all proven by **compile-and-run** with assertions
against a **published** GUM value, plus a runnable gate and a dissertation-aligned example that consumes
the stdlib rather than reinventing it.

## 4. Scope

### In
1. **Importability fix** — make `epistemic::gum` importable + runnable from a standalone in-repo program.
2. **Run-proof driver** — exercises the full public API, asserts against a published GUM textbook result.
3. **Formatted report** — a `gum_report(...)` that prints `y = value ± U (k=…, 95%), u_c=…, ν_eff=…`
   to stdout (print-based; no `string` assembly, sidestepping the `Str`/`string` split).
4. **Runnable gate** — compile+run the driver, assert exit 0 + numeric tolerances; wired like existing gates.
5. **Consumer example** — a dissertation-aligned measurement chain that `use`s stdlib GUM.

### Out
- No fix to `knowledge.sio` (the older `Epistemic` struct path; fails check) — additive, leave it.
- No file/stdin/argv I/O (compiler-blocked). Output is stdout only.
- No new GUM theory; we expose and harden what exists. Missing ops added only if the run-proof needs them.
- No compiler edits. If the importability blocker proves compiler-side (E137 dangling-import is a known
  compiler root cause), record it as a forensic dispatch and take the **fallback**: ship a self-contained
  hardened GUM driver + report + gate (items 2–5) without the re-export, so the vertical still lands.

## 5. Design

### 5.1 Importability (item 1)
Diagnose by bisection from a minimal repro, most-likely-first:
- **Hypothesis A (lead):** `gum.sio`'s `pub fn`s depend on **non-`pub` helpers** (`gum_sqrt`, `t95`,
  `t99`, `ws2`, `gum_result_from_parts`, `dof_to_i64`, `gum_min/max/abs`). If the importer pulls a `pub fn`
  whose body references a private symbol that doesn't resolve across the module boundary, that yields
  exactly `E137 use of undeclared variable`. Fix: make the required helpers `pub` (or inline them), so the
  public surface is self-resolving. This is additive and low-risk.
- **Hypothesis B:** an embedded test `main` or a top-level construct trips "visibility preflight". Test by
  importing after neutralising it. (knightian has a `main` and imports fine, so B is secondary.)
- **Hypothesis C (fallback trigger):** the resolver genuinely can't import this file without a compiler
  fix → dispatch + self-contained fallback.

The fix must keep `gum.sio` self-checking (`souc check stdlib/epistemic/gum.sio` stays green) and must not
change any existing public signature.

### 5.2 Report (item 3)
`gum_report(label: string, unit: string, r: GUMResult) with IO, Mut, Div, Panic` prints, using
`print`/`print_f64`:
```
<label> = <value> ± <U95> <unit>   (k = <k95>, 95%)
    u_c = <std_u> <unit>,  ν_eff = <dof>
```
Pure stdout; no return string. Lives in `gum.sio` (so it's part of the importable surface).

### 5.3 Run-proof (items 2, 4) — the oracle
Assert against a **published** GUM result so "real" is externally defined (repo culture: cross-check vs an
independent oracle). Chosen anchor: **ISO GUM (JCGM 100) Annex H.1** end-gauge calibration, OR the
simpler documented calibration in `examples/real_world/04_gum_measurement_simple.sio` whose expected
numbers (recovery 98.30, u_c 0.2939, U(k=2) 0.5878) we already reproduced at runtime. The driver computes
via **stdlib** `gum_*` and asserts each within tolerance (e.g. |Δ| < 1e-3), returning non-zero on any miss.

### 5.4 Consumer example (item 5)
`examples/epistemic/gum_measurement_chain.sio` — a short, commented measurement chain (e.g. two Type-B
sources + one Type-A, combined and scaled) that imports `epistemic::gum` and calls `gum_report`. Compiles,
runs, prints a clean report. Demonstrates reuse (the anti-reinvention proof).

## 6. Module layout
```
stdlib/epistemic/gum.sio                     (modify: helper visibility fix + gum_report)
tests/stdlib/epistemic/test_gum_stdlib.sio   (new: run-proof driver w/ published-value asserts)
examples/epistemic/gum_measurement_chain.sio (new: consumer example using stdlib gum)
scripts/epistemic_gum_gate.sh                (new: compile+run gate)
```

## 7. Verification
- `souc check stdlib/epistemic/gum.sio` → green (unchanged public API).
- `souc compile tests/stdlib/epistemic/test_gum_stdlib.sio -o out && ./out` → exit 0, published values
  matched within tolerance. **Compile-and-run, never `check` alone** (repo rule: `check:OK` ≠ works).
- `scripts/epistemic_gum_gate.sh` → runs driver + example, both exit 0.
- Regression: `souc check stdlib/clinical/vancomycin_pbpk.sio` still green (shares the epistemic package).

## 8. Success criteria
1. A standalone in-repo program `use`s `epistemic::gum`, runs as native ELF, and prints a correct GUM report.
2. The run-proof asserts against a published GUM value and passes.
3. `gum_report` produces the specified formatted output.
4. The consumer example reuses stdlib GUM (no inline reinvention) and runs.
5. No compiler files touched; vancomycin regression stays green.

## 9. Risks
| Risk | Mitigation |
|---|---|
| Import blocker is compiler-side, not stdlib | Fallback (§4 Out): self-contained driver + dispatch; vertical still ships. |
| `print_f64` formatting too coarse for tolerance asserts | Assert on the raw f64 accessors (`gum_value` etc.) in-program, not on printed text. |
| Making helpers `pub` collides with names elsewhere in the epistemic package | Namespaced `gum_` prefixes already; check for collisions before publishing. |
