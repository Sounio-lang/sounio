# Epistemic / GUM Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `stdlib/epistemic/gum.sio` a genuinely-usable, importable GUM module — a program can `use` it, propagate ISO-GUM uncertainty, and print a correct report — proven by compile-and-run against a published GUM value. No compiler changes.

**Architecture:** Fix the stdlib import blocker so `epistemic::gum` resolves; add a print-based `gum_report`; prove the full API runs as native ELF with assertions against a published GUM result; ship a consumer example + a runnable gate.

**Tech stack:** Sounio (`./bin/souc` → Madaros v0.80.0). Verified working surface only: `print`/`print_f64` stdout, `string`+`str_*`, structs+`impl`, fixed slabs. **`check:OK` ≠ works — every runtime claim must be `souc compile … -o out && ./out`.**

**Preamble — run once per shell:**
```bash
cd /workspace/sounio
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
SOUC=./bin/souc
SCRATCH=/tmp/claude-1000/-workspace-sounio/def94f67-8a00-442d-991e-327034e5bf67/scratchpad
mkdir -p "$SCRATCH"
```

**Ground rules:** Never touch `self-hosted/` or `bootstrap/`. Do not change any existing `pub` signature in `gum.sio`. Do not `git add` unrelated pre-existing WIP. EN-UK; atomic commits; no AI attribution.

**Spec:** `docs/superpowers/specs/2026-07-13-epistemic-gum-hardening-design.md`.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `stdlib/epistemic/gum.sio` (modify) | Publish helpers so the module imports; add `gum_report` | 1, 2 |
| `tests/stdlib/epistemic/test_gum_stdlib.sio` (create) | Run-proof: import stdlib gum, assert published values | 3 |
| `examples/epistemic/gum_measurement_chain.sio` (create) | Consumer example using stdlib gum + `gum_report` | 4 |
| `scripts/epistemic_gum_gate.sh` (create) | Compile+run gate | 5 |

---

## Task 1: Make `epistemic::gum` importable

**Files:** Modify `stdlib/epistemic/gum.sio`. Scratch repro in `$SCRATCH`.

**Context:** `use epistemic::gum::gum_type_b_uniform` currently fails `E137 use of undeclared variable`; `use epistemic::gum::*` fails "visibility preflight". Sibling `knightian.sio` (same dir, no `module` decl, has a `main`) imports fine via `use epistemic::knightian::*`. Lead hypothesis: `gum.sio`'s `pub fn`s call **non-`pub` helpers** (`gum_sqrt`, `gum_abs`, `gum_min`, `gum_max`, `t95`, `t99`, `dof_to_i64`, `gum_result_from_parts`, `ws2`) that don't resolve across the import boundary → `E137`. Fix = publish the helpers the public surface needs.

- [ ] **Step 1: Minimal failing repro (in-repo).**

Create `examples/_scratch_gum_import.sio`:
```sounio
use epistemic::gum::*
fn main() -> i32 with IO, Mut, Div, Panic {
    let ub = gum_type_b_uniform(0.5)
    let ua = gum_type_a(0.070710, 5)
    let r = gum_combine2(98.3, ua, ub)
    print("U95="); print_f64(gum_u95(r)); print("\n")
    return 0
}
```
Run: `$SOUC check examples/_scratch_gum_import.sio`
Expected: FAIL (`E137` and/or "visibility preflight failed"). Record the exact error + byte offset.

- [ ] **Step 2: Confirm the cause is helper visibility.**

Inspect the non-`pub` helpers in `stdlib/epistemic/gum.sio`:
```bash
grep -nE "^\s*fn (gum_sqrt|gum_abs|gum_min|gum_max|t95|t99|dof_to_i64|gum_result_from_parts|ws2)\b" stdlib/epistemic/gum.sio
```
Confirm each is `fn` (not `pub fn`) and is referenced inside a `pub fn` body. If the failing symbol in Step 1 is one of these, Hypothesis A holds → Step 3. If instead the error points at an embedded `main` or a top-level statement, STOP and report DONE_WITH_CONCERNS describing what you found (may be Hypothesis B/C — controller will decide fallback).

- [ ] **Step 3: Check for name collisions before publishing.**

For each helper without a `gum_` prefix (`t95`, `t99`, `dof_to_i64`, `ws2`, `gum_result_from_parts`), verify making it `pub` won't collide across the epistemic package:
```bash
for s in t95 t99 dof_to_i64 ws2 gum_result_from_parts; do echo "== $s =="; git grep -nE "pub fn $s\b" -- 'stdlib/epistemic/*.sio'; done
```
Expected: no existing `pub fn` with these names elsewhere. If a collision exists, rename the `gum.sio` helper with a `gum_` prefix (e.g. `t95`→`gum_t95`) and update its call sites within `gum.sio` only.

- [ ] **Step 4: Publish the helpers.**

In `stdlib/epistemic/gum.sio`, change each required helper from `fn NAME` to `pub fn NAME` (apply any rename decided in Step 3, updating in-file call sites). Do not alter bodies or the already-`pub` API.

- [ ] **Step 5: Module still self-checks.**

Run: `$SOUC check stdlib/epistemic/gum.sio`
Expected: `check: OK`.

- [ ] **Step 6: Import now resolves — compile AND run.**

Run:
```bash
$SOUC compile examples/_scratch_gum_import.sio -o "$SCRATCH/gi.elf" && "$SCRATCH/gi.elf"
```
Expected: prints `U95=<number>` and exits 0. If still failing, the blocker is not helper-visibility → report DONE_WITH_CONCERNS with findings (fallback path).

- [ ] **Step 7: Clean up scratch + commit.**
```bash
rm -f examples/_scratch_gum_import.sio
git add stdlib/epistemic/gum.sio
git commit -m "fix(epistemic): make gum helpers pub so epistemic::gum is importable"
```

## Task 2: Add `gum_report` formatted stdout

**Files:** Modify `stdlib/epistemic/gum.sio`. Depends on Task 1 (module importable).

- [ ] **Step 1: Write the report function.**

Append to `stdlib/epistemic/gum.sio` (after the accessors, before any `main`):
```sounio
pub fn gum_report(label: string, unit: string, r: GUMResult) with IO, Mut, Div, Panic {
    print(label)
    print(" = ")
    print_f64(gum_value(r))
    print(" ± ")
    print_f64(gum_u95(r))
    print(" ")
    print(unit)
    print("   (k = ")
    print_f64(gum_k95(r))
    print(", 95%)\n    u_c = ")
    print_f64(gum_std_u(r))
    print(" ")
    print(unit)
    print(",  nu_eff = ")
    print_f64(gum_dof(r))
    print("\n")
}
```
> If `print`/`print_f64` are not both in scope inside `gum.sio`, mirror how existing functions in the file print (grep `print` in `gum.sio`; the file already prints in its `main`).

- [ ] **Step 2: Self-check stays green.**

Run: `$SOUC check stdlib/epistemic/gum.sio`
Expected: `check: OK`.

- [ ] **Step 3: Smoke it via an in-repo driver, compile+run.**

Create `examples/_scratch_report.sio`:
```sounio
use epistemic::gum::*
fn main() -> i32 with IO, Mut, Div, Panic {
    let ua = gum_type_a(0.070710, 5)
    let ub = gum_type_b_uniform(0.5)
    let r = gum_combine2(98.3, ua, ub)
    gum_report("recovery", "%", r)
    return 0
}
```
Run: `$SOUC compile examples/_scratch_report.sio -o "$SCRATCH/rep.elf" && "$SCRATCH/rep.elf"`
Expected: a formatted line `recovery = 98.3… ± … %   (k = …, 95%)` then the `u_c … nu_eff …` line, exit 0.

- [ ] **Step 4: Clean up + commit.**
```bash
rm -f examples/_scratch_report.sio
git add stdlib/epistemic/gum.sio
git commit -m "feat(epistemic): gum_report — formatted ISO-GUM stdout report"
```

## Task 3: Run-proof driver with published-value assertions

**Files:** Create `tests/stdlib/epistemic/test_gum_stdlib.sio`.

**Context:** Published anchor (already reproduced at runtime from `examples/real_world/04_gum_measurement_simple.sio`): Type-A s/√n with s≈0.070710 over n=5; Type-B rectangular half-width 0.5 → variance a²/3 = 0.083333, std 0.288675; combined at value 98.30 → u_c ≈ 0.293914, U(k=2) ≈ 0.587828. Assert on the raw f64 accessors (not printed text) within tolerance.

- [ ] **Step 1: Write the driver (this is the test — compile-and-run is the gate).**

Create `tests/stdlib/epistemic/test_gum_stdlib.sio`:
```sounio
use epistemic::gum::*

fn near(a: f64, b: f64, tol: f64) -> bool {
    let d = if a > b { a - b } else { b - a }
    d < tol
}

fn main() -> i32 with IO, Mut, Div, Panic {
    // Type-B rectangular, half-width 0.5  -> variance a^2/3
    let ub = gum_type_b_uniform(0.5)
    // Type-A, s = 0.070710 over n = 5
    let ua = gum_type_a(0.070710, 5)
    // Combine at measured value 98.3
    let r = gum_combine2(98.3, ua, ub)

    // Published expectations (ISO GUM combination)
    if !near(gum_value(r), 98.3, 1.0e-6) { print("FAIL value\n"); return 1 }
    if !near(gum_std_u(r), 0.293914, 1.0e-3) { print("FAIL u_c\n"); return 2 }
    // U95 = k95 * u_c ; for large dof k95 -> ~2.0 so U95 ~ 0.5878
    if !near(gum_u95(r), 0.587828, 5.0e-2) { print("FAIL U95\n"); return 3 }
    if gum_k95(r) < 1.9 { print("FAIL k95 range\n"); return 4 }

    gum_report("recovery", "%", r)
    print("GUM_STDLIB_OK\n")
    return 0
}
```
> Tolerances: `u_c` tight (1e-3); `U95` looser (5e-2) because `k95` depends on effective dof via
> Welch–Satterthwaite and the reference used k=2 exactly. If the driver's `u_c` matches but `U95` is off
> because `k95` ≠ 2 at this dof, that is CORRECT GUM behaviour — keep the loose bound; do not retrofit.

- [ ] **Step 2: Compile and run (the gate).**

Run: `$SOUC compile tests/stdlib/epistemic/test_gum_stdlib.sio -o "$SCRATCH/tg.elf" && "$SCRATCH/tg.elf"; echo "exit=$?"`
Expected: report line(s) + `GUM_STDLIB_OK`, `exit=0`.
If an assertion fails: do NOT loosen tolerances to pass. Investigate — either the stdlib value is wrong
(report it) or the expectation/tolerance derivation is wrong (fix the derivation, cite it). Report
DONE_WITH_CONCERNS if the discrepancy is real.

- [ ] **Step 3: Commit.**
```bash
git add tests/stdlib/epistemic/test_gum_stdlib.sio
git commit -m "test(epistemic): run-proof GUM stdlib driver vs published values"
```

## Task 4: Consumer example (reuse, not reinvention)

**Files:** Create `examples/epistemic/gum_measurement_chain.sio`.

- [ ] **Step 1: Write the example.**

Create `examples/epistemic/gum_measurement_chain.sio`:
```sounio
// GUM measurement chain using the stdlib epistemic::gum module.
// Two Type-B sources + one Type-A, combined and reported per ISO GUM.
use epistemic::gum::*

fn main() -> i32 with IO, Mut, Div, Panic {
    print("=== GUM measurement chain (stdlib epistemic::gum) ===\n")

    // Reference standard: certificate ±0.5% expanded, k=2 -> Type-B expanded
    let u_ref = gum_type_b_expanded(0.5, 2.0)
    // Rounding / resolution: rectangular half-width 0.05
    let u_res = gum_type_b_uniform(0.05)
    // Repeatability: s = 0.12 over n = 8 -> Type-A
    let u_rep = gum_type_a(0.12, 8)

    // Combine three components at measured value 99.4
    let r = gum_combine3(99.4, u_ref, u_res, u_rep)
    gum_report("assay", "%", r)
    return 0
}
```

- [ ] **Step 2: Compile and run.**

Run: `$SOUC compile examples/epistemic/gum_measurement_chain.sio -o "$SCRATCH/chain.elf" && "$SCRATCH/chain.elf"; echo "exit=$?"`
Expected: header + a formatted `assay = 99.4… ± … %` report, exit 0.

- [ ] **Step 3: Commit.**
```bash
git add examples/epistemic/gum_measurement_chain.sio
git commit -m "docs(epistemic): example — GUM measurement chain using stdlib gum"
```

## Task 5: Runnable gate

**Files:** Create `scripts/epistemic_gum_gate.sh`.

- [ ] **Step 1: Write the gate.**

Create `scripts/epistemic_gum_gate.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== check gum.sio =="
$SOUC check stdlib/epistemic/gum.sio || fail=1

echo "== run-proof driver =="
if $SOUC compile tests/stdlib/epistemic/test_gum_stdlib.sio -o "$OUT/tg.elf"; then
  "$OUT/tg.elf" | grep -q "GUM_STDLIB_OK" || { echo "FAIL: driver assertions"; fail=1; }
else echo "FAIL: driver compile"; fail=1; fi

echo "== consumer example =="
if $SOUC compile examples/epistemic/gum_measurement_chain.sio -o "$OUT/chain.elf"; then
  "$OUT/chain.elf" >/dev/null || { echo "FAIL: example run"; fail=1; }
else echo "FAIL: example compile"; fail=1; fi

echo "== regression: vancomycin (shares epistemic package) =="
$SOUC check stdlib/clinical/vancomycin_pbpk.sio || { echo "REGRESSION"; fail=1; }

[ $fail -eq 0 ] && echo "EPISTEMIC_GUM_GATE_OK"
exit $fail
```

- [ ] **Step 2: Run it.**

Run: `chmod +x scripts/epistemic_gum_gate.sh && bash scripts/epistemic_gum_gate.sh`
Expected: ends with `EPISTEMIC_GUM_GATE_OK`, exit 0.

- [ ] **Step 3: Commit.**
```bash
git add scripts/epistemic_gum_gate.sh
git commit -m "test(epistemic): compile-and-run gate for the GUM vertical"
```

---

## Self-review
- **Spec coverage:** importability fix (Task 1), `gum_report` (Task 2), run-proof vs published value
  (Task 3), consumer example (Task 4), runnable gate + vancomycin regression (Task 5). All spec items
  mapped.
- **`check`≠run enforced:** Tasks 2–5 each require `souc compile … && ./elf`, not just `check`.
- **Fallback wired:** Task 1 Steps 2/6 route to DONE_WITH_CONCERNS if the blocker isn't helper-visibility,
  so the controller can trigger the self-contained fallback from the spec without a dead end.
- **No tolerance-retrofit:** Task 3 explicitly forbids loosening bounds to pass and explains the correct
  `k95`/dof behaviour.
- **Consistency:** symbol names (`gum_type_a/_b_uniform/_b_expanded`, `gum_combine2/3`, `gum_value/std_u/
  u95/k95/dof`, `gum_report`) match `gum.sio`'s verified surface.
