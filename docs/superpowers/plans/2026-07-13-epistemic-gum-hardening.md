<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-13-epistemic-gum-hardening
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-13-epistemic-gum-hardening
-->

# Epistemic / GUM Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `stdlib/epistemic/gum.sio` a genuinely-usable, importable GUM module — a program can `use` it, propagate ISO-GUM uncertainty, and print a correct report — proven by compile-and-run against a published GUM value. No compiler changes.

**Architecture:** The module already imports (via wildcard); document the working idiom, add a print-based `gum_report`, prove the full API runs as native ELF with assertions against a published GUM result, and ship a consumer example + a runnable gate.

**Tech stack:** Sounio (`./bin/souc` → Madaros v0.80.0). Verified working surface only: `print`/`println` (incl. `print(f64)`/`println(f64)`) to stdout, `string`+`str_*`, structs+`impl`, fixed slabs. **`check:OK` ≠ works — every runtime claim must be `souc compile … -o out && ./out`.**

**Compiler workarounds (dispatched: `docs/audit/MADAROS_MULTIMODULE_PRINT_IMPORT_BUGS_2026-07-13.md`):**
- Import stdlib modules with **`use module::*`** (single-symbol `use module::name` fails `E137`).
- In importing programs print floats with **`print(f64)`/`println(f64)`**, never `print_f64` (spurious `E137`).

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
| `stdlib/epistemic/gum.sio` (modify) | Header usage-note; add `gum_report` | 1, 2 |
| `tests/stdlib/epistemic/test_gum_stdlib.sio` (create) | Run-proof: import stdlib gum, assert published values | 3 |
| `examples/epistemic/gum_measurement_chain.sio` (create) | Consumer example using stdlib gum + `gum_report` | 4 |
| `scripts/epistemic_gum_gate.sh` (create) | Compile+run gate | 5 |

---

## Task 1: Verify `epistemic::gum` imports + runs; document the idiom

**Files:** Modify `stdlib/epistemic/gum.sio` (header comment only). Scratch driver in `$SCRATCH`.

**Context (corrected — spec §2):** `epistemic::gum` is **already importable** via the wildcard form. The earlier "not importable" reading was a misdiagnosis of two compiler bugs (single-symbol import; `print_f64` in importing programs) — both dispatched, both with caller-side workarounds. No visibility edit to `gum.sio` is needed. This task proves import+run and records the idiom in the module header.

- [ ] **Step 1: Prove import + run (in-repo, native ELF).**

Create `examples/_scratch_gum_import.sio`:
```sounio
use epistemic::gum::*
fn main() -> i32 with IO, Mut, Div, Panic {
    let ub = gum_type_b_uniform(0.5)
    let ua = gum_type_a(0.070710, 5)
    let r = gum_combine2(98.3, ua, ub)
    print("u_c=")
    println(gum_std_u(r))     // println(f64) — NOT print_f64
    return 0
}
```
Run: `$SOUC compile examples/_scratch_gum_import.sio -o "$SCRATCH/gi.elf" && "$SCRATCH/gi.elf"; echo "exit=$?"`
Expected: prints `u_c=0.29…`, `exit=0`. If it fails, capture the error and report DONE_WITH_CONCERNS — do not edit `self-hosted/`.

- [ ] **Step 2: Add a usage note to the module header.**

At the top of `stdlib/epistemic/gum.sio`, inside the existing comment block, add:
```sounio
// USAGE: import with `use epistemic::gum::*` (single-symbol `use epistemic::gum::name`
// currently fails to resolve — Madaros bug, docs/audit/MADAROS_MULTIMODULE_PRINT_IMPORT_BUGS_2026-07-13.md).
// Print floats with print(x)/println(x); do NOT use print_f64 in an importing program.
```
Change no code and no signature.

- [ ] **Step 3: Self-check stays green.**

Run: `$SOUC check stdlib/epistemic/gum.sio`
Expected: `check: OK`.

- [ ] **Step 4: Clean up scratch + commit.**
```bash
rm -f examples/_scratch_gum_import.sio
git add stdlib/epistemic/gum.sio
git commit -m "docs(epistemic): document gum import + float-print idiom (Madaros workarounds)"
```

## Task 2: Add `gum_report` formatted stdout

**Files:** Modify `stdlib/epistemic/gum.sio`. Depends on Task 1.

- [ ] **Step 1: Write the report function.**

Append to `stdlib/epistemic/gum.sio` (after the accessors, before any `main`). Floats print via `print(f64)` — never `print_f64`:
```sounio
pub fn gum_report(label: string, unit: string, r: GUMResult) with IO, Mut, Div, Panic {
    print(label)
    print(" = ")
    print(gum_value(r))
    print(" ± ")
    print(gum_u95(r))
    print(" ")
    print(unit)
    print("   (k = ")
    print(gum_k95(r))
    print(", 95%)\n    u_c = ")
    print(gum_std_u(r))
    print(" ")
    print(unit)
    print(",  nu_eff = ")
    print(gum_dof(r))
    print("\n")
}
```
> If `print(f64)` inline does not compile inside `gum.sio` itself (single-module context may differ), fall back to `print_f64` **inside gum.sio only** — the multi-module bug affects importing callers, and `gum.sio` compiled standalone may accept `print_f64`. Verify by the Step 3 run (which imports it).

- [ ] **Step 2: Self-check stays green.**

Run: `$SOUC check stdlib/epistemic/gum.sio`
Expected: `check: OK`.

- [ ] **Step 3: Smoke via an importing driver, compile+run.**

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
Run: `$SOUC compile examples/_scratch_report.sio -o "$SCRATCH/rep.elf" && "$SCRATCH/rep.elf"; echo "exit=$?"`
Expected: `recovery = 98.3… ± … %   (k = …, 95%)` then `u_c … nu_eff …`, `exit=0`.

- [ ] **Step 4: Clean up + commit.**
```bash
rm -f examples/_scratch_report.sio
git add stdlib/epistemic/gum.sio
git commit -m "feat(epistemic): gum_report — formatted ISO-GUM stdout report"
```

## Task 3: Run-proof driver with published-value assertions

**Files:** Create `tests/stdlib/epistemic/test_gum_stdlib.sio`.

**Context:** Published anchor (reproduced at runtime from `examples/real_world/04_gum_measurement_simple.sio`): Type-B rectangular half-width 0.5 → variance a²/3 = 0.083333, std 0.288675; Type-A s≈0.070710 over n=5; combined at 98.30 → u_c ≈ 0.293914, U(k=2) ≈ 0.587828. Assert on the raw f64 accessors (not printed text) within tolerance.

- [ ] **Step 1: Write the driver (compile-and-run IS the gate).**

Create `tests/stdlib/epistemic/test_gum_stdlib.sio`:
```sounio
use epistemic::gum::*

fn near(a: f64, b: f64, tol: f64) -> bool {
    let d = if a > b { a - b } else { b - a }
    d < tol
}

fn main() -> i32 with IO, Mut, Div, Panic {
    let ub = gum_type_b_uniform(0.5)
    let ua = gum_type_a(0.070710, 5)
    let r = gum_combine2(98.3, ua, ub)

    if !near(gum_value(r), 98.3, 1.0e-6) { print("FAIL value\n"); return 1 }
    if !near(gum_std_u(r), 0.293914, 1.0e-3) { print("FAIL u_c\n"); return 2 }
    if !near(gum_u95(r), 0.587828, 5.0e-2) { print("FAIL U95\n"); return 3 }
    if gum_k95(r) < 1.9 { print("FAIL k95 range\n"); return 4 }

    gum_report("recovery", "%", r)
    print("GUM_STDLIB_OK\n")
    return 0
}
```
> Tolerances: `u_c` tight (1e-3); `U95` looser (5e-2) because `k95` depends on effective dof via Welch–Satterthwaite and the reference used k=2 exactly. If `u_c` matches but `U95` differs because `k95` ≠ 2 at this dof, that is CORRECT GUM behaviour — keep the loose bound; do NOT retrofit tolerances to force a pass.

- [ ] **Step 2: Compile and run (the gate).**

Run: `$SOUC compile tests/stdlib/epistemic/test_gum_stdlib.sio -o "$SCRATCH/tg.elf" && "$SCRATCH/tg.elf"; echo "exit=$?"`
Expected: report line(s) + `GUM_STDLIB_OK`, `exit=0`.
If an assertion fails: do NOT loosen tolerances. Investigate — either the stdlib value is wrong (report it) or the expectation is wrong (fix the derivation, cite it). Report DONE_WITH_CONCERNS if the discrepancy is real.

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

    let u_ref = gum_type_b_expanded(0.5, 2.0)   // certificate ±0.5% expanded, k=2
    let u_res = gum_type_b_uniform(0.05)         // resolution, rectangular half-width 0.05
    let u_rep = gum_type_a(0.12, 8)              // repeatability, s=0.12 over n=8

    let r = gum_combine3(99.4, u_ref, u_res, u_rep)
    gum_report("assay", "%", r)
    return 0
}
```

- [ ] **Step 2: Compile and run.**

Run: `$SOUC compile examples/epistemic/gum_measurement_chain.sio -o "$SCRATCH/chain.elf" && "$SCRATCH/chain.elf"; echo "exit=$?"`
Expected: header + `assay = 99.4… ± … %` report, `exit=0`.

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
- **Spec coverage:** import verify+document (Task 1), `gum_report` (Task 2), run-proof vs published value (Task 3), consumer example (Task 4), runnable gate + vancomycin regression (Task 5). All spec items mapped.
- **`check`≠run enforced:** Tasks 1–5 each require `souc compile … && ./elf`, not just `check`.
- **Compiler workarounds baked in:** wildcard imports; `print(f64)`/`println(f64)` not `print_f64`.
- **No tolerance-retrofit:** Task 3 forbids loosening bounds and explains the correct `k95`/dof behaviour.
- **Consistency:** symbol names (`gum_type_a/_b_uniform/_b_expanded`, `gum_combine2/3`, `gum_value/std_u/u95/k95/dof`, `gum_report`) match `gum.sio`'s verified surface.
