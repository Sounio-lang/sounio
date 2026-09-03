<!-- docs:meta
topic_id: repo.docs.implementation.d3-residual-typecheck-sprint
authority: historical
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.d3-residual-typecheck-sprint
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# D3 Residual Typecheck Sprint

> **Revision 2026-05-28 (post-D3-A-v2 empirical run)**: the original D3-A
> strategy in this doc (file-wide `pub fn` sweep) was empirically falsified.
> Two pub-sweeps (`ir/lower.sio`, `check/check.sio`) produced **zero**
> cascade delta. Only 1 of 439 E200s is preceded by a non-pub warning — `pub
> fn` is advisory only (see `feedback_sounio_visibility_model`). The real
> lever is **missing-imports + empty-stub population + missing-symbol
> definitions**. D3-A section below has been rewritten accordingly.
>
> **Phase D as it actually landed on main**: struct-of-arrays attempt
> (`OcpConstFoldState`) was REVERTED in `1e29c2545` — Sounio cannot index
> array fields of structs (`state.field[idx]` → "value is not indexable").
> Only the cap-raise survived (Codex's `7d3166367`). Bootstrap fp on main
> is `541bc868140beac2d54da976ba8ea976` (gen3 == gen4), not the earlier
> `baa0c060…` from the pre-revert attempt.

## Goal

Drive the modular compiler entrypoint to **0 real errors** with the cap-raise
binary (`local_cap()` = 2048) Phase D landed on main.

This sprint is the closure of task ledger item #20 ("D3 residual typecheck"),
deleted as future-session scope at the end of the 2026-05-28 modular session.
Phase D's structural job is complete; D3 is the cleanup that consumes the new
capacity.

## Predecessors

- **Phase A** (`777316727`): A1-A4 surface-pubs across hlir/, gpu/mod.sio.
  Now understood as cosmetic — see Empirical Revision note above.
- **Phase B** (`ac936159a`): B1-B5 parser body fixes + `resolve_path` rename.
  The real win was the `use parser::{types,exprs,stmts,patterns}::*` imports
  B2 added — NOT the pub markers.
- **Phase A-ext** (`1bff9588f`): A-extended pub sweep on check/ structs +
  ir/ir.sio Ir* structs. The struct pubs DID matter (hard error class) —
  function pubs in same commit were cosmetic.
- **Phase B-ext** (`8bf6d8f96`): `check.sio` body bugs — 21 → 3 phantom via
  missing-import sweep (compat.sio, epistemic.sio, effects.sio). Pattern
  confirmed: imports close cascades, pub markers do not.
- **Phase D struct revert** (`1e29c2545`): removed OcpConstFoldState attempt;
  Sounio struct-array-field-indexing limitation documented.
- **Phase D cap-raise** (`7d3166367`, Codex Review): `local_cap()` 1024 → 2048
  + 138 `[T; 1024]` tables grown to `[T; 2048]`; ~700KB BSS growth.
  Bootstrap fp `541bc868140beac2d54da976ba8ea976` (gen3 == gen4).

Reference commits showing the **real** fix shape: `8b8648b2` (items.sio
113 → 12 via `use` + annotation, NOT pub), `5bba0375` (check phase imports +
type annotations).

## Baseline

After Phase D landed on main, the modular entrypoint reports **898 real
errors** (verified 2026-05-28 against main `03d7c842b`):

- **459 explicit errors** — typecheck failures on identifiers, paths, match
  arms, fields.
- **439 E200 errors** — body-level resolution drift in expression positions.

Top error classes:

| Class | Count |
|---|---:|
| `E200 unknown identifier` | 439 |
| `error: unknown field access` | 162 |
| `error: value is not indexable` | 99 |
| `error: assignment type mismatch` | 84 |
| `error: if condition must be bool` | 34 |
| `error: arithmetic operands must have matching numeric types` | 19 |
| `error: logical and requires bool operands` | 18 |
| `error: comparison operands must have the same type` | 16 |

Top files by standalone error count (verified 2026-05-28):

| File | Errors |
|---|---:|
| `self-hosted/gpu/hlir_to_gpu.sio` | 359 |
| `self-hosted/ir/dce.sio` | 263 |
| `self-hosted/ir/const_prop.sio` | 162 |
| `self-hosted/wasm/lower.sio` | 158 |
| `self-hosted/ir/opt_cleanup.sio` | 100 |
| `self-hosted/wasm/encode.sio` | 85 |
| `self-hosted/check/check.sio` | 21 |
| `self-hosted/native/codegen_x86_linux.sio` | 13 |
| `self-hosted/native/lower_ir.sio` | 9 |
| `self-hosted/ir/lower.sio` | 8 |

### Binary invocation gotcha

The `./bin/souc` shell wrapper on main does NOT dispatch `compile`/`check`
subcommands correctly to the underlying `mini_native` binary, which accepts
only `<src> <out>` 2-arg form. Capture the baseline by invoking the binary
directly:

```bash
./bin/souc-linux-x86_64 self-hosted/compiler/main.sio /tmp/d3_compile_out \
  > /tmp/d3_baseline.log 2>&1
grep -cE '^(error:|E[0-9]+)' /tmp/d3_baseline.log    # → 898
grep -E '^(error:|E[0-9]+)' /tmp/d3_baseline.log | \
  sed -E 's/at line [0-9]+//; s/`[^`]+`/X/g' | \
  sort | uniq -c | sort -rn > /tmp/d3_buckets.txt
```

Per-file checks via the wrapper still work: `./bin/souc check <file>`.

## Plan

Three sub-phases, mirroring the established A → B → B-ext rhythm. Each
sub-phase is one session, max.

### D3-A — Missing imports + missing-symbol fixes (REVISED)

The pub-sweep strategy in this section's earlier version was empirically
falsified by the D3-A v2 trial run (2026-05-28). Two whole-file pub passes
(`ir/lower.sio`, `check/check.sio`) produced zero cascade delta because
`pub fn` is advisory-only. The real leverage points discovered by that
trial are five concrete missing-symbol/missing-import causes:

**1. Missing `use ir::ir::*` in `dce.sio` and `const_prop.sio`** (highest
ROI). Both files use `IrOpcode::*` enum variants and `IrFunction`/`IrInstr`
struct fields with no import. Adding the import closes ~290 standalone
errors and the downstream main.sio cascade.

```sounio
// At top of self-hosted/ir/dce.sio and self-hosted/ir/const_prop.sio
use ir::ir::*
```

**2. Empty stub `self-hosted/wasm/core.sio`** containing only `module
wasm::core`. Importers `wasm/encode.sio` (85 errors) and `wasm/lower.sio`
(158 errors) do `use wasm::core::*` which resolves to an empty namespace,
causing `WASM_TYPE_*`, `WasmLocal` etc. to E200. Fix: populate the stub
with the symbols those importers actually need. Audit referenced names
with `grep -rE '\bWasmLocal|WASM_TYPE_' self-hosted/wasm/{encode,lower}.sio`.

**3. Missing constant `ESPV_OP_BITWISE_OR`** in
`self-hosted/gpu/epistemic_spirv.sio` (only `ESPV_OP_BITWISE_XOR` exists at
line 88). 5 sites in the file reference the missing constant. Add:

```sounio
let ESPV_OP_BITWISE_OR: i64 = 198    // adjacent to BITWISE_XOR=197
```

Note: per `feedback_module_let_constant_import_bug.md`, module-level `let`
constants don't export cleanly across modules. The 5 use sites here are
intra-file, so this works. Cross-module `let`-constant references that
remain in main.sio after D3-A are scope for D3-B.

**4. Missing builtin `f64_to_bits`** referenced in `native/lower_ir.sio`,
`native/machine_ir.sio`, `compiler/main.sio`, `wasm/lower.sio`. Defined
nowhere. Add a definition in a stdlib or native utility file and export it.
Sounio has no `f64.to_bits()` method intrinsic; this is a real missing
symbol. Each call site generates one E200 — count the call sites with
`grep -rn 'f64_to_bits' self-hosted/` and add to the closure count.

**5. Tuple-element bool-binding pattern** triggers
"if condition must be bool" in 43+ sites across `check/check.sio` (18×),
`ir/tailcall.sio` (4×), `ir/opt_cleanup.sio` (21×):

```sounio
// Before — fires error even though pair.0 is declared bool
let is_valid = pair.0
if is_valid { … }

// After — inline avoids the souc binding-loses-bool issue
if pair.0 { … }
```

This is a souc bug (the binding should preserve its type) but the inline
workaround is mechanical and closes 43+ errors.

### D3-A execution order

Tackle by ROI:

1. (1) `dce.sio` + `const_prop.sio` use lines — closes ~290 standalone errors
2. (2) `wasm/core.sio` stub population — closes ~240 wasm standalone errors
3. (5) tuple-bool inline pass — closes ~43 errors
4. (3) `ESPV_OP_BITWISE_OR` constant — closes 5 errors
5. (4) `f64_to_bits` builtin — closes 5-10 errors

Gate after each fix:

```bash
./bin/souc check <file>                           # local clean (no regression)
./bin/souc-linux-x86_64 self-hosted/compiler/main.sio /tmp/m > /tmp/d3.log 2>&1
grep -cE '^(error:|E[0-9]+)' /tmp/d3.log          # → monotone reduction
```

Target: 898 → ≤ 350. Stop when remaining errors are isolated body bugs
(D3-B scope) rather than cascade roots.

**Watch out**: `feedback_module_let_constant_import_bug.md` documents a
35-second stack overflow on explicit `use module::{CONST}` for module-level
`let` constants. For fix (3), keep the constant intra-file. For
cross-module constant references, defer to D3-B (workarounds: inline-define
in callers, or replace `let X: i64 = N` with `pub fn X() -> i64 { N }` +
update call sites).

### D3-B — Body-bug class fixes (parser-style annotations)

For each remaining bucket, apply the `parse_param_list:387` pattern (commit
`8b8648b2`): replace ambiguous bare returns with explicitly annotated locals.

```sounio
// Before
return (self, None)

// After
let none_params: Option<Box<ParamList>> = None
return (self, none_params)
```

Body-bug shapes already catalogued (from `items.sio` post-`8b8648b2`):

- if-arm type mismatches in error-recovery blocks (cf. lines 124, 592, 608)
- unknown field accesses (cf. lines 722, 723, 742)
- initializer type mismatches (cf. lines 993, 1075)
- function-table ambiguity where a local helper shadows a module-level fn
  (cf. `parse_refinement_type` ambiguity, line 2341)

Reference `feedback_souc_line_124_diagnostic_bug.md` to distinguish phantom
from real before editing.

### D3-C — Acceptance gate + commit

1. `./bin/souc compile self-hosted/main.sio` → 0 real errors.
2. `make clean && make build`; gen2 == gen3 (must remain re-blessed; D3 edits
   should not invalidate again — they are surface-only).
3. Re-run modular liveness probes:
   ```bash
   ./bin/souc run tests/modular/probe_full_synthesis.sio
   ./bin/souc run tests/modular/probe_three_phases.sio
   ./bin/souc run tests/modular/probe_two_phases.sio
   ./bin/souc run tests/modular/probe_gum_propagation.sio
   ./bin/souc run tests/modular/probe_ir_data_flow.sio
   ./bin/souc run tests/modular/probe_emit_phase_alive.sio
   ./bin/souc run tests/modular/probe_modular_plus_sota.sio
   ```
4. Commit per file with the established message shape:
   `modular(<dir>): close N <file> typecheck errors via use+annotation`.
5. Emit gate marker `MODULAR_MAIN_TYPECHECK_PASS` via a new
   `scripts/ci/modular_main_typecheck_gate.sh` (gates are emitted by shell
   scripts; they are not registered in `topic-registry.v1.json`).

## Scope boundaries — what D3 is NOT

- **Not D4** (task #21, "acceptance gate"). That's the CI wiring that runs the
  D3 gate marker in `scripts/run_sio_test_suite.sh`. Leave for a separate
  session per the original ledger.
- **Not a logic rewrite of `main.sio`**. If a fix requires changing semantics
  (e.g. removing a local that shadows a function), record it in
  `.claude/pending.md` and skip; that is future Phase E.
- **Not another bootstrap re-bless**. `local_cap` stays at 2048; structural
  state stays as Phase D left it. If you find yourself editing
  `self-hosted/compiler/lean_single.sio:891`, you are outside D3 scope.

## Critical files

| Path | Lines | Role |
|---|---:|---|
| `self-hosted/compiler/main.sio` | 2,464 | D3 primary cascade target; owns the 898 residual errors via imports. |
| `self-hosted/ir/dce.sio` | — | D3-A fix #1: add `use ir::ir::*`; ~263 standalone errors. |
| `self-hosted/ir/const_prop.sio` | — | D3-A fix #1: add `use ir::ir::*`; ~162 standalone errors. |
| `self-hosted/wasm/core.sio` | — | D3-A fix #2: populate empty stub; unblocks wasm/encode + wasm/lower. |
| `self-hosted/gpu/epistemic_spirv.sio` | — | D3-A fix #3: add `ESPV_OP_BITWISE_OR`. |
| `self-hosted/gpu/hlir_to_gpu.sio` | — | 359 standalone errors; investigate in D3-B (mix of body bugs + missing symbols). |
| `self-hosted/ir/opt_cleanup.sio` | — | Phase D revert surface; do not regress. 100 standalone errors (21 tuple-bool sites for D3-A fix #5). |
| `self-hosted/compiler/lean_single.sio` | — | `local_cap()` = 2048; do not edit in D3. |
| `tests/modular/probe_*.sio` | 7 files | Liveness probes — keep all green. |

## End-to-end verification

```bash
make clean && make build                                 # gen3 == gen4 == 541bc868…
./bin/souc-linux-x86_64 self-hosted/compiler/main.sio /tmp/m  # 0 errors
bash scripts/ci/modular_main_typecheck_gate.sh           # MODULAR_MAIN_TYPECHECK_PASS
./bin/souc check self-hosted/check/check.sio             # ≤ 3 phantom (no real regression)
./bin/souc check self-hosted/parser/items.sio            # ≤ 12 (unchanged)
```

## Out-of-band: revisit the deleted ledger items

After D3 ships, re-open task ledger item #21 (D4 acceptance gate). The CI gate
is a one-session wiring task once `MODULAR_MAIN_TYPECHECK_PASS` exists — add
it to `scripts/run_sio_test_suite.sh`'s gate registry and wire its required
inputs.
