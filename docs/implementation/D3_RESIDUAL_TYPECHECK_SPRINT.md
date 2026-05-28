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

## Goal

Drive `./bin/souc compile self-hosted/main.sio` to **0 real errors** on the
`modular/typecheck-drift-clean` branch, after Phase D (`OcpConstFoldState`
refactor + `local_cap` 1024 → 2048) clears the E007 structural blocker.

This sprint is the closure of task ledger item #20 ("D3 residual typecheck"),
deleted as future-session scope at the end of the 2026-05-28 modular session.
Phase D's structural job is complete; D3 is the cleanup that consumes the new
capacity.

## Predecessors

- **Phase A** (`modular/pub-pass`, `7773167`): A1-A4 surface-pubs across hlir/,
  gpu/mod.sio.
- **Phase B** (`modular/body-bugs`, `ac93615`): B1-B5 parser body fixes +
  `resolve_path` rename.
- **Phase A-ext** (`1bff958`): A-extended pub sweep — 113 hard errors cleared
  on `parser/items.sio`.
- **Phase B-ext** (`8bf6d8f`): `check.sio` body bugs — 21 → 3 phantom (zero
  real).
- **Phase D** (`8e1cb0c`, `ef715d2`, `1c44156`): `OcpConstFoldState` struct,
  64-tracker bundling, `local_cap()` raised to 2048. Bootstrap fixed-point
  re-blessed at md5 `baa0c060b7ad3f007f8a0d0176d27b7e` (gen2 == gen3).

Reference commits showing the fix shape D3 will apply at scale:
`8b8648b2` (items.sio 113 → 12), `5bba0375` (check 35 → 0), `f5b6b655`
(import/visibility drift across 5 phase modules).

## Baseline

After phase-d merges into `modular/typecheck-drift-clean`, `souc compile
self-hosted/main.sio` is expected to report **827 real errors**:

- **388 explicit errors** — typecheck failures on identifiers, paths, match
  arms, fields.
- **439 E200 errors** — body-level resolution drift in expression positions
  (same class as the 21 → 3 phantom cluster B-ext closed on `check.sio` and
  the 113 cluster A-ext closed on `items.sio`).

Capture the baseline as the first action:

```bash
./bin/souc compile self-hosted/main.sio 2>&1 | tee /tmp/d3_baseline.log
grep -E '^error\[E[0-9]+\]' /tmp/d3_baseline.log | sort | uniq -c | sort -rn \
  > /tmp/d3_buckets.txt
```

## Plan

Three sub-phases, mirroring the established A → B → B-ext rhythm. Each
sub-phase is one session, max.

### D3-A — Surface-pub + `use` sweep on `main.sio` and direct dependencies

Apply the Phase A pattern: `use parser::types::*`, `use parser::exprs::*`,
`use parser::stmts::*`-style imports and `pub` on any helper referenced
cross-module. Iterate by bucket: expect a few files to dominate (in A-ext,
`items.sio` alone held 113 of ~250 cleared).

Gate after each file:

```bash
./bin/souc check <file>                           # local clean
./bin/souc compile self-hosted/main.sio 2>&1 \
  | grep -c '^error\['                            # monotone reduction
```

Stop the sub-phase when the error count plateaus on `use`/`pub` edits alone —
the tail is body-level (D3-B).

**Watch out**: `feedback_module_let_constant_import_bug.md` documents a
35-second stack overflow on explicit `use module::{CONST}` for module-level
`let` constants. Use the three documented workarounds; do not trigger the bug
path.

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
| `self-hosted/main.sio` | 2,464 | D3 primary target; owns the 827 residual errors. |
| `self-hosted/ir/opt_cleanup.sio` | 7,525 | Phase D refactor surface; do not regress. |
| `self-hosted/compiler/lean_single.sio` | — | `local_cap()` at line 891; do not edit in D3. |
| `self-hosted/check/check.sio` | 16,061 | Cleared to 3 phantom / 0 real by B-ext; gate via `souc check`. |
| `self-hosted/parser/items.sio` | 4,308 | Reference shape for D3-B body fixes. |
| `tests/modular/probe_*.sio` | 7 files | Liveness probes — keep all green. |

## End-to-end verification

```bash
make clean && make build                                 # gen2 == gen3 == baa0c060…
./bin/souc compile self-hosted/main.sio                  # 0 errors
bash scripts/ci/modular_main_typecheck_gate.sh           # MODULAR_MAIN_TYPECHECK_PASS
./bin/souc check self-hosted/check/check.sio             # 0 errors (no regression)
./bin/souc check self-hosted/parser/items.sio            # ≤ 12 (unchanged)
```

## Out-of-band: revisit the deleted ledger items

After D3 ships, re-open task ledger item #21 (D4 acceptance gate). The CI gate
is a one-session wiring task once `MODULAR_MAIN_TYPECHECK_PASS` exists — add
it to `scripts/run_sio_test_suite.sh`'s gate registry and wire its required
inputs.
