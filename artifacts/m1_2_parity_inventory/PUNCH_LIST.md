# M1.2 punch list — Track A → Track N-v2 parity gaps

Generated 2026-04-30 by `scripts/ci/track_a_nv2_parity_inventory.sh` against
two corpora. The TSVs in this directory are the raw evidence; this file is
the prioritized summary used to drive M1.2 of the three-track convergence
plan.

## Headline numbers

| Corpus | Files | ok | nv2_compile fail | nv2_run | a_bug | both_fail |
|---|---|---|---|---|---|---|
| `tests/run-pass` | 392 | 50 (12.7%) | **324** | 11 | — | 6 |
| broader (examples + tests/native* + selfhost-driver-output) | 841 | 72 (8.6%) | **416** | 34 | 10 | 309 |

Headline: **N-v2 currently accepts roughly 1 in 8 programs that Track A
accepts.** The gap is real but the failure modes cluster into a small
number of structural root-causes.

## Root-cause clusters (sorted by leverage)

### 1. Global identifier resolution (~120+ failures, top priority)

Markers in the cluster output: `kind=123 text=RNG_A` (61), `text=MULT` (9),
`text=Oct` (7), `text=OCT_P` (3), `text=Span` (5), `text=Box` (3),
`text=CovMat` (3), `text=EState` (2), `text=e30/e200/e308` (8+5+4 — these
are **identifiers named like float exponents**, not float literals).

These all hit `unsupported_frontend reason=unresolved_call` or just bare
`kind=123 text=…`. The pattern is: N-v2 cannot resolve a top-level
identifier — likely because it does not yet wire global `let` / `const` /
top-level enum variant references through the same name table that
`lean_single` uses.

**Estimated leverage**: closing this single gap likely turns 100–150
failures into passes, since each failing program tends to reference 1–3
unresolved globals before it gives up.

### 2. `println` builtin (~107 failures)

`kind=123 reason=unresolved_call text=println` — 47 in run-pass, 60 in
broader. `print` works; `println` is just unwired. Trivially fixable
(emit `print(s)` followed by a `\n`).

### 3. `measure` builtin (~44 failures)

`kind=123 reason=unresolved_call text=measure` — appears widely. Likely
a benchmarking macro. Defer for M1.2 if it's macro-heavy; otherwise
implement as a no-op inline.

### 4. Assignment in expression position (~45 failures combined)

`kind=137 text==` (22 in run-pass, 23 in broader). N-v2 likely accepts
top-level statement-position `x = expr` but rejects `=` when it shows up
in expression position (e.g. inside a nested expression, struct literal
field initializer, or subscript LHS). Need to drill in to see what
expressions are blocked.

### 5. `if` as expression (~16 failures combined)

`kind=20 text=if` — 4 in run-pass, 12 in broader. Suggests N-v2 has
`if` as a statement but not always as an expression (e.g. on the RHS of
`let` bindings).

### 6. Operator in non-trivial context (~50 failures combined)

`kind=138 text=<` (17+3), `kind=132 text=+` (5), `kind=133 text=-` (7),
`kind=134 text=*` (4+3), `kind=136 text=%` (3+6), `kind=152 text=<<`
(2+7), `kind=143 text=^` (1), `kind=141/142 text=&/|` (2 each).

These show up at unexpected positions, suggesting expression-parsing
gaps in specific contexts (e.g. inside function-call arguments, inside
recursive-call arguments — the original `fib` failure).

### 7. Field access / indexing in non-trivial position (~22 failures)

`kind=183 text=.` (14+9), `kind=178 text=[` (8). Likely
field-access-after-call or chained subscript expressions that N-v2's
expression grammar does not yet handle.

### 8. `for` / `*` / `(` (~22 failures)

`kind=174 text=(` (12+6), `kind=134 text=*` (4+3), `kind=123 text=for`
(4). Probably tuple expressions, deref operator in some position, and
`for` loops respectively.

### 9. Defer for now (out of M1.2 scope)

- `gpu_thread_id_x` (10) — GPU ops, blocked by larger GPU work.
- `spawn` (6+) — concurrency primitives.
- `heap_alloc` (7) — heap allocation primitives.

## `user_fn_failed` — needs drill-down

22 failures in run-pass, 5 in broader, give only `native_compile:
user_fn_failed` with no further detail. The driver knows a function
failed but suppresses the underlying error. Action: change
`native_compile_driver.sio` to print the inner error — without that
diagnostic these 27 cases are opaque.

## Recommended M1.2 attack order

1. **Improve diagnostics in N-v2 driver** — surface the inner error
   behind `user_fn_failed` (cheap, unblocks visibility).
2. **Wire `println` builtin** — quick win, ~107 cases.
3. **Wire global identifier resolution** — biggest leverage, 100+ cases.
4. **Expression-position `=`, `if`, operators in non-trivial context** —
   structural parser/lower fixes.
5. **`measure` builtin** — likely 1-line stub or skip if macro-heavy.
6. Re-run inventory after each step; track punch-list count downward.

## Re-run

```bash
# Pilot (15 files, ~5s)
bash scripts/ci/track_a_nv2_parity_inventory.sh examples/native

# Full sweep (~830 files, ~8 min)
bash scripts/ci/track_a_nv2_parity_inventory.sh examples tests/run-pass tests/selfhost-driver-output tests/native-v2 tests/native examples/native

# Inspect a single failure
D=$(ls -td /tmp/sounio-parity-inventory.* | head -1)
awk -F'\t' '$8=="nv2_compile" {print $1}' "$D/inventory.tsv" | head
cat "$D/logs/<slug>.nv2.log"
```
