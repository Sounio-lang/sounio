# M1.2 punch list — Track A → Track N-v2 parity gaps

Generated 2026-04-30 by `scripts/ci/track_a_nv2_parity_inventory.sh` against
two corpora. The TSVs in this directory are the raw evidence; this file is
the prioritized summary used to drive M1.2 of the three-track convergence
plan.

## Headline numbers

| Corpus | Files | ok | nv2_compile fail | nv2_run | a_bug | both_fail |
|---|---|---|---|---|---|---|
| `tests/run-pass` (M1.2 baseline) | 392 | 50 (12.7%) | **324** | 11 | — | 6 |
| `tests/run-pass` (after M1.2 step 4 if/binop) | 392 | 63 (16.1%) | — | — | — | — |
| `tests/run-pass` (after M1.2 step 2 redo) | 392 | 74 (18.9%) | 285 | 26 | 1 | 6 |
| `tests/run-pass` (after Layer B1 for-loop, B2 verified-noop, B3 reverted) | 392 | **75 (19.1%)** | **279** | 33 | 1 | 4 |
| broader (examples + tests/native* + selfhost-driver-output) | 841 | 72 (8.6%) | **416** | 34 | 10 | 309 |

Headline: after step 2 (`println` redo) + step 4 partial (`if`/binop-tail),
**N-v2 accepts roughly 1 in 5 `tests/run-pass` programs** (up from 1 in 8
pre-M1.2). `println` cluster in run-pass went from ~47 → 1 remaining (the
surviving case uses `println("lit" ++ msg)` — concat `++`, blocked by a
different cluster, not println itself). Broader corpus inventory was not
re-run in step 2; see "Re-run" below.

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

### 4. Array-init `var arr: [T; N] = [...]` — PARTIALLY CLOSED (step 4 bisect, 2026-04-30)

Landed in `parse_stmt_ir` (TK_VAR/TK_LET array-decl branch): after the
zero-fill, if `= [` follows the type, parse the RHS as either
`[val; N]` (single element copied to every slot) or `[v0, v1, ...]`
(comma-separated per-slot expressions). Each element is parsed through
`parse_expr_ir` with `V2_LAST_STRUCT_IDX` reset on entry and exit.
Stage1-smoke activated via `examples/native/array_init_tail.sio` and
stays green through the bisect. See
[blockers/m1_2_step4_array_init.md](blockers/m1_2_step4_array_init.md)
for the spin-hang history and the bisect outcome note. Remaining work:
top-level globals of the form `var NAME: [T; N] = [0; 256]` still fall
through the non-driver global-registration path (separate task).

### 2. `println` builtin — CLOSED (step 2 redo, 2026-04-30)

Landed as `V2_BUILTIN_PRINTLN = 18` with parse-time expansion in
`parse_fn_call_ir`. Dispatch is based on the argument's runtime type:
f64 register → `print_f64 + '\n'`, leading `TK_STRING` token → `print +
'\n'`, otherwise `print_int + '\n'`. Zero-arg `println()` emits just a
newline. Self-compile gate (with new stage1-smoke) stays green. Post-fix
run-pass inventory: 47 → 1 remaining (surviving case uses `++` string
concat, which is a different cluster).

### 3. `measure` builtin (~44 failures)

`kind=123 reason=unresolved_call text=measure` — appears widely. Likely
a benchmarking macro. Defer for M1.2 if it's macro-heavy; otherwise
implement as a no-op inline.

### 4. Assignment in expression position — CLOSED (Layer B2 verification, 2026-04-30)

Re-verified at commit `00678f44`: zero `kind=137 text==` failures across
run-pass (392 files) and broader (1235 files) corpora. The original
~45-case estimate was stale — all sub-contexts (subscript LHS,
struct-field-LHS, `(*p).f =`) are absorbed by `c91127fd`
(expression-position `if` + binop tail-call) and `36030fc9` (array-init
tail-literal). See
[blockers/m1_2_layer_b2_assign_expr_noop.md](blockers/m1_2_layer_b2_assign_expr_noop.md).

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

- **`for` (4 cases) — CLOSED** by `870e7b66` (Layer B1, 2026-04-30):
  added `parse_for_ir` mirroring `parse_while_ir`; lowers `for x in A..B {…}`
  to `var x=A; while x<B {…; x=x+1}` (and `..=` → `<=`). 2 of 4 for-tests
  in run-pass now compile; remaining 2 (`for_in_loops`,
  `while_for_struct_patterns`) fail on unrelated features (array
  iteration, while-let).
- **`kind=174 text=(` (~18)** — tuple expressions / paren in some position. Open.
- **`kind=134 text=*` (~7)** — deref operator. Open.

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
