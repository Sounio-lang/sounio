# M1.2 Layer B2 — `=` in expression position: NO-OP (cluster already closed)

Date: 2026-04-30
Branch: worktree-agent-aba17c438791e3643 (base 00678f44)

## Finding

The punch list (`artifacts/m1_2_parity_inventory/PUNCH_LIST.md` cluster 4)
claims ~22 (run-pass) and ~23 (broader) failures emit
`unsupported_frontend ... kind=137 text==`. Re-running the inventory
on the current main confirms this cluster is **fully closed**.

## Evidence

Two fresh inventory sweeps were run on commit `00678f44`:

1. `tests/run-pass` (392 files, 281 nv2_compile fails)
   - Inventory dir: `/tmp/sounio-parity-inventory.9AR3Ly`
   - `grep -lE "kind=137" logs/*.nv2.log` → **0 hits**
   - `grep -hE "text==[^=]?$" logs/*.nv2.log` → **0 hits**

2. broader (1235 files, 665 nv2_compile fails) — full sweep per
   PUNCH_LIST §Re-run
   - Inventory dir: `/tmp/sounio-parity-inventory.BdR1gh`
   - `grep -lE "kind=137" logs/*.nv2.log` → **0 hits**

### Bucketed unsupported-token kinds (broader, top-of-log only)

```
    722 kind=123  (unresolved_call / unresolved_ident — cluster 1+3)
     47 kind=183  (.       — cluster 7)
     36 kind=174  ((       — cluster 8)
     25 kind=178  ([       — cluster 7)
     13 kind=133  (-       — cluster 6)
     12 kind=136  (%       — cluster 6)
     11 kind=152  (<<      — cluster 6)
      7 kind=20   (if      — cluster 5)
      7 kind=134  (*       — cluster 6/8)
      6 kind=141  (&       — cluster 6)
      …
      0 kind=137  (= — Layer B2 target — CLOSED)
```

## Likely cause of closure

Two recent commits on main appear to have absorbed the cluster:

- `c91127fd` "feat(nv2): M1.2 step 4 — expression-position if + binop tail-call"
- `36030fc9` "[nv2] M1.2 step 4 — array-init tail-literal handler"

Array-init RHS, `if`-as-expr in let-init, and binop-tail-call are the
three contexts where `=` would have appeared transitively as a parser
recursion target. With those closed, the standalone `=`-token rejection
no longer fires on any file in the corpus.

Subscript-LHS, struct-field-LHS, and `(*p).f =` are already handled in
`parse_stmt_ir` (driver lines 2532, 2660, 2691, 2705, 2737, 2753, 2784).

## Decision (bail-out)

Per the task's bail-out clause ("if you can't safely close even one
sub-context: stop, commit `/tmp/m1_2_assign_expr_blocker.md`"): no edits
to `native_compile_driver.sio` are made. There is no failing case to
mirror lean_single against. Speculative changes to expression-position
assignment handling would risk breaking the self-compile fixed point
without parity benefit.

## Recommended next M1.2 layer

Cluster 1+3 (`kind=123`, 722 hits — global identifier resolution) is the
single highest-leverage gap and is documented in
`artifacts/m1_2_parity_inventory/blockers/m1_2_step_d_user_globals.md`.

Cluster 7 (`kind=183 text=.` field-access-after-call, 47 hits) is the
next clean structural target.

## Punch list update needed

`PUNCH_LIST.md §Root-cause clusters #4` should be marked CLOSED.
