<!-- docs:meta
topic_id: repo.docs.audit.madaros-tuple-let-desugar-2026-06-25
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-tuple-let-desugar-2026-06-25
-->

---
docs:meta
title: Madaros tuple-let binding desugar (parser)
date: 2026-06-25
status: fix landed (parse+check verified; value-semantics seed-verified)
scope: self-hosted/parser/stmts.sio
---

# Madaros tuple-destructuring `let (a, b) = e` — binding desugar

## Problem (the dispatch)

The residual self-host typecheck errors after the line-start operator parser fix
(`MADAROS_SELFHOST_TYPEENV_SRET_2026-06-25.md`) included a distinct class:
tuple-destructuring `let (a, b) = e` did not bind its pattern variables, so every
later use of `a`/`b` reported E137 (use of undeclared variable). Concentrated in
the GPU SPIR-V modules (`gpu/epistemic_spirv.sio` etc.), which thread state with
`let (idsN, x) = spv_alloc_id(s.ids)`.

## Root cause

`Stmt` (parser/ast.sio:330) carries a single `name: Name` — it cannot represent a
tuple pattern. `parse_let_stmt` parsed the pattern but only set `name` when
`pat.kind == PatBinding`; for a tuple pattern the parsed pattern was **discarded**
and `name` stayed `empty_name()`. So `let (a, b) = e` produced one `StmtLet` with
an empty name and the initializer — `a` and `b` were never recorded.

## Fix (parser-only desugar)

`self-hosted/parser/stmts.sio`. `let (a, b) = e` desugars to:

```
let __tupN = e      // primary StmtLet, returned by parse_let_stmt
let a = __tupN.0     // pushed to TPLET_PENDING
let b = __tupN.1     // pushed to TPLET_PENDING
```

drained by `parse_block` in source order. `N` is a per-parse counter so multiple
tuple-lets in one block (the GPU pattern) get unique temps. Wildcards (`_`) are
skipped (no binding emitted).

### Why desugar, not a `pat` field on `Stmt`

Adding `pat: Option<Box<Pattern>>` to `Stmt` and binding it in the checker would
clear E137 but **break codegen**: `lower.sio` reads `s.name`/`s.expr`, so `a`/`b`
would typecheck yet never destructure into IR — a green metric hiding a build
break. The desugar instead reuses the fully-proven single-name `let` and tuple
`.N` indexing paths on **both** the check and codegen sides, so it is correct
regardless of which checker let-path runs and through lowering. Zero `Stmt`/
checker changes.

### Mechanism notes

- Helpers are free functions defined before `impl Parser` to satisfy
  define-before-use without depending on `stmt_list_append` /
  `reverse_stmt_list_opt` (which appear later in the file).
- `TPLET_PENDING` is a LIFO cons; `tplet_emit_elems` recurses on the tail
  **before** pushing the head, so the pending list reads head→tail in source
  order (FIFO). `parse_block` conses each onto the reverse-built `rev_stmts_opt`,
  so after the block-end reverse the order is `primary, a, b`. Mirrors the
  `reverse_stmt_list_loop` and `LAST_ASSIGN_TARGET` idioms already in the file.
- `current_name()` reads the token's source text, so the real parser already
  produces field name `"0"` for `pair.0`; the synthetic `ExprFieldAccess` built
  here uses the same digit-`Name` representation the checker decodes
  (check.sio:4538 reads `e.name.buf` as ascii digits).

## Verification

- **Build:** seed (lean_single) compiled the full self-hosted source with the
  modified parser → working madaros (fns=9907). A break in non-tuple `let`
  parsing would have failed this build over ~550k LOC.
- **Check (Madaros, fresh build):** literal-RHS tuple-lets typecheck 100% clean
  (E137 eliminated): `let (a,b)=(7,9)`, multiple lets per block, wildcard
  `let (_,v)`, 3-tuple — all 0 errors. Non-tuple `let` unchanged (`check: OK`).
- **Value semantics (lean_single seed, a known-working backend):** the
  hand-written **desugared form** for `let (a,b)=(5,7)` prints **5 then 7** —
  element ordering correct (`a←.0`, `b←.1`, not swapped). This is the load-bearing
  proof that `check`-clean alone cannot give (a swapped desugar would also
  typecheck). The seed itself cannot compile 3-element `.2` (old-engine limit);
  the GPU modules use exclusively 2-tuples, and the 3-tuple path is the same code
  with `idx+1` and is check-clean on the new build.
- **Regression:** 0 error-count regressions across a 25-file run-pass sample
  (artifacts vs new build); the desugar path only fires when a tuple-let
  populated the queue, so non-tuple parsing is behaviour-identical.

## Honest scope — what this does NOT do (next dispatch)

Removing the E137s **unmasks** pre-existing, orthogonal bugs that were hidden
behind them (E000 is a per-module fallback that only prints when no other
diagnostic emitted; the E137s were setting `diag_emitted`, masking these):

1. **Tuple types in function signatures fail typecheck (E000).** An *uncalled*
   `fn pr() -> (i64, i64) { (1,2) }` errors; a tuple param errors. So the GPU
   modules' `spv_alloc_id`-style tuple-returning helpers still fail to typecheck
   after this fix. (A tuple *literal* bound locally — `let p = (1,2)` — is fine.)
2. **`[i8;32]` vs `[i64;32]` array binding mismatches** surface in
   `epistemic_spirv.sio` (2× E001) once the checker reaches past the former
   E137 early-bail. These cannot originate from this diff (it only emits tuple
   `.N` field access); they are real pre-existing errors now reached.
3. **Madaros native run/`println(int)` backend is broken for all programs**
   (SIGSEGV on `println(5+3)` too) — value witnesses here were taken via the
   lean_single seed, not the madaros backend.

**Therefore: the 588 residual E137 are addressed at the binding level, but the
GPU SPIR-V modules do NOT yet fully typecheck or compile** — the tuple-signature
and array-type bugs above are the next dispatch.

## AI disclosure

Diagnosis and fix by AI agent (Claude) under human direction; every claim is
re-runnable against the cited source lines, the fresh build, and the lean_single
seed.
