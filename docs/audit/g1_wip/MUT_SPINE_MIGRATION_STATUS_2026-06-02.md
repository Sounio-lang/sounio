# *mut spine migration — status + the Call/MethodCall keystone (2026-06-02)

Toward "the working compiler": eliminating the frame-disease SIGSEGVs in the modular
`mc.elf` `--check` by routing each expression kind off the by-value `check_expr` path
(whose ~164KB Checker self-copies trip a bin/souc large-struct codegen miscompile) onto
the `*mut` spine.

## State (HEAD 2c2968806)

Migrated expr kinds (handler = `checker_check_<X>_inplace`, wired in
`checker_check_expr_inplace`): **IntLit/FloatLit/Bool/String/Char/Return** (leaves) +
**Path, Binary, If, While, Block, ArrayLit, FieldAccess, Index, Match, Unary**.

Crash count (847 examples, rc=139): **481 → 330** across the whole arc. This session:
365 → 330 (+35 via While/Block/ArrayLit/FieldAccess/Index/Match/Unary). 0 regressions, 0
pass→fail at every step; g1 gate green throughout; bin/souc untouched. Enum support is
complete + canonical-faithful (separate arc, `cef81349a`/`3e3a239d2`, probe battery 8/8).

## The pattern (proven, reusable)

Split-and-bridge / full-transcribe: a `*mut` handler checks the kind's SUB-EXPRESSIONS via
the spine (`checker_check_expr_inplace` / `checker_check_opt_expr_inplace` /
`checker_check_block_inplace`), then either FULL-TRANSCRIBES the non-recursive work or
BRIDGES an op-tail (a verbatim extraction of `check_X` taking ALREADY-COMPUTED sub-types,
which never re-checks a sub-expr). Side effects (scope push/pop, error_count, loop_depth,
borrow track, env bind) preserved; same TypeEntry per case; the dispatch owns expr_depth.

## Why the crash count drops SLOWLY — and the keystone

Rescue is **cumulative and non-linear**: a program stops crashing only when its ENTIRE hot
path is `*mut`-covered. The diagnosis (instrument the bridge `checker_check_expr_mut` to
print the kind, histogram the last print before each crash over the 365 crashers) gave:

  While 221 · Call 64 · StructLit 30 · Block 16 · ArrayLit 12 · Unary 5 · Match 3 · …

But migrating While only routed the LOOP STRUCTURE off the by-value path — the
**calls/method-calls inside the loop body still bridge to by-value and crash**. Function
bodies are dominated by CALLS, so:

**The big drop is gated on ExprCall + ExprMethodCall.** Until those are migrated, almost
every crashing body still re-enters by-value `check_expr` through a call argument.

## The Call/MethodCall keystone — NOT YET DONE (needs focused work)

A fan-out workflow designed these but they are NOT integratable as-is:
- **ExprCall** (split-and-bridge, ~246 lines): the design references **4 helpers that do
  not exist** — `call_expr_should_bridge_by_value`, `checker_check_call_args_inplace`,
  `checker_check_expr_list_no_type_inplace`,
  `checker_check_expr_list_no_type_no_linear_consume_inplace`. Its verify agent never
  completed, so it's an unvetted sketch that won't compile.
- **ExprMethodCall** (split-and-bridge, 32-line handler + 139-line tail extraction that
  MODIFIES `check_method_call`): also unverified.

### The correct Call migration (next session, fresh context)
`check_call_expr` (check.sio:13846) has a fan of NON-crash early returns (the ExprIndex
redirect when `e.right=Some`, ~21 builtin-call predicates, the contest-witness predicate).
Plan:
1. Write `call_expr_should_bridge_by_value(e) -> bool` = the disjunction of those early-
   return predicates (they are pure reads of `e.left/e.right/e.args`; the predicate names
   exist — `call_expr_is_builtin_*`, `call_expr_contest_witness_kind`).
2. Write `*mut` arg-list checker(s) that route each arg through `checker_check_expr_inplace`
   faithfully to the by-value arg-checking (unit/refinement/knowledge/borrow boundary +
   linear-consume; the by-value `check_call_arg_*` helpers EXIST and can be bridged per-arg
   on already-computed types).
3. `checker_check_call_expr_inplace`: if `call_expr_should_bridge_by_value(e)` →
   `checker_check_expr_mut(c, e)` (whole-call by value — NOT the crash path). Else: preamble
   (multitest/hypothesis), callee via spine, args via the new `*mut` arg checker, then the
   resolution tail (array-like / TyFn / E009/E010/E038/E039) transcribed in place.
4. Verify: build + full 847 sweep — MUST be 0 pass→fail (the sweep is the faithfulness net;
   calls are everywhere, so any subtle infidelity shows up immediately) — then commit.
   Expect a LARGE crash drop once Call lands (it unblocks the call-bearing loop bodies).
5. Then ExprMethodCall (same shape; its tail extraction modifies `check_method_call`).

After Call+MethodCall: re-run the bridge-histogram diagnosis to find the next long-tail
(StructLit 30, then For/Loop/Cast/Closure/Tuple/Range/Contest/Handle/AuditAttach).

## Process lessons (for the next workflow)
- Workflow agents that share one worktree and have Edit/Write WILL edit the shared file
  concurrently → corruption. FIX (used in batch 2, worked): an ABSOLUTE "return code only,
  do NOT edit files" instruction in the prompt. Integrate from the journal's verified
  `corrected_*` fields, never the live tree; `git checkout` before integrating.
- Agents invent helper names for complex kinds. ALWAYS grep-confirm every referenced
  helper exists before integrating; the build + sweep are the final gate.
- Builds must stay serial (≤2 concurrent souc builds — concurrent builds saturate CPU and
  have crashed the pod). The workflow does DESIGN (no builds); integration is serial.
