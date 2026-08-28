<!-- docs:meta
topic_id: repo.docs.audit.madaros-closures-step2-capture-2026-06-24
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-closures-step2-capture-2026-06-24
-->

# Madaros capturing closures — working (closures step 2, 2026-06-24)

*Branch off `main` (with zero-capture closures #418). Resolves step 2 of the closure
triage: `let k = …; let f = |x| x + k; f(…)` now captures `k` correctly.*

## Approach — extra leading params + call-site injection (no heap closure object)

A capturing closure that is **directly bound and called within the same function** does not
need a first-class runtime closure object. The synthetic closure fn takes the captured
values as **leading params**, and the captured enclosing registers are **injected at the
direct call site**. The closure value therefore never enters the unified bare-fn-ref
representation, so **#418 zero-capture closures and fn-pointer interop are untouched**, and
no new aggregate-store sites are introduced (dodging the by-value-aggregate miscompile).

### Pieces (all in `self-hosted/ir/lower.sio`)
1. **Free-variable scan** (`scan_captures_opt`/`scan_captures_expr`, methods): recursively
   walk the closure body BEFORE wiping locals; an ident that resolves via `self.lookup_local`
   and is not a closure param is a capture. (Method form uses `self` for the enclosing
   scope — see the parser note below.)
2. **Leading capture params**: in `lower_closure_expr_ref`, bind each captured name to a
   leading synthetic-fn param (reusing the extract-Box-first param binding from #418), then
   the lambda's own params.
3. **Capture table** keyed by synthetic fn_id (`ClosureCapTable` + `record_closure_caps` /
   `find_closure_cap_idx` methods) storing the captured **enclosing** regs.
4. **Binding side-channel**: `pending_closure_fn_id` (set by the closure lowering, consumed
   by the let-binding) + a `closure_fn_id` slot on each local (`bind_local_closure_fn_id`).
5. **Call-site injection**: in `lower_call_expr_ref`, a call of a capturing-closure local
   emits a **direct** `ir_call(synthetic_fn, [captured_regs…, arg_regs…])` (IrCall relocs by
   fn_id; up to 6 reg args + stack spill).
6. **Escaping guard**: a capturing-closure local used as a **value** (passed to a fn,
   returned, aliased) — i.e. reaching the ident-as-value lowering rather than the callee
   path — is a **loud compile error**, not a silent wrong result.

## Verified (madaros built from this source, `ulimit -s unlimited`)
- Headline `let k=7; \|x\| x+k; f(5) → 12`; capture-only `let k=100; \|x\| k; f(99) → 100`.
- Multi-capture `let a=10;let b=2; \|x\| x+a+b; f(30) → 42`; multi-call `\|x\| x+k (k=5);
  f(1)+f(2) → 13`.
- **#418 no-regression**: `(\|x\| x)(42) → 42`, zero-capture `callit(f) → 42`, `let f=add1;
  f(41) → 42`; structs/enums/methods/fn-pointers compose.
- **Escaping guard**: `callit(\|x\| x+k)` now emits *"capturing closure cannot be used as a
  value (escaping closures are not yet supported)"* — previously a silent wrong `21`.
- No-regression sweep: identical **26/50** run-pass exit-0 vs prebuilt main (0 regressed);
  madaros self-builds.

## Honest scope
- **Captured `let` values, read at call time.** A captured `var` mutated between the
  closure's creation and its call would diverge (call-time read) — out of scope; use a
  `let` snapshot. Up to 8 captures; up to 6 total (caps+args) in registers, rest spill.
- **Direct call only.** Escaping capturing closures (passed/returned/stored) are a compile
  error by design — the next increment would give them a heap closure object (fn-ptr + env)
  using this same free-var analysis.
- The closure body scan handles `left/right/else/args` sub-expressions; block-bodied
  closures with captures inside `{ … }` statements are not yet scanned.

## Parser/checker notes (workarounds, not bugs introduced)
- The parser rejects `&(*field_box)` (a shared ref of a deref of a **field-accessed** box) —
  `&!(*local_box)` is fine. So the scan and table lookups are **methods** using `self`
  rather than free functions taking `&LowerLocalStack` / `&(*self.closure_caps)`.
- One **non-fatal** checker diagnostic remains (`expected &[struct;256], got &![struct;256]`)
  on the boxed `ClosureCapTable.entries` access — a `&`/`&!`-kind false positive. The
  emitted code is correct (every test passes, 0 regressions, self-build); extracting the
  table to a local would risk the by-value-aggregate copy miscompile, so the direct box
  access is intentional.

## AI disclosure
Implementation by AI agent (Claude) under human direction, on advisor guidance to use the
extra-params scheme over a heap closure object. Every claim backed by a re-runnable
`madaros compile/run` probe.
