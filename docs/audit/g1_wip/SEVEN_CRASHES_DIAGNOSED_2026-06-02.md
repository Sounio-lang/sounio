# The 7 residual mc_eff crashes — diagnosed (2026-06-02)

The 7 crashes left after the fall-through codegen fix (in the non-shipped effect-validation
config, mc_eff) are **3 distinct mechanisms**, pinned by a 4-strategy adversarial gdb workflow.
ALL 7 are TRIGGERED by the effect patch (the +7 baseline never reaches these paths), but NONE is
invented by it — they are pre-existing latent defects the patch newly exercises.

## Cluster A — 4 progs — CODEGEN (a SECOND match-lowering bug, distinct from the fall-through)
- Progs: epistemic_bmi, pbpk_simple, vancomycin_auc_epistemic, real_sounio_native_knowledge_demo
  (all Knowledge<T>-heavy). Identical fault: `mov (%rax),%rax`, rax=0, rip=0xcd8f93.
- Source: `types_equal` (`self-hosted/check/epistemic.sio:653-663`):
  `match (a.inner, b.inner) { (Some(inner_a),Some(inner_b)) => types_equal(*inner_a,*inner_b),
   (None,None) => true, _ => false }` — well-guarded + exhaustive in source.
- ROOT (codegen): bin/souc's match-arm dispatcher (`lean_single.sio ~20024-20100`) handles
  Some(57)/None(58)/Ok/Err/ident/literal/wildcard/or-pattern but has **NO tuple-pattern `(`
  handler** — the catch-all at ~20100 just `EP=EP+1` (skips one token). So a tuple-arm match
  emits a broken decision tree: it tests ONLY the first element's discriminant (slot -0x40 =
  &a.inner) — twice, redundantly — NEVER the second (b.inner is loaded to -0x28/-0x20 but never
  compared), drops the `(Some,Some)` recursive call entirely, and emits the `(Some,Some)`-body
  deref of `inner_a` (bound only on the Some path, slot -0x48) on the `(None,None)` fall-through
  path. DISPOSITIVE: types_equal is a LEAF in the binary (no `call` in 0xcd8e36..0xcd901b) yet
  the source `(Some,Some)` arm REQUIRES a recursive call — it's absent.
- IMPACT: **shipped bin/souc bug.** Miscompiles ANY real tuple-arm pattern, e.g. the second one
  already in shipped source: `check.sio:10208 (Some(arg_elem),Some(param_elem))`.

## Cluster C — 2 progs — CODEGEN (SRET large-aggregate-return overrun)
- Progs: fibonacci, darwin_atlas/lib. rip jumps into the 0x7fffb9.. heap region (zeros);
  `[rsp-8]==rip` ⇒ a `ret` to a poisoned slot.
- ROOT: a function returning a ~168KB aggregate by value. Epilogue `rep movsq` of 0x51ff qwords
  (167928 B) into the caller-allocated SRET buffer (rdi=r12) OVERRUNS the callee's own saved
  rbp/return-addr slot → the trailing `ret` jumps to a heap field of the copied aggregate.
- IMPACT: **shipped bin/souc bug** — large by-value aggregate (Checker-derived) return lowering
  places the SRET buffer where the constant-length copy clobbers control data. This is the
  large-by-value-Checker-return class the `*mut`/move-codegen arc was built to avoid.

## Cluster B — 1 prog — LOGIC (check.sio, latent baseline bug)
- Prog: ossm_multihead. `rep movsq` with rsi=0x40 (NOT null) at rip=0x4dc3e7e = OOB array read.
- ROOT: `FnSigTable { entries: [FnSig;64], count: i64 }`. `get(idx)` (`defs.sio:1242`) indexes
  `entries[idx]` with NO bounds check. The registration (`check.sio:2278-2284`) guards the table
  WRITE with `if sig_id < 64` but binds `ty: ty_fn(sig_id)` OUTSIDE the guard (line 2284), so the
  65th fn gets a dangling `fn_sig_id==64`; the call check (`check.sio:3691`) guards only `>=0`
  (no `< count`) → `get(64)` reads `entries[64]` = the inline `count` word (0x40) and copies it by
  value (deref 0x40) → SIGSEGV. Same latent twin at `check.sio:15438`.
- IMPACT: **latent committed-source bug** (>64-fn programs). Dormant in the baseline (it never
  reaches get with the dangling id); the patch triggers it. Capacity-raise only defers the cliff.

## Fixes (all distinct from the just-landed fall-through fix)
- **A (bin/souc codegen):** add a tuple-pattern arm handler in `lean_single.sio` match-lowering —
  destructure the tuple, AND-combine a per-element discriminant test, place each arm body only
  under its own combined guard. SOURCE-SIDE UNBLOCK (no re-bootstrap): rewrite the 2 tuple matches
  (`epistemic.sio:657`, `check.sio:10208`) as nested single-scrutinee matches (the existing
  Some/None dispatcher lowers those correctly).
- **C (bin/souc codegen):** fix SRET placement so a large-aggregate return buffer can't overlap
  the callee's saved rbp/ret slot, OR eliminate the 168KB by-value Checker return (the `*mut`
  refactor).
- **B (check.sio logic):** bounds-check `defs.sio:1242 get` + move the `ty_fn(sig_id)` env bind
  INSIDE the `if sig_id < 64` guard (`check.sio:2279`); same at `:15438`.

Cross-cut: do NOT ship the effect-validation patch until A/C codegen + B logic are fixed
(it's already marked NET-NEGATIVE / not shippable, `49f035fd9`). A & C are independent shipped
bin/souc codegen bugs worth fixing regardless of the effect work.
