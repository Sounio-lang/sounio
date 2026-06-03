# The 7 residual mc_eff crashes — diagnosed (2026-06-02)

The 7 crashes left after the fall-through codegen fix (in the non-shipped effect-validation
config, mc_eff) are **3 distinct mechanisms**, pinned by a 4-strategy adversarial gdb workflow.
ALL 7 are TRIGGERED by the effect patch (the +7 baseline never reaches these paths), but NONE is
invented by it — they are pre-existing latent defects the patch newly exercises.

## RESOLUTION (final, 2026-06-02)
Re-tested against the **shipped +7 baseline** (mc built from committed check.sio, NO effect patch):
fibonacci / darwin_atlas-lib / ossm_multihead = `rc=1`, epistemic_bmi = `rc=0` — **zero crashes**
(consistent with the census 481→0 and the 8000-slot tuple reproducer NOT firing). So the "7
residual crashers" framing was premise-wrong: shipped bin/souc is already crash-clean; all 7
manifest ONLY under the unshipped, NET-NEGATIVE (−95) effect patch.
- **Cluster A — FIXED + LIVE** (`1ff453590` source, `b765525fb` swap). The ONLY genuinely
  shipped bin/souc bug — reproduced standalone (no patch), fixed in lean_single, re-bootstrapped.
- **Cluster B — FIXED + COMMITTED** (`d7d580797`). Bounds-checked `FnSigTable.get` (defs.sio);
  verified a pure safe hardening — **0 verdict-flips across all 847 examples** (pre-fix vs post-fix
  mc, same bin/souc). The check.sio registration guard was dropped (redundant with the get
  bounds-check + conflicts with the effect patch).
- **Cluster C — LEFT DOCUMENTED (not pursued).** It is the known large-by-value-Checker(8MB)-SRET
  miscompile (see below). NOT pursued because: (1) the direct bin/souc large-SRET codegen fix is
  intractable-without-gdb (B-repro verdict); (2) the only tractable fix — route the remaining
  by-value Checker return via `*mut` — lives in the effect patch, which is net-negative and will
  not ship; (3) fixing the crashes does not make that patch shippable anyway (its blocker is the
  ~95 false-passes, not the 7 crashes). User decision: leave documented as the known miscompile.

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
- IMPACT: **latent codegen weakness in shipped bin/souc, but NOT triggered by any shipped-corpus
  program** (the +7 baseline is crash-clean — confirmed). Large by-value aggregate (Checker-derived)
  return lowering places the SRET buffer where the constant-length copy clobbers control data; only
  the effect patch introduces a by-value Checker return big enough to reach the cliff. This is the
  large-by-value-Checker-return class the `*mut`/move-codegen arc was built to avoid. The patch's
  OWN comment (`fn_sigs_e008_env_ontology_reports.patch` line 103) already names it — it routed 3
  report helpers via `*mut` to dodge it; fibonacci/darwin hit a DIFFERENT remaining by-value return.
  Direct fix is B-repro-verdict-intractable; the reliable fix is `*mut`-routing the remaining return
  (lives in the non-shipping effect patch). LEFT DOCUMENTED — see RESOLUTION above.

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

## Fixes — final status
- **A (bin/souc codegen) — DONE + LIVE** (`1ff453590` source, `b765525fb` swap). Added a
  tuple-pattern arm handler in `lean_single.sio` match-lowering (destructure the tuple, AND-combine
  a per-element discriminant test, each arm body only under its own combined guard). Validated:
  reproducer 139→111, 4-case correctness, gen2t==gen3t fixed point, 507/507 run-pass + 847/847
  examples 0 divergences.
- **B (check.sio logic) — DONE + COMMITTED** (`d7d580797`). Bounds-checked `defs.sio:1242 get`
  (`idx<0 || idx>=64 || idx>=count → empty_fn_sig()`) — universal (covers the by-value twin at
  `:15438`); verified 0 verdict-flips / 847 examples. (The registration-guard half — moving the
  `ty_fn(sig_id)` env bind inside the `if sig_id<64` guard — was dropped: redundant with the get
  bounds-check and conflicts with the effect patch.)
- **C (bin/souc codegen) — LEFT DOCUMENTED, NOT FIXED.** Candidate fixes remain: fix SRET placement
  so a large-aggregate return buffer can't overlap the callee's saved rbp/ret slot (intractable per
  B-repro verdict), OR eliminate the 168KB by-value Checker return (the `*mut` refactor — lives in
  the non-shipping effect patch). See RESOLUTION for why not pursued.

Cross-cut: A & B are landed. C is NOT a shipped-corpus crash (the +7 baseline is crash-clean); it
is a latent large-SRET codegen weakness reachable only via the effect patch, which is independently
NET-NEGATIVE / not shippable (`49f035fd9`). Do not ship the effect-validation patch (false-passes,
not crashes, are its blocker).
