# Lane charter: by-value struct-argument-from-(−1) checker crash — 2026-06-03

Shared lane between the native/codegen and G1 frontend efforts. This is **bug #2**
of the front-half pair: distinct from the nested-`*mut`-store bug (#1), which is
already fixed on the base branch. Opening it per `.claude/PARALLEL_BLOCKER_CONTRACT.md`.

## Parallel Lane Contract

```text
Lane:          byval-arg-crasher (dominant body-check SIGSEGV)
Owner:         claude (native/codegen lane) — CLAIMED 2026-06-03
Base:          codegen/deref-nested-store (has the nested-store fix #1, so the
               crasher is REACHABLE; branching off g1/e008-bridge-fix would NOT
               reach it because there fn_sigs stays empty and the checker bails)
Worktree:      /workspace/sounio-byval-crasher
Branch:        codegen/byval-arg-crasher
Write-Set:     self-hosted/compiler/lean_single.sio  (x86 codegen — the fix site)
Read-Set:      self-hosted/check/check.sio, self-hosted/check/defs.sio
               (the by-value call-arg path — inspect only)
Required-Gates: bash scripts/ci/native_v2_e2e_exit_code_gate.sh ;
               504-corpus census delta (crash count must fall, see Acceptance)
Merge-Target:  g1/e008-bridge-fix (front-half integration), after #1
Known-Blockers: BLK-20260603-byval-arg-crasher-deref (below)
```

## Blocker record

```text
Blocker-ID:    BLK-20260603-byval-arg-crasher-deref
Status:        review-ready          (diagnosis complete + pinned; fix HANDED to G1 frontend lane 2026-06-03)
Severity:      B1                    (dominant front-half crash; 131/170 genuine SIGSEGVs)
Class:         compiler-semantics    (codegen: by-value method-call struct-arg passing)
Owner:         G1 frontend lane (g1/e008-bridge-fix) — fix (A) is in check.sio (their domain).
               Diagnosed/pinned by the native/codegen lane (claude); handed off here.
Lane:          byval-arg-crasher
Worktree:      /workspace/sounio-byval-crasher
Branch:        codegen/byval-arg-crasher
Files-Owned:   self-hosted/compiler/lean_single.sio  (this lane — fix (B) only, not pursued)
Files-Read-Only: self-hosted/check/check.sio, self-hosted/check/defs.sio
Do-Not-Touch:  bin/souc, bin/souc-linux-x86_64 (serialize re-bootstrap via souc-build-lock.sh)
Repro:         (needs a compiler built WITH the #1 nested-store fix, e.g. /tmp/mc_fixed.elf)
               ulimit -s 1048576; <mc_with_fix> --check repro/bodycheck_crash_paramd_call_min.sio
Observed:      SIGSEGV (rc=139) in a by-value boundary method called as (*c).method(...)
               from the knowledge/unit/ontology call-arg `_inplace` wrappers (layout-
               sensitive exact site; see Instrumentation result)
Expected:      check completes (rc 0 or a clean type error), no SIGSEGV
Acceptance-Gate: rebuild modular mc with the patched check.sio, re-run the 504
               run-pass census (1 GB stack); the ~131-crash cluster must drop to ~0
               with no new crashers; native_v2_e2e gate stays green
Evidence-Level: E3 (gdb + marker-instrumented rebuilds on the live binary)
Evidence:      this doc (Instrumentation result) ; STRUCT_RETURN_FIX_ATTEMPT_2026-06-02.md ;
               census TSVs docs/audit/g1_wip/census_mc_{baseline,fixed}_2026-06-02.tsv
Fallback-Path: none
Legacy-Kept:   n/a
LLM-Offload:   not-required
Next-Action:   G1: *mut-convert check_call_arg_{knowledge,unit,ontology}_boundary
               (+ report_unit_call_mismatch as a latent sibling) so the boundary
               `_inplace` wrappers stop doing `(*c).<by-value method>(...)`; verify
               via the Acceptance-Gate census. (Fix (B), the lean_single codegen of
               4th-struct-arg by-value-method passing, stays with this lane as a
               follow-up — layout-sensitive, not repro-isolable.)
```

## Root cause (machine-level, PROVEN; source-level OPEN)

PROVEN (gdb on live mc_fixed, no rebuilds — see STRUCT_RETURN_FIX_ATTEMPT doc):
- Fault at `0x4c2805b` for **131 of 170** genuine crashers (deterministic, one site).
- The faulting function is a **by-value `Checker` method**: prologue `sub $0xa4250,%rsp`
  (672 KB frame); copies its args by `rep movsq` — arg0 ≈ 164 KB (a `Checker` by
  value), arg1/arg2 = 272 B (`TypeEntry`), **arg3 = 16 B**.
- It crashes doing `mov 0x0(%rdx),%rax` to copy arg3, with **rdx (arg3 ptr) = −1** —
  a `find()`-miss sentinel used, unguarded, as the source ADDRESS of a by-value
  struct argument.
- It is a struct-**ARGUMENT** bug, NOT struct-return (label corrected).

OPEN (could not settle by reading source):
- Exact faulting function. Fingerprint: by-value, returns `Checker`, args =
  `Checker + 2×TypeEntry + 1×16-byte`. Candidate family: by-value `report_*_mismatch`
  / call-arg-boundary checkers.
- Why a plain `f(5)` reaches a by-value path at all. The bridge
  `call_expr_should_bridge_by_value` reads as returning false for `ExprCall`
  (check.sio:15304: "ExprCall never sets e.right"), contradicting the observed
  by-value crash. **G1's own NOTE resolves the meta-point:** *"source read ≠ execution
  path here … do not trust the by-value `check_*` source as 'what runs' without
  instrumenting."* So do NOT trust the dispatch source — instrument the binary.
- The bare-enum-pattern miscompile family is RULED OUT (dispatch already uses
  qualified if-equality; 0 bare ExprCall/ExprIndex arms).

## Why isolated bootstrap repros fail to reproduce

7 faithful bootstrap models (repro/sret_norepro_attempts/) all compile correctly under
the fixed bootstrap. The bug only manifests in the full check.sio codegen — so the
fix path is **instrument-the-real-binary**, not model-and-fix.

## Convergence with the G1 lane

G1 independently hit the same crash: commits `9e72844b1` ("gdb the arg-checker crash
— large-by-value-TypeEntry-arg codegen miscompile") and `7f8c4dac8` ("sig.params lead
REFUTED"). Both lanes agree: large by-value `TypeEntry`/aggregate argument codegen,
sentinel −1 deref, sig.params is NOT the culprit. See ALIGNMENT_WITH_G1_LANE_2026-06-03.md.

## Concrete first moves for the owner

1. Build a marker-instrumented mc: add a unique `print`/syscall at the top of each
   candidate by-value call-arg function in check.sio, rebuild via
   `scripts/dev/souc-build-lock.sh`, run the repro under gdb — the last marker before
   the fault names the function. (Cheaper than a full symbol build.)
2. Once named: trace where its 16-byte struct arg (a `Span`? a `TypeEntry`?) is
   sourced; find the `find()`/lookup that returns −1 and is passed unguarded.
3. Fix = guard the −1 at the call site, OR (if the dispatch itself is wrong) route to
   the working `*mut` in-place path. Verify via the Acceptance-Gate census.
4. G1's not-yet-run discriminating test: route the check-pass table reads
   (`fn_sigs.find/.get`, `structs.find`, …) through direct `*mut` scans and see if the
   crash count drops — may localise the class cheaply.

## Instrumentation result (2026-06-03) — PINNED to the residual by-value boundary methods

Marker-instrumented mc (3 rebuilds via souc-build-lock) on the repro, gdb on the live
binary. Findings:

1. **Path = `*mut` in-place spine** (not the by-value bridge): only `@@DBG_INNER_INPLACE@@`
   fired; the by-value `check_call_args_inner` markers never did. (`call_expr_should_bridge_by_value`
   was a red herring; the in-place path runs, as G1's "source≠execution" NOTE warned.)
2. **Crash is in the call-arg boundary checks**, specifically the residual boundary
   `_inplace` wrappers that STILL pass `(*c)` by value. The OBSERVED crash for `f(5)`
   is the **knowledge/ontology** boundary call (the wobble below) —
   `(*c).check_call_arg_{knowledge,ontology}_boundary(arg_ty, param_ty, call_span)`
   (e.g. check.sio:4016 for ontology). **`report_unit_call_mismatch` is EXONERATED for
   `f(5)`** — its guard is false (instrumented unit_id = −1 in/out), so it is NOT the
   observed crash; it is only a **LATENT** same-pattern site (would crash if the
   unit-mismatch guard ever fires). G1: do NOT chase the guarded unit-mismatch path first.
   All these `(*c).method(...)` calls **deref the *mut and pass the 164 KB `Checker` BY VALUE**.
   `check_call_arg_{knowledge,unit,ontology}_boundary(self: Checker, arg_ty: TypeEntry,
   param_ty: TypeEntry, call_span: Span)` matches the crash fingerprint EXACTLY
   (164 KB + 272 B + 272 B + Span; the 4th struct arg `call_span` passed from −1).
3. **Layout-sensitive (confirms prior verdict):** adding `print` markers MOVED the crash
   (0x4c2805b → 0x4c28613; instr2 faulted at unit_boundary, instr3 at the ontology call).
   The *pattern* is stable (`(*c).<by-value boundary method>(…, call_span)` with the 4th
   struct arg from −1); the exact instance wobbles with instrumentation. So marker
   bisection localizes the CLASS, not a single fixed line.
4. **`call_ref_inner_or_self` is NOT the culprit** (its TypeEntry return is clean:
   instrumented unit_id = −1 in and out; the unit-mismatch guard does NOT fire for `f(5)`).
   The crash is the by-value *method-call arg passing itself, not a corrupted TypeEntry.

### Root cause (refined)

The dominant crasher is the **codegen for a by-value method call `(*c).m(self_by_value,
struct, struct, struct)`** where `self` is the 164 KB `Checker` and there are ≥3
additional struct args — the **4th struct argument's source address is miscompiled to
−1**. The lane's *mut migration converted the borrow/refinement boundary checkers to
direct `*mut` (no by-value self) but left **knowledge / unit / ontology** still calling
`(*c).<by-value method>(...)` — those are the live crash sites.

### Two fixes

- **(A) Source, in check.sio (G1 domain), lane's PROVEN strategy:** finish the `*mut`
  migration — give `check_call_arg_{knowledge,unit,ontology}_boundary` (and
  `report_unit_call_mismatch`) `*mut Checker` transcriptions so the boundary `_inplace`
  wrappers stop doing `(*c).method(...)` (no 164 KB by-value self, no 4-struct-arg call).
  Removes the trigger exactly as it did for borrow/refinement. Verify via census
  (the 131-cluster should clear). **Cross-lane: needs check.sio ownership / G1 coordination.**
- **(B) Codegen, in lean_single.sio (this lane):** fix the 4th-struct-argument source
  address in by-value method-call lowering. The REAL bug, but layout-sensitive
  (Heisenbug) — not isolable by a small repro (faithful models compile correctly), so it
  needs in-binary work, and instrumentation perturbs it. Higher-risk, higher-value.

**Recommendation:** (A) is the tractable unblock and matches the lane's established
pattern; (B) is the durable codegen fix but is the hard Heisenbug. Pursue (A) with G1;
keep (B) as the codegen-hardening follow-up.

## Fix (B) investigation — started 2026-06-03 (lean_single codegen)

Per directive, started the lean_single codegen fix. Did NOT make blind edits (no fast
verify loop; layout-sensitive). Findings from reading the arg-passing codegen + more
in-binary gdb:

### Arg-passing model (lean_single 1677–1796)
- Args are evaluated and pushed to the stack, then loaded into arg registers by
  `emit_setup_call_args_shift_x86(argc, shift)` → `emit_load_rsp_arg_x86`.
- **Struct-by-value args are passed BY POINTER** (an 8-byte address per arg in a reg;
  the callee `rep movsq`-copies the struct from it — matches the crash prologue).
- A function returning a large struct uses **SRET with shift=1**: rdi = hidden return
  buffer, real args shift to rsi/rdx/rcx/r8/r9 (`emit_direct_fn_call_x86`, ret_slots>0).

### Hypothesis (concrete, testable — but verify is 2:36/rebuild + layout-sensitive)
The crash callee prologue copies **rdi as the 164 KB self** (a shift=0 / self-in-rdi
layout), yet `check_call_arg_{knowledge,unit,ontology}_boundary(self: Checker, …) ->
Checker` RETURNS a large struct → it *should* be SRET (shift=1, rdi=buffer). A
**caller/callee SRET-shift mismatch** (or a `ret_agg_nslots` miscount for the Checker
return at this call site) would misplace the arguments by one slot, so the 4th struct
arg (`call_span`) is loaded from the wrong stack slot → −1.

### Counter-evidence / why it's still hard
- The faulting frame has BOTH `rcx = −1` (bad 4th arg) AND `[rbp+8] = 0` (null return
  address) → caller-side frame/convention corruption, not just one mis-loaded arg.
- Layout-sensitive: instrumentation moves the crash, so a "fix" could pass by shifting
  layout rather than correcting logic — verification needs the full census, not a unit
  check, and a green census wouldn't prove the logic is right.
- NOT repro-isolable: 6 faithful bootstrap models (incl. `(*ptr).method(re-passed
  struct params)` with a 164 KB self) all compile correctly under ds_fixed2.

### Status of fix (B)
Concrete hypothesis in hand (SRET-shift / ret_slots-at-call-site for large-struct-
returning by-value methods), but no fast verify loop and a layout-sensitive target.
Proceeding means hypothesis-driven edits to `emit_direct_fn_call_x86` /
`ret_agg_nslots` + full mc rebuild + 504-census per iteration. Higher-risk than fix
(A). Recommended order remains: land fix (A) (G1, removes the trigger) first; pursue
(B) as codegen hardening with the SRET-shift hypothesis as the entry point.
