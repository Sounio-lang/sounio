# E008/E170 — PROVEN root cause: nested `*mut` field writes don't persist (2026-06-02)

**Status: ROOT CAUSE PROVEN by instrumented rebuilds + census. Earlier theories in this file's
history (by-value-spine drop / by-value-Checker truncation / lossy FnSigTable.find) were each
falsified by a rebuild and replaced.** Worktree `/workspace/sounio-e008`, branch
`g1/e008-bridge-fix`, base `g1/qualify-bare-patterns @ ed581987e`, `ulimit -s 1048576`.
Source reverted clean — this is a diagnosis + an exact patch the `*mut`/codegen lane should own.

## TL;DR
1. `--check` runs the **in-place `*mut` spine** (collect-then-check on one heap `*mut Checker`).
2. The in-place collect writes signatures with **two-level nested field writes through the
   pointer**: `(*c).fn_sigs.entries[i] = sig` and `(*c).fn_sigs.count = i+1`. **These do not
   persist** — the backend stores into a temporary copy of `fn_sigs` that is never written back.
   PROVEN: after two `DBG_COLLECT_INPLACE` adds, `COL_END count=0` (and `CHK_START count=0`).
3. So at check time `fn_sigs.find` returns −1 → `current_return_type` is never set → every
   explicit `return <expr>` mismatches `unit` → **spurious E008 "expected ()"** (122 progs),
   and `current_effects` is empty → **E170** (27). It also **silently skips the body-type check**
   (`fn f()->i64{"hello"}` type-checks OK) — a latent correctness hole.
4. The **single-level** write `(*c).fn_sigs = (*c).fn_sigs.add(sig).0` DOES persist (same idiom
   as the working `(*c).env = (*c).env.push_scope()`, and what the struct/enum collectors at
   check.sio:2343/2387 already use). Applying it (the patch below) makes `count=2`, `find`
   resolve, E008 clear where types match, and closes the body-type hole. **But it is
   net-negative on the corpus** and must NOT land on the gate (see §Census).
5. The proper fix is a **codegen fix for nested `*mut` field-write persistence** (so the
   cheap manual writes work) — this is the `*mut`/native lane's domain. Sounio has no
   field-pointer (`&!(*c).fn_sigs`) idiom to write a single struct-field level by hand.

## Proof chain (instrumented mc.elf builds, `fn f()->i64{return 5}` + `fn main()->i32{0}`)
- by-value `check_fn_item` DBG never fires; `DBG_CRE_INPLACE`/`DBG_EMIT_2623` fire → **in-place
  spine**, not `check_program_with_artifacts`'s by-value methods (which are dead here).
- `DBG_COLLECT_INPLACE added ret_kind=0` (f→i64) and `=16` (main→i32) → collect builds correct sigs.
- `COL_END count=0`, `CHK_START count=0`, `DBG_CFI_INPLACE sig_id=-1` → **the writes were lost**.
- After the §Patch: `COL_END count=2`, `CHK_START count=2`, `sig_id=0/1`; `return 5` in an i64 fn
  → clean; `let x:i64=5; return x` → `check: OK`. `fn f()->i64{"hello"}` → now correctly E008.

## Census (504 run-pass; baseline mc.elf `0889ac6d` vs fix `mc_e008_fix3`)
| | PASS | FAIL | CRASH | E008(first) |
|---|---:|---:|---:|---:|
| baseline | 125 | 376 | 3 | 122 |
| with fix | **112** | 222 | **170** | 28 |
- Real wins: **26** FAIL→PASS (E008/E170 where types genuinely match), e.g. `import_chain_c`,
  `issue11_ptr_*`, `knowledge_kas1_policy`, `knowledge_struct_field_ok`.
- Regressions: 16 PASS→FAIL + **23 PASS→CRASH**; total CRASH explodes 3→170.

## Why the fix is net-negative — what is proven vs hypothesised
**PROVEN:**
- The crashers have only **2–4 functions** and pass `COL_END`/`CHK_START` (collect is fine), then
  crash **inside the check pass** — so the 170 crashes are **NOT** collect copy-volume.
- The body-type check was being **skipped** while `fn_sigs` was empty (sig_id<0); the fix
  re-enables it. Demonstrated on `fn f()->i64{"hello"}` (one program) now correctly erroring —
  a real frontend body-type hole.

**HYPOTHESIS (NOT isolated — do not over-read the dramatic 170):** the crashes have ≥3 possible
causes I did not separate: (a) latent deep *checker-logic* bugs newly reached; (b) the fix's own
large-aggregate `*mut` copies / now-triggered recursion on newly-exercised paths; (c) **more
instances of the same nested-write / large-`*mut`-copy codegen class just root-caused**.

**Evidence points AWAY from (a), toward (c):** these are `tests/run-pass/*.sio` — the **canonical
`bin/souc` (the fixed-point compiler bundling the SAME `check.sio` logic) checks them fine**. So
the checker *logic* is sound; the 170 crashes are `mc.elf` **codegen** artifacts, not "the checker
can't check." Do **not** read this as "the modular frontend is fundamentally broken" or "the 125
baseline passes were mostly false" — that generalisation is unproven and would mis-route the war.

**Useful reframe:** the crashes are very plausibly the same `*mut` codegen disease as the E008
root cause, so **one codegen fix (nested-write persistence + large-aggregate `*mut` copy) likely
unblocks BOTH the E008 lever and most of the 170 crashes** — a single actionable redirect, not two
separate hopeless problems. (Discriminating test, not yet run: route the check-pass table reads
`fn_sigs.find/.get`/`structs.find`/… through direct `*mut` scans and see if the crash count drops.)

> NOTE (unresolved — flag for the next reader): preflight → `check_program_epistemic_into` →
> `check_program_with_artifacts` reads as the **by-value** spine in source, yet the **in-place**
> spine demonstrably executes (DBG). The patch targets the in-place collect, which runs, so it is
> correct — but **source read ≠ execution path here**; do not trust the by-value `check_*` source
> as "what runs" without instrumenting.

## §Patch (verified-correct persistence fix; do NOT land on the gate yet)
In `checker_collect_fn_def_inplace` (check.sio ~2278), replace the manual nested writes:
```
// BEFORE (lost):
let sig_id = (*c).fn_sigs.count
if sig_id < 64 { (*c).fn_sigs.entries[sig_id as usize] = sig; (*c).fn_sigs.count = sig_id + 1 }
... (*c).env.bindings[benv_idx] = TypeBinding{...}; (*c).env.count = benv_idx + 1 ...
// AFTER (persists):
let add_result = (*c).fn_sigs.add(sig); (*c).fn_sigs = add_result.0; let sig_id = add_result.1
(*c).env = (*c).env.bind(name, ty_fn(sig_id), false)
```
(Optionally read sigs in the check pass via a direct `*mut` scan to avoid the per-function 45KB
table copy that `.find`/`.get`/`.add` incur — a free helper scanning `(*c).fn_sigs.entries[i]`.)

## Remaining work (the actual unblock, in priority order)
1. **Codegen: make two-level nested `*mut` field writes persist** (`(*c).a.b = x`,
   `(*c).a.entries[i] = x`) and audit large-aggregate `*mut` copies. Single highest-leverage
   codegen fix — lets EVERY in-place collector use cheap manual writes, is almost certainly
   miscompiling other in-place ports silently, AND (leading hypothesis) likely accounts for most
   of the 170 crashes too, so it plausibly unblocks the E008 lever and the crashes together.
   (native/move-codegen lane.)
2. Re-run the corpus after (1); only the residual crashes need separate triage. Run the
   discriminating direct-`*mut`-scan test above first to confirm how much is codegen-class.
3. Only after the crashes are gone does the E008 lever convert to a real corpus gain.

Owner: Claude (worktree /workspace/sounio-e008). Census harnesses + logs under `.build/census*/`.
Binaries: `/tmp/mc_e008_base.elf` (baseline), `/tmp/mc_e008_fix3.elf` (fix, net-negative).
