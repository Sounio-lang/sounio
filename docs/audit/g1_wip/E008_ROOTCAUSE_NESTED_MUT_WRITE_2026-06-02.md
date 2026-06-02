# E008/E170 — PROVEN root cause: nested `*mut` field writes don't persist; the fix exposes a far less-functional checker (2026-06-02)

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

## Why the fix is net-negative — the real state of the checker
The crashers have only **2–4 functions** and pass `COL_END`/`CHK_START` (collect is fine), then
crash **inside the check pass**. So the 170 crashes are **NOT** copy-volume — they are **latent
`*mut` bugs in the deeper body/expr/stmt checking that were never reached before**, because the
broken `fn_sigs` made the checker bail to a shallow accept-everything path. **The baseline's 125
"passes" were largely FALSE PASSES** — the modular checker was not actually type-checking
function bodies. Making `fn_sigs` work turns shallow non-checking into real checking, which both
clears spurious E008 *and* exposes that the in-place check spine crashes on genuine checking.
The modular checker is substantially less functional than the corpus census implied.

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
   `(*c).a.entries[i] = x`). This is the single highest-leverage codegen fix — it lets EVERY
   in-place collector use cheap manual writes and is almost certainly miscompiling other in-place
   ports silently. (native/move-codegen lane.)
2. **Fix the latent `*mut` body-check crashers** the persistence fix exposes (closures,
   match-deref, large-aggregate copies in expr/stmt checking) — the same `*mut` codegen class.
3. Only after 1+2 does the E008 lever convert to a real corpus gain without crashes.

Owner: Claude (worktree /workspace/sounio-e008). Census harnesses + logs under `.build/census*/`.
Binaries: `/tmp/mc_e008_base.elf` (baseline), `/tmp/mc_e008_fix3.elf` (fix, net-negative).
