# E008/E170 root cause — by-value `Checker` copy TRUNCATION, not bridge-state logic (2026-06-02)

**Status: PROVEN by rebuild. Redirects the #1 front-half lever away from the documented
hypothesis.** Worktree `/workspace/sounio-e008`, branch `g1/e008-bridge-fix`, base
`g1/qualify-bare-patterns @ ed581987e`. `ulimit -s 1048576`. Read-only conclusion (the one
source edit attempted was reverted after it was empirically falsified).

## TL;DR for whoever owns check.sio / the *mut migration
The prior docs (FRONT_HALF_LEVERAGE_HANDOFF, MODULAR_CORPUS_FAILURE_BACKLOG) say E008 is a
**bridge-state-loss logic bug** — "carry `current_return_type` across the by-value bridge at
check.sio:1146/~2489." **That class of source fix cannot work in the by-value spine, and I
proved it by rebuild.** Setting `current_return_type` + `current_effects` in `check_fn_item`
directly from the AST changed nothing: E008 and E170 were byte-identical after the rebuild.
The cause is a **codegen-level miscompilation of by-value `Checker` copies in `mc.elf`**:
specific fields — `fn_sigs`, `current_return_type`, `current_effects` — do not survive the
by-value `self` threading through the check spine, so anything written to them in that spine is
discarded before the read site. The only real fixes are (A) route the whole `--check` through
the in-place `*mut` spine (one heap Checker, no copy), or (B) fix the codegen copy bug — both
are the `*mut` migration's domain.

## Baseline census (mc.elf md5 `0889ac6d`, 504 run-pass programs)
PASS=**125**, FAIL=376, CRASH=3. First-error buckets: parse 123, **E008 122**, parse_failed 49,
**E170 27**, E004 18, E011 8, E001 8, E016 5, E015 3. → E008+E170 ≈ 149 = the #1 lever.

## The proof chain (all on `mc.elf`, cheap `--check` probes)
1. Bug is **explicit `return <expr>` only**: `fn f()->i64{return 5}` → E008 "expected ()";
   `fn f()->i64{5}` (implicit) → OK.
2. The implicit body-type check is **silently skipped** too: `fn f()->i64{"hello"}`,
   `fn f()->i64{Foo{a:1}}` → rc=0. Unambiguous mismatches uncaught ⇒ at check time
   `c.fn_sigs.find((*fd).name)` returns **−1** (`sig_id<0`), so both line 13908 (set
   `current_return_type`) and 13922 (body check) are skipped.
3. `--check` path = `check_program_with_artifacts` (mod.sio:1515) → **by-value** spine
   `checker.collect_items` then `checker.check_items`. `collect_fn_def` (12851) adds the sig
   **unconditionally**; `FnSigTable.add`/`.find` (defs.sio:1191) share identical
   `entries[]`+`count` storage. So the table logic is correct — the sig is **lost between the
   two passes**.
4. **Direct fix attempt (reverted):** set `current_return_type` + `current_effects` in
   `check_fn_item` straight from the AST `FnDef` (Box pointer, not subject to the copy bug),
   exactly as `collect_fn_def` lowers them. Rebuilt `mc.elf` (md5 `d73d0818`). Result:
   **E008 unchanged AND E170 unchanged.** Both my field-sets were discarded.
5. **Which fields survive the by-value spine (PROVEN) — mechanism (NOT fully pinned):**
   - SURVIVE: `env` (calls resolve), `structs`/`enums` (literals accepted; E015 only 3),
     `had_error`/`error_count` (failures ARE reported, rc=1 — so the verdict propagates).
   - DROPPED: `fn_sigs` (`find` = −1, proven by the silent body-type hole),
     `current_return_type` + `current_effects` (my AST-sets discarded, proven by rebuild;
     `current_return_type` reads as its init `ty_unit()` ⇒ "expected ()").
   Struct field order (check.sio:105) is `env`(1) … `structs`(5) `enums`(6) **`fn_sigs`(7)** …
   `error_count`(53) `had_error`(55) **`current_return_type`(56)** **`current_effects`(58)**.
   A naive "copy keeps a prefix, truncates after field 7" story is **contradicted**:
   `had_error`(55)/`error_count`(53) survive while `current_return_type`(56) does not. So it is
   **not a clean positional cut** — it is a field/size-specific miscompilation of the
   large-aggregate by-value copy (note the dropped fields include the two largest tables
   `fn_sigs` plus the deep `current_*` block; the surviving scalars `had_error`/`error_count`
   are also set DEEP and flow *outward* via returns, whereas `current_return_type` must flow
   *inward* via `self` param-passing — direction may matter). Pinning the exact codegen
   mechanism needs gdb on the move-codegen probe; it is **not required** to act — (3)+(4) already
   prove the by-value source fix is dead.

## Consequences
- **122 spurious E008 + 27 E170** (the #1 lever) are symptoms of this single codegen
  truncation, *not* a logic bug.
- **Silent correctness hole:** with `fn_sigs` dropped, the body-type check is skipped, so
  wrong return/body types pass clean (`fn f()->i64{"hello"}` → OK). This is latent and
  dangerous independent of the spurious errors.

## Fix options (all in the *mut migration's lane — DO NOT attempt a by-value source fix)
- **(A) Route `--check` through the in-place `*mut` spine.** `checker_collect_*_inplace` +
  `checker_check_items_inplace` keep one heap `*mut Checker` (no copy), so `fn_sigs` and
  `current_return_type` (set at 2516, read at 2622) work. **Blocker:** the in-place COLLECT is
  incomplete — `checker_collect_item_inplace` (2391) handles only `ItemFn`/`ItemStruct`/
  `ItemEnum`; **impl, effects, units, typealias, all policies, trait, algebra, study, ontology,
  models are no-ops.** Routing now would regress every program using those. **Remaining work =
  port those ~12 collectors to `*mut`** (mirror `checker_collect_fn_def_inplace`'s in-place
  insert idiom). Then materialize `(*cptr)` back into `CheckArtifactsResult` without a lossy
  by-value read.
- **(B) Fix the codegen truncation** of large by-value struct copies in `bin/souc`
  (lean_single.sio lowering) — the move-codegen lane's domain. Higher blast radius; would fix
  this class wholesale.

## What this redirects
Stop pursuing "carry `current_return_type` across the bridge" — it is a no-op against the
truncation (proven). The front-half E008 lever is **coordination-blocked behind the `*mut`
migration**, specifically behind **completing the in-place collect spine**. The exact missing
collectors are enumerated above.

Owner: Claude (worktree /workspace/sounio-e008, branch g1/e008-bridge-fix). Census harness +
logs under `.build/census/`. Binaries `/tmp/mc_e008_base.elf` (baseline), `/tmp/mc_e008_fix.elf`
(reverted-fix proof).
