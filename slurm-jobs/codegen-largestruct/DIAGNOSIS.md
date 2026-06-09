# c634b38f large-struct codegen miscompile — diagnosis

Branch `claude/codegen-largestruct-fix` off `claude/kw-demote-module` @ `d731cc3ce`.
Worktree `/workspace/sounio-codegen`. Compiler under test: `bin/souc` (built from
`self-hosted/compiler/lean_single.sio`).

## Reproducers (deterministic, local, ~seconds — NO full build / SLURM)

- `repro_boxstore.sio` — the three documented miscompiles (lower.sio:3324-3350).
- `repro_huge.sio`, `repro_largestruct.sio` — by-value param + SRET return; **PASS**
  (so plain by-value large-struct, even 210KB, is already handled correctly).
- `/tmp/min.sio` — the minimization (below).

## Behavioral fingerprint (PINNED)

Two store shapes miscompile under `bin/souc`:

1. `(*lo.inner).counter = 42` where `lo.inner` is a `Box<Inner>` **field of a local
   struct** `lo` → store writes a **discarded copy**; read-back = **0** (expected 42).
2. `(*m).items[3].field = v` (nested array-element struct-field store through a box) →
   **no-op**; read-back = 0.

Controls that WORK:
- `(*m).counter = 42` where `m` is a **direct box local** (not a field) → **42** ✓
- bound-local rewrite `var m = lo.inner; (*m).counter = 42; lo.inner = m` → **42** ✓
- **Reading** `(*lo.inner).counter` (rvalue) → correct ✓ (proven by probe1_fixed)

### Conclusions
- **NOT size-dependent**: a 16-byte struct fails; a 210KB by-value struct passes.
- The **read/rvalue path is correct**; only the **store-target lvalue** is wrong.
- Trigger = the deref base of the assignment lvalue is a **field-access** (`lo.inner`)
  rather than a simple local (`m`). i.e. lowering of `(*<field-access>).field = v`
  materializes a by-value copy of the pointed-to struct, stores into the copy, discards it.

⇒ The defect lives in the **assignment-lvalue / store address-of lowering** for a
`Deref(FieldAccess(...))` (and `Deref(...).index[].field`) base — in `lean_single.sio`
(bootstrap) and, very likely mirrored, in the modular `self-hosted/ir/lower.sio` +
`self-hosted/native/codegen_x86_linux.sio`. The existing in-tree workaround (bind box to a
local first, lower.sio:3334-3351) confirms this is the live class; the goal here is the
**root codegen fix**, not another bind-to-local workaround.

## Phase 1 — RESULT: root cause pinned (in `lean_single.sio` = `bin/souc`)

`bin/souc`'s codegen handles deref-stores with **token-pattern detectors** (lean_single.sio
4279-4366) that each require a **bare identifier** as the deref base (`( * ID )` =
tokens `6 19 3 7`). The handlers (e.g. `compile_deref_field_array_store_x86` 17473;
`stmt_is_deref_field_store` dispatch 19308) get the base pointer via
`var_find(ns,ne)` → `emit_load_var(slot)` (17526-27) — i.e. they LOAD THE POINTER FROM A
NAMED VARIABLE SLOT.

`(*lo.inner).counter = v` tokenizes as `( * ID . ID )` = `6 19 3 50 3 7` → the base is a
**field-access, not a named var** → matches NO detector → falls through to the generic
expression-statement path (20793-98) → the assignment is compiled as a discarded
expression → **store lost** (read-back 0).

### Why reads work but writes don't
The rvalue read path computes the base pointer correctly via the general expression
compiler (`compile_or`, which yields the pointer in RAX + `EXPR_TY`). The write handlers
DON'T use it — they require `var_find`. So the address-computation capability already
exists; the store handlers just don't reuse it.

### The fix (advisor option b — general, not whack-a-mole)
In the deref-store handlers, replace the `var_find(ns,ne)+emit_load_var(slot)` bare-variable
base load with **`compile_or()` on the base pointer expression** (the tokens between `*` and
`)`), then store at `[RAX + foff]`. Resolve the field via `EXPR_TY`/`ptr_hash_inner_ty` (as
the read path does). This subsumes every special-case detector (`(*a.b).c`, `(*a.b).c[i]`,
`(*a[i]).b`, …) instead of adding another. Must be mirrored in the a64 path (29954+) and in
the modular `lower.sio` place-lowering (5101-5172) if that path proves buggy too.

## OPEN CAVEATS (advisor, must respect before the expensive rebootstrap)
1. **This may not be THE wall.** The reproduced bug is a STORE producing a wrong VALUE.
   The last session's wall was a CRASH: SIGBUS + unpatched `e8 00000000` call relocations +
   a dropped READ in a print. Those are plausibly *distinct* members of the c634b38f class
   (the 0deb43bcb writer bug is explicitly "branch-target, NOT a data bug"; the
   `[IrFunction;1400]` materialization fault is far larger than this 210KB repro reached).
   ⇒ Fixing this lvalue store is genuinely in-class and worth doing, but do NOT assume it
   clears the SIGBUS. Test the actual wall after the fix.
2. **Verifying a `lean_single` fix requires the heavy rebootstrap** (`bin/souc` IS lean_single).
   Per the SUPREME-DIRECTIVE memory, self-host builds must route via SLURM. There is no fast
   local verify loop for a bootstrap-codegen change.
3. **Modular (`lower.sio`) bug status UNCONFIRMED:** modular emit paths are themselves broken
   here (`--native-compile` writes no file even for a trivial program; `--native-v2-compile`
   hits a front-half parse issue on `Box`), so the ~2s behavioral isolation isn't available.
4. **Cheap independent check available:** the `flat_reloc` 4096 cap (real latent overflow) can
   be exercised directly by a program with >4096 call sites — no 6642-fn build needed.

## Phase 2 — FIX IMPLEMENTED (lean_single x86, source-only, UNVERIFIED pending rebootstrap)

Purely **additive** change to `lean_single.sio` (zero edits to existing handlers ⇒ zero
regression risk to working paths; only previously-broken complex-base stores get newly handled):

- `stmt_is_deref_complex_field_store(p0)` — detector: `( * <complex> ) . field [= | [` where the
  base between `(*` and `)` is NOT a bare identifier (paren-depth scan to the matching `)`).
  Placed AFTER the bare-ident deref fast paths in dispatch, so they claim their forms first.
- `compile_deref_complex_field_store_x86()` — handler: skips `( *`, computes the base pointer via
  the general expression compiler **`compile_or()`** (the same path the rvalue READ uses, proven
  to yield the pointer — cf. working `var m = lo.inner`), spills it to a `NEXT_SLOT` var, resolves
  the field from the pointer's inner struct type (`type_is_pointer_like`/`ptr_hash_inner_ty`/
  `st_find`/`st_field_offset` — identical to the simple handler at 19325-31), then stores via the
  existing `emit_store_to_pointer_offset_x86` (`.field =`) or a mirror of
  `compile_deref_field_array_store_x86` (`.field[idx] =`).

This is advisor option (b): generalize the base-pointer acquisition (compile_or) instead of
`var_find`; subsumes the special-case detectors rather than adding more.

### Verification status
- NO local verify available: `bin/souc` (mini_native) only full-compiles (= the heavy rebootstrap,
  deferred per operator); `souc.elf --check` rejects the *pristine* lean_single at line 76 (modular
  checker vs mini_native dialect) so it can't gate the edit either.
- Type-resolution + emit idioms hand-verified against the existing working handlers
  (`emit_store_to_pointer_offset_x86` 7163, `compile_deref_field_array_store_x86` 17473, simple
  `(*PTR).field=` handler 19308, `type_is_pointer_like` 2804). Behavioral correctness awaits the
  bundled rebootstrap (then re-run `repro_boxstore.sio` / `/tmp/min.sio` → expect 42/42/55/55).

### Remaining (bundle before the single rebootstrap)
- **a64 mirror** (`compile_*_a64`, dispatch ~29954+) — not needed for the x86_64 self-host
  rebootstrap; latent for ARM targets. Add after x86 validates.
- **Modular `lower.sio` place-lowering** (5101-5172, base lowered by-value via `lower_expr_ref`):
  bug status UNCONFIRMED (modular emit broken locally). Confirm/patch when emit path is usable.
- **The SIGBUS/reloc wall** — keep investigating in parallel; it is plausibly a *different*
  class member (see caveat 1). Bundle any further lean_single codegen fixes into the one
  rebootstrap.

## Phase 2b — FIX CONFIRMED ON THE WALL PATH (not merely latent)

`grep` found **23 live complex-base deref-store sites in `self-hosted/ir/lower.sio`** with NO
bind-to-local workaround — the exact pattern my fix newly handles. Critically these include the
hottest IR-building paths:
- `intern_string` (3311-3315): `var lo = self; (*lo.module).string_table[idx]=n; (*lo.module).string_count=idx+1`
- `find_or_add_fn_id` neighbours, `export_names`/`export_count` (3765-66), `param_regs`/`param_count`
  (4259-63), `compile_strategy`/`prof_counter_id` (4534/4551/4705/4721), closure `name` (7725),
  `epistemic` (209/348).

These compile INTO `souc.elf` via `bin/souc`. Confirmed miscompiled by the current `bin/souc`:
- `/tmp/disc.sio` — small (16B, register-passed) AND large (hidden-pointer-passed, Lowerer-like)
  holders BOTH return 0 ⇒ size is NOT the discriminator.
- `/tmp/method.sio` — the EXACT `intern_string` shape (`fn bump(self)->(Holder,i64)`,
  `var lo=self; (*lo.inner).counter=42`, tuple return) returns 0/0 ⇒ `souc.elf`'s `intern_string`
  is genuinely broken.

⇒ `souc.elf` builds CORRUPT IR (lost string/export/param counts) — consistent with its observed
brokenness (`--native-compile` writes no file, `--native-v2-compile` parse-fails). **My fix to
`bin/souc` makes these stores work**, fixing a real on-path contributor to the self-host wall.

### Honest scope (the c634b38f class has ≥4 distinct members)
- (A) explicit-deref STORE through a complex base `(*lo.module).X=v` → discarded copy — **MY FIX**.
- (B) AUTO-deref READ `self.module.X` (no explicit `(*)`) → materializes the 96MB IrModule → the
  documented front-half SIGSEGV (memory `project_lowercodegen_oom_2026-06-08`). lower.sio dodges
  this with explicit `(*self.module)` reads; residual auto-derefs remain. NOT my fix.
- (C) by-value IrModule (~96MB) copies → fault/OOM. NOT my fix.
- (D) branch-target / `e8 00000000` reloc (the 0deb43bcb writer bug). NOT my fix.

My fix is necessary-but-not-sufficient for the wall: it removes a real STORE-corruption
contributor, but (B)/(C)/(D) remain and must be addressed (and bundled) before gen2==gen3.

### Token-safety of the new detector (verified vs tokenizer 5567-5584)
`==`→21, `!=`→22, `<=`→25, `>=`→26, `=>`→52, `=`→16. Detector fires only on `after==16` (`=`) or
`41` (`[`) ⇒ cannot false-match comparisons, fat-arrows, or method calls `(*a.b).c(...)`.

## Phase 3 (D) — call-relocation table overflow FIXED (lean_single, bundled)

`PATCH_OFF/FN/KIND[65536]` is the WHOLE-PROGRAM direct-call relocation table (reset once per
backend in `compile_all` 23450 x86 / 31044 a64), and the 12 write sites
(`emit_direct_fn_call_x86` etc.) wrote **UNCHECKED**: `PATCH_OFF[PATCH_COUNT]=CL; PATCH_COUNT++`
with no bound guard. Each direct call also emits `e8` + `em32(0)` (rel32=0 placeholder) and relies
on the patch entry to fill the displacement later.

If `PATCH_COUNT` exceeds 65536, the writes go **out of bounds** into adjacent globals (corruption)
and those calls' relocations are lost ⇒ the `e8 00000000` stays unpatched — **exactly the recent
session's symptom** + SIGBUS in the full-IR `--native-compile` build.

Scale: the self-hosted source bundle has **~127k call-like tokens** (>> 65536), so main.sio
plausibly overflows the table. (Exact emitted-patch count needs the build to measure — UNCONFIRMED
— but the unchecked write is a genuine latent buffer overflow regardless.)

**Fix (commit pending):** grew `PATCH_*` 65536 → **524288**, added `patch_cap()` +
`record_call_patch(fn_idx, kind)` (bounded, `note_limit_error(3)` on overflow = loud, not silent),
replaced all 12 write sites. For `PATCH_COUNT < cap` behavior is byte-identical to the old inline
code ⇒ no regression; only overflow changes from silent OOB corruption to a hard error. BSS-resident
arrays ⇒ frame-safe (no by-value materialization). Bundled into the same rebootstrap as the (A) fix.

### Remaining wall members still open
- (B) auto-deref READ of the 96MB IrModule, (C) by-value IrModule copies — the documented
  architectural front-half SIGSEGV (heap-indirect IR storage; job 2356 /
  [[project_lowercodegen_oom_2026-06-08]]). NOT surgical, NOT bundleable as a lean_single codegen
  fix — a separate workstream. (Empirical note: an 8MB auto-deref read does NOT fault under
  bin/souc — `/tmp/bread.sio` returns correct values — so the fault is specific to the 96MB
  scale / repeated copies, consistent with the by-value-elimination diagnosis, not a generic
  read-materialization codegen bug.)
- The original 0deb43bcb branch-target writer miscompile — may be (partly) explained by the PATCH
  overflow (D) corrupting code/branch bytes; re-test after the (A)+(D) rebootstrap before treating
  it as a separate unsolved bug.

### Cap-sweep result (other whole-program fixed arrays in lean_single)
- `FN_*[65536]` — safe (main.sio ~6642 fns ≪ 65536).
- `FNSIG_*[256]` (codegen signature table, deduped by shape, indexed `sig*16` into
  `FNSIG_PARAM_TY[4096]`) — already **guarded** (`if sig >= 256 { return 0 }`, 4580): a bounded
  soft fallback (wrong-sig at scale), NOT OOB corruption. Whether main.sio has >256 distinct
  signatures is UNKNOWN; growing it is an index-web (must grow `FNSIG_PARAM_TY` too). NOT grown
  blind. **Rebootstrap-time candidate:** if the rebuilt compiler shows signature/type-resolution
  errors, grow the whole FNSIG family together (256→e.g. 8192, PARAM→×16) + loud guard.

## Bundle status for the single rebootstrap
- (A) deref-field store through complex base — committed `22dece7b9`, confirmed on wall path.
- (D) call-relocation table guard+grow — committed `7b3017474`, plausibly the unpatched-`e8` cause.
- Both are x86; a64 mirrors deferred (not needed for x86_64 self-host).
- B/C and the FNSIG family are NOT in this bundle (architectural / unverified-index-web).
- NEXT: 3-stage SLURM rebootstrap (confirm submission path w/ operator) → re-run repros
  (expect `repro_boxstore`/`min` = 42/42/55/55) + `--native-compile main.sio` should get further
  (watch for `error: too many call relocations` = D guard firing = need bigger cap) +
  `canonical_compiler_gate` + `release_gate` + run-pass 525/0.
