# source→ELF: registration fixed; bootstrap-codegen quagmire is the wall (2026-06-03)

Branch `modular/native-v2-source-to-elf` @ `0f1b08dbf` (worktree `/workspace/sounio-srcelf`, off
`modular/native-v2-e2e-gate`). Goal: modular compiler emits a runnable ELF from `.sio` via
source → IR → native-v2 backend → ELF. **Not pushed.** 6 mc builds this session.

## What works (verified)

1. **Registration fix** (`0f1b08dbf`, `self-hosted/ir/lower.sio`): `find_or_add_fn_id`'s
   `(*lo.module).functions[idx]=..` / `(*lo.module).fn_count=..` are inline derefs of an **owned
   struct's Box field**, which bin/souc miscompiles to a **discarded copy** → source→IR produced
   `fn_count=0` (the lowering "fn_count<=0 → IR lowering failed" path). Fix: bind the Box to a local,
   store through `(*m)`. **Confirmed `REG_FNCOUNT=1`** via a crash-surviving file-write diagnostic.
   - The `(Lowerer,i64)` tuple-SRET is **NOT** the cause. Build #1 (by-value path) reached the
     `fn_count<=0` check with **no null-deref**, proving the tuple return preserves the module pointer.
     A detour routing registration through the `*mut` family bought an unpinned SIGSEGV — reverted.
   - Store-shape rule (probes /tmp/probe_{shapes,large,owned,byval_tuple,copyrebox}.sio): storing
     through a deref of an **inline expression** (`(*x.box).f=v`, `(*(*p).box).f=v`) writes a discarded
     copy; binding the pointer to a `let` local first (`let m=x.box; (*m).f=v`) lands it. Inline-deref
     *reads* and copy-rebox (`var c=*box; ..; box=Box::new(c)`) are fine. Never auto-deref (`m.f`).
2. **Pipeline reaches the backend, no crash**: with the fix, void `fn main(){}` lowers end-to-end
   ("Merged IR: 1 functions"; seam prints "emitted fns=1"). The feared body-stage tuple wall did not
   *crash* for an empty body — but the IR it produced was **not validated** (see caveat below);
   "emitted fns=1" proves only `fn_count==1`, not that main's body/name are sound.
3. **Back-half (IR→ELF) confirmed**: `--native-v2-emit13` on **ac08e3b8**-built mc writes an 8200-byte
   ELF that **exits 13**. Seam `--native-v2-compile <src> -o <out>` wired in `main.sio`.

## The wall: bootstrap-dependent codegen (NO single souc compiles the whole path)

| bootstrap | builds mc? | ELF writer in mc | void-main lowering |
|---|---|---|---|
| `c634b38f` (tuple-match FP) | yes | **BROKEN** (write_file ok, no file; emit13 too) | OK (fn_count=1, "emitted fns=1") |
| `ac08e3b8` (`bin/souc-linux-x86_64`, branch's real bootstrap) | yes | **OK** (emit13 exits 13) | **BROKEN** (prints a wild string = process env/argv; no ELF) |
| `d26657dd` (committed `bin/souc` at some commits) | **NO** ("too many locals max 1024", pre-cap-raise) | — | — |

So **full void→ELF is not achieved**. The writer works *standalone* with c634b38f — its mc failure is
an at-scale/in-context miscompile. **Do not pin `c634b38f` to build mc** (it breaks the writer); use
`bin/souc-linux-x86_64` (ac08e3b8). My branch reverts an incorrect c634b38f pin.

> **RESOLVED (dumped main's lowered IR via a temporary in-seam diagnostic):** under **c634b38f** the
> void-main module is **SOUND** — `name=main` (len 4), `instr_count=2`, **ends in `ret`**
> (last_op_is_ret=YES), backend reports `emitted fns=1`. The "malformed module / missing ret"
> hypothesis is **REFUTED**: the lowering (incl. the registration fix) is correct. The wall is
> **bootstrap-dependent codegen in DISTINCT subsystems**:
> - `c634b38f`: sound module in, but the **backend ELF-writer** miscompiles in mc (`write_file`
>   returns ok, no file) — even emit13 doesn't write. The writer works *standalone* with c634b38f.
> - `ac08e3b8`: the **checker** miscompiles — void main raises a spurious
>   `error[E008] expected i64, found i8`, then SIGSEGV + ~7MB garbage. (Its writer is fine: emit13
>   exits 13.)
>
> So NEXT = a backend-writer fix (or a bootstrap whose writer survives mc; the sound module is already
> in hand) and/or the E008 checker miscompile under ac08e3b8 — **not** a lowering problem.

## c634b38f writer miscompile — CHASED TO GROUND (2026-06-03)

Instrumented `native_v2_write_min_elf64_to_file` with print markers + ran emit13 on c634b38f-built mc:
- `WRITER ENTER` ✓, `POST-LOCAL` ✓, `W-A`(post file_len-check) ✓, `W-B`(post ELF-header puts) ✓,
  `W-C`(post phdr emits) ✓, `W-C1`(post code fill-loop) ✓ — but `W-D`(post rodata fill-loop) ✗, and the
  `write_file` call is never reached, yet the function returns rc=0 (caller prints "emitted") and writes
  no file. So control **jumps to the function epilogue from the fill region** (~codegen_x86_linux.sio
  7507-7518), skipping `write_file`.
- **It is a c634b38f BRANCH-TARGET codegen miscompile, not a data bug:** the bound is sane
  (`code_len=577`), no fault/crash; and for emit13 `rodata_len=0` so **loop 2's body never executes** —
  yet the divergence is in the loop-2 *region*, i.e. the emitted labels/branches there, not any run
  logic. Removing the compound `&&` from the loop conditions did NOT help; adding markers moves the
  apparent point — **layout/span-sensitive**, the class memory already adjudicates intractable to fix by
  source perturbation ([[project_modular_span_sensitive_crash]], [[project_modular_B_repro_verdict]]).
- **The SOURCE IS PROVEN CORRECT:** ac08e3b8 compiles this exact writer correctly — emit13 writes 8200
  bytes and **exits 13**. So the defect is purely in c634b38f's codegen, not in the writer source.
- ⇒ NOT fixable by a writer source-workaround.

## ac08e3b8 checker E008 route — ALSO intractable (proven, no build)

Pursued route (b). `mc_ac` (ac08e3b8-built) `--check` spuriously raises **E008 on valid programs
broadly** — `fn f()->i64{5}` and `fn g(){let x=1}` both fail, while c634b38f-mc says OK on both. The
void-main E008 is the slow-path `report_mismatch((*fd).span, 8, sig2.return_type, body_ty)`
(check.sio:1587) with **both** types corrupted; `sig2`/`body_ty` come from `(*c).fn_sigs.get(sig_id)`
(FnSig by value) and `(*c).check_block(...)` — core by-value-struct returns. So ac08e3b8 (2026-05-29)
**pervasively miscompiles the current `*mut`/SRET checker** (transcribed 2026-06-03, *after* ac08e3b8):
it's too old. "Fixing" it would mean un-transcribing the `*mut` checker = reverting the 170-crasher fix.
Not viable.

**Conclusion — both source routes are dead, and no clean bootstrap exists** (only ac08e3b8 + c634b38f
build mc post-cap-raise; all `.prev*` are pre-2026-05-28 → 1024-local-cap → can't build). The milestone
needs a **codegen-level fix in `lean_single.sio` + bootstrap regeneration**. Cleanest target: the
**writer branch-target regression in the c634b38f lineage** (recent, single; c634b38f already compiles
the `*mut` checker correctly) — bisect lean_single codegen changes 05-29→06-03, fix, reach the bootstrap
fixed point, rebuild mc. Multi-session, with bootstrap-brick risk.

## Next steps (multi-session)

- Need a souc that compiles **both** the lowering and the 128KB-stack ELF writer
  (`native_v2_write_min_elf64_to_file`, codegen_x86_linux.sio:7444) correctly — either a cleaner
  bootstrap or fixing the per-bootstrap codegen bugs.
- Latent wall for non-trivial programs: `lower_block_ref` and the expr/stmt lowering are all
  `-> (Lowerer,i64)` tuple-returning by-value (the documented *mut-conversion surface).
- Value-returning `main` additionally needs the E008 lane (`g1/e008-bridge-fix`).

## Coordination

A parallel agent on `/tmp/sounio-nv2-integ` is working the **same bridge** (`ret13.sio`,
value-returning main, "let-module idiom" = the same bind-deref). Converge, don't double-apply.
See memory `project_source_to_elf_2026-06-03` and `project_double_deref_box_store_miscompile_2026-06-03`.
