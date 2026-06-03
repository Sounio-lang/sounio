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

> **CAVEAT (do not over-trust the "two independent bootstrap bugs" framing):** the ac08e3b8 void-main
> corruption was **NOT root-caused**, and **no bootstrap has yet produced a validated module→ELF**
> (under c634b38f the writer was broken so there was no artifact to inspect; under ac08e3b8 the
> lowering corrupted). An equally-live hypothesis is that the **lowered void-main module is itself
> malformed** — a corrupt `Name` buf/len (exactly what the wild-string env-dump implies) or a main
> with no real `ret` — which the two bootstraps merely *surface* differently (c634b38f's writer
> silently no-ops on it; ac08e3b8's path prints garbage). Before hunting bootstraps, **dump the
> lowered module and validate main's name + that it ends in a `ret`.**

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
