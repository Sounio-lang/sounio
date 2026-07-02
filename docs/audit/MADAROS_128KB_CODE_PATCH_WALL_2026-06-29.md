<!-- docs:meta
topic_id: repo.docs.audit.madaros-128kb-code-patch-wall-2026-06-29
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-128kb-code-patch-wall-2026-06-29
-->

# Madaros forensic dispatch — the 128KB code-patch wall (large native compilation)

Date: 2026-06-29
Branch: `research/solver-ts3-parallel` (= `recover/solver-gpu-arc`, tip `8fa7cea06`)
Class: hard capacity wall + latent silent-miscompile at scale
Status: root-caused + reproduced; fix is a multi-site campaign, NOT yet applied.
Prereq: pairs with the Debt#4 4MB ELF buffer (preserved on branch `recover/debt4-4mb-wip`,
commit `481f3e9de`) — both must land together.

## The cap chain (discovered this session)

For a native-compiled module, three caps stack. The first two are now handled; the third
is this dispatch:

1. **IR_MAX_INSTRS = 1024** per function — FIXED (`a6c9a596f`, loud reject). Forces the
   DIMACS corpus generator to ≤32 clauses/block (`8fa7cea06`).
2. **256KB ELF file buffer** (`native_v2_write_min_elf64_to_file`) — Debt#4 lifts it to 4MB
   (on `recover/debt4-4mb-wip`). Correct + byte-verified.
3. **128KB `NativeCompiler.code` buffer + patch path** — THIS. The real binding wall for
   large *correct* programs.

**The profound finding:** the rc=13 ELF cap (#2) was only ever reached via *silently
truncated* blocks (#1). Now that truncation is correctly rejected, large correct programs
hit #3 *before* #2 — so Debt#4 alone converts a loud rc=13 into a **crashing** large binary.

## The bug

`NativeCompiler.code` is `CodeBuffer { bytes: [i8; 131072], len }` — struct defined in
`self-hosted/native/encode.sio:8`, `encode_core.sio:11`, `contract.sio:33`,
`codegen.sio:90`, `codegen_x86_linux.sio:108`. It is held **by value** in `NativeCompiler`;
widening the struct is the aggregate-pressure typecheck trap (a prior global CodeBuffer
widening failed the self-hosted build) — DO NOT widen the struct.

Machine code is emitted via `nc_emit_byte` to BOTH the `.text` mirror `NATIVE_V2_TEXT_BUF`
(widened to 4MB by Debt#4 — this is what the ELF `.text` is copied from) AND `code.bytes`
(guarded `< 131072`). BUT later back-patching — label resolution, branch/jump-offset
patching, relocation apply, trampoline — writes patched bytes to `code.bytes[pos]` only for
`pos < 131072`. For `pos >= 131072` the byte was never written to `code.bytes` AND the patch
never reaches the 4MB mirror. So a module with >128KB of code emits an ELF whose `.text`
beyond ~128KB is DENSE (real bytes) but has **unpatched jump/branch targets → runtime SIGSEGV**.

(There are ~13 `code.bytes` references in `codegen_x86_linux.sio`; the WRITE/PATCH ones are
the targets. The working precedent is `NATIVE_ELF_BUF` — a 16MB global at
`codegen_x86_linux.sio:40` — with `nc_elf_put_*` at ~9876 indexing it directly.)

## Reproduction (on `recover/debt4-4mb-wip` + its rebuilt madaros)

```bash
# generator block_size=32 already on 8fa7cea06; craft a 400-var/600-clause CNF (seed 9),
# run scripts/research/generate_sounio_dimacs_harness.py, then:
./bin/souc compile mid600.sio -o mid600.elf   # compiles -> 884744-byte (864KB) ELF
./mid600.elf                                   # exit 139 (SIGSEGV) despite dense .text
```
`.text` tail above offset 262144 is 72–78% nonzero (NO zero-cliff — it is the *patches*,
not zero-fill, that are missing). `test_smt` harnesses (<147KB) run fine. A 4096-clause
corpus (block_size=32 → 128 blocks → **205 merged functions**) crashes the **compiler**
itself (exit 139 after "Merged IR: 205 functions") — likely a deeper memory/function-count
limit (the by-value `IrModule` is `[IrFunction;2048]` ≈540MB) to investigate alongside.

## Proposed fix (multi-site campaign)

Mirror-ize the patch path: every code-byte PATCH (label / branch-offset / relocation /
trampoline) must ALSO write the patched byte into `NATIVE_V2_TEXT_BUF[pos]` (4MB), so the
full `.text` up to 4MB is correctly patched. Prefer routing all code-byte writes/patches
through ONE helper that writes `code.bytes` (if `<131072`) AND `NATIVE_V2_TEXT_BUF` (if
`<4194304`) — and that READS from the mirror for read-modify-write patches at `pos>=131072`.
Alternative: make the mirror the *primary* patch target (adopt the `nc_elf`/`NATIVE_ELF_BUF`
direct-global pattern), treating `code.bytes` as a vestigial <128KB shadow. Apply WITH the
Debt#4 4MB ELF buffer (cherry-pick `recover/debt4-4mb-wip`).

## Verification gate

1. `test_smt` 6/6 (no regression). 2. small programs print correctly. 3. the 600-clause
corpus compiles to ~864KB AND **RUNS** to a real `SOUNIO_DIMACS_RESULT 0|1` (was SIGSEGV) —
the primary gate (println(i32) is already fixed so RUNs work). 4. STRETCH: the 4096-clause
corpus compiles AND runs (also needs the 205-function compiler crash resolved).

## UPDATE 2026-06-29 — PRECISE root cause (direct investigation)

The "patch path" framing above is partly a red herring. With Debt#4 applied, the WRITE/PATCH
paths are already correct to 4MB: `nc_emit_byte` (1394) dual-writes mirror+code.bytes;
`nc_patch_u32_le` (1649) mirrors via `native_v2_text_patch_u8` (now `<4194304`);
`apply_relocations_into` (4653) routes ALL relocs (call/rodata/data/fn-rip) through
`nc_patch_u32_le`; the reloc applicability checks (`native_v2_reloc_is_call_patch` 4638,
`is_rip_disp_patch` 4643) read via `native_v2_text_byte_at` (mirror, `<4194304`). Yet the
864KB corpus still SIGSEGVs. The actual gap:

**The by-value BUILTIN emit path truncates at the 128KB `CodeBuffer`.** Builtins
(`print_int`/`print_f64`/etc.) are emitted by the by-value `emit_*` helpers (encode.sio)
into a `CodeBuffer { bytes:[i8;131072] }` COPY, then `native_v2_persist_builtin_emit_into`
(codegen_x86_linux.sio:3674) syncs them to the mirror via `native_v2_text_sync_from_narrow`
(1350), whose loop at **line 1357** is `while i < (*nc).code.len && i < 131072 && i < 4194304`
— bounded at 131072. Debt#4 added the `4194304` clause but CANNOT drop `i < 131072` because
the SOURCE (`code.bytes`) is the 131072 by-value buffer. So any builtin emitted at code
offset ≥131072 in a large module is **truncated at the source, before the mirror** → wrong
`.text` → crash. (The dense-but-wrong `.text` is the user-function code via nc_emit_byte;
the wrong spots are where builtins landed beyond 128KB.)

This IS the aggregate-trap `CodeBuffer` (the struct Debt#4 deliberately did not widen). The
mirror widening cannot rescue bytes lost in the 128KB by-value buffer upstream.

**The real fix (a major refactor, two options):**
- (A) Make `CodeBuffer.bytes` a module-level GLOBAL like the working 16MB `NATIVE_ELF_BUF`
  (codegen_x86_linux.sio:40), so the by-value emit no longer caps at 131072. Touches the
  struct in encode.sio/encode_core.sio/contract.sio/codegen.sio/codegen_x86_linux.sio — risk
  the documented by-value-aggregate typecheck pressure, but a global (not by-value field)
  sidesteps it (NATIVE_ELF_BUF proves multi-MB globals work).
- (B) Re-route every builtin emitter (emit_builtin_print_int/_f64/...) through the
  mirror-aware `nc_emit_byte` instead of the by-value `CodeBuffer emit_*` helpers.
Either way, pair with Debt#4 (recover/debt4-4mb-wip) and verify the 600-clause corpus RUNS.

## Why deferred

This is a multi-site codegen campaign on the aggregate-trap `CodeBuffer`. Three consecutive
multi-agent workflow attempts were killed by transient API/infra errors mid-stream (the
read-only investigation + the fix). It deserves a fresh session with stable infra, done
directly/incrementally with a rebuild+verify per change. Prereqs are preserved on origin.
