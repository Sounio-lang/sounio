<!-- docs:meta
topic_id: repo.docs.audit.native-reloc-shape-fail-closed-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.native-reloc-shape-fail-closed-2026-09-26
-->

# Native relocations: patched or refused, never left at `[rip+0]` (2026-09-26)

## Defect

`apply_relocations_into` (`self-hosted/native/codegen_x86_linux.sio`) patched a
recorded kind 2/3/4 RIP-relative relocation only when
`native_v2_reloc_is_rip_disp_patch` recognised the bytes before the disp32, a
four-entry whitelist:

| bytes | instruction |
|---|---|
| `48 8d 05` | `lea rax,[rip+d]` |
| `48 8b 1d` | `mov rbx,[rip+d]` |
| `48 89 05` | `mov [rip+d],rax` |
| `f2 0f 10 05/0d` | `movsd xmm0/xmm1,[rip+d]` |

Every other encoding was skipped with no diagnostic. The site kept its
placeholder disp32 of 0, so a load read the next instruction's bytes, and the
program exited 0 with the wrong answer. Kind 1 (call) had the same silent skip
for a non-`e8` site or an out-of-range target (`call +0` calls the next
instruction). The whitelist excluded `mov rax,[rip+d]` (`48 8b 05`), which the
profiling-counter dump (`emit_mov_rax_rip_disp32`, codegen_x86_linux.sio) emits.

Reported by sleepy-easley on the coordination bus, 2026-09-26 22:01Z.

## Fix

1. `native_v2_reloc_is_rip_disp_patch` accepts every `[rip+disp32]` form whose
   disp32 is the **last** field of the instruction (the only thing the patch
   formula `next_ip = patch_offset + 4` assumes):
   - `REX.W[+R] 8B/89/8D`, modrm `mod=00 rm=101`, any `reg` (`mov rcx`, `mov r8`, ...)
   - `F2 [REX] 0F 10/11`, same modrm (`movsd` load and store)

   An instruction with an immediate after the disp32 (`c7 05 d32 imm32`) does
   not match: its `next_ip` is further on, so patching it would be wrong too.
2. Any relocation still not patched (unrecognised shape, out-of-range target,
   unknown kind) is latched in `NativeCompiler.reloc_unpatched_*` (frame.sio).
   Both x86-64 ELF writers (`native_v2_write_min_elf64_to_file` and the legacy
   `compile_native_finalize_and_write_ref`) then refuse with **rc=25** before
   writing any byte:

   ```
   Error: native relocation left unpatched -- count=1 first: kind=3
   reason=unrecognised-instruction-shape fn_index=0 patch_offset=53
   bytes_before=00 00 00 c7 05 (the site would keep its placeholder 0; ...)
   ```

## Witness

`--native-v2-emit-reloc-shape <shape> <out>` (main.sio) emits a `main` that
stores 37 into a `.data` slot through a recognised `mov [rip+d],rax`, then reads
it back through the shape under test, which carries a kind-3 relocation.
`scripts/ci/native_reloc_shape_gate.sh` (named in ci.yml, run against the
fresh build) accepts only "exit 37" or "rc=25, no ELF".

Measured on the workspace, 2026-09-26:

| shape | patched Madaros (md5 `cdd395da`) | sabotage (md5 `c0043259`) |
|---|---|---|
| 0 `48 8b 0d` mov rcx | exit 37 | exit **72** (0x48, next instruction's byte) |
| 1 `4c 8b 05` mov r8 | exit 37 | exit **76** (0x4c) |
| 2 `c7 05 d32 imm32` | refused, rc=25, no ELF | ELF written, **SIGSEGV** (store to `[rip+0]`, in .text) |
| gate | `NATIVE_RELOC_SHAPE_GATE_OK` | `..._FAIL: shape 0: exit 72` |

Sabotage = the same tree with main's four-entry whitelist restored and the
refusal disabled, i.e. main's behaviour. Both ELFs were built from source with
bare `make build-madaros` (operating principle 15); the baseline is main
2e8b76d31 (md5 `5764851f`).

## Corpus A/B

`scripts/ci/madaros_corpus_regression_gate.sh`, run on SLURM with the exact
ELFs above, 10 jobs each, in refresh mode so the full failure list is kept:

| | node | failures | killed compiles |
|---|---|---:|---:|
| baseline main 2e8b76d31 (`5764851f`) | gpuorangefs-5860 | 187 / 1988 | 0 |
| patched (`cdd395da`) | dl380 | 187 / 1988 | 0 |

The two failure lists are identical: 0 new, 0 fixed, 0 changing kind. No program
in the corpus recorded a relocation the widened acceptor refuses, so the rc=25
refusal fires only on the witness.

Unrelated finding: against the checked-in `tests/madaros_corpus_baseline.txt`
(270 entries), main's own Madaros has 60 failures that file does not list, so
the gate is red on an unmodified main. The comparison above is base-vs-patched
for that reason.

## Reverting

Revert the commit. There is no persistent state: the new `NativeCompiler`
fields are reset in `compiler_new`.
