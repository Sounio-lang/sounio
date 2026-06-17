# GLM 5.1 Madaros BSS/Bootstrap Handoff

Purpose: help the GLM 5.1 lane avoid chasing the wrong root cause while
investigating Madaros build/global/BSS behavior.

## Immediate Correction

The build-process correction is right:

- `scripts/ci/build_modular_madaros.sh` builds Madaros from
  `self-hosted/compiler/main.sio`.
- The seed compiler must be a checked-in `lean_single` ELF.
- The seed must not be the `bin/souc` wrapper, because that wrapper can route to
  Madaros and would create an unverified self-host fixed point.

Evidence:

- `scripts/ci/build_modular_madaros.sh:26-30` says the seed must be the
  `lean_single` bootstrap ELF, never the wrapper.
- `scripts/ci/build_modular_madaros.sh:31-39` skips candidates whose first two
  bytes are `#!`, so wrapper scripts are rejected.

Do not change this script to seed from `bin/souc`.

## Important Misread

The provisional conclusion "lean_single does not have BSS/global handling" is
wrong. The searches probably missed the local naming.

Search these identifiers in `self-hosted/compiler/lean_single.sio`:

- `GL_BSS_SIZE`
- `GL_BSS_BASE`
- `bss_alloc_aligned`
- `local_bss_spill_bytes`
- `RET_AGG_BSS_OFF`

Evidence:

- `self-hosted/compiler/lean_single.sio:870-882` defines the global table,
  `GL_BSS_SIZE`, and `GL_BSS_BASE`.
- `self-hosted/compiler/lean_single.sio:960-964` defines
  `bss_alloc_aligned`.
- `self-hosted/compiler/lean_single.sio:1380-1394` allocates globals by storing
  their BSS offset in `GL[(idx * 4 + 1)]` and incrementing `GL_BSS_SIZE`.
- `self-hosted/compiler/lean_single.sio:1771-1774` spills large struct-return
  destinations into BSS.
- `self-hosted/compiler/lean_single.sio:25061-25072` defines the BSS layout and
  sets Linux/Windows `GL_BSS_BASE = 0x10000000 + 16`.
- `self-hosted/compiler/lean_single.sio:26519-26528` emits an ELF `PT_LOAD`
  RW segment with `p_filesz = 0` and `p_memsz = GL_BSS_SIZE`.

The huge BSS segment seen in Madaros ELFs is consistent with lean_single's
explicit BSS arena model.

## Likely Real Gap

The likely gap is not "lean_single has no BSS." It is:

The modular/native-v2 path may not yet carry the same global/BSS arena semantics
that lean_single uses for module globals, aggregate-return scratch, and large
local spills.

Compare:

- `self-hosted/compiler/lean_single.sio` BSS model above.
- `self-hosted/native/elf_bulk.sio:513-522`, where text/rodata/data segments are
  emitted, but the data segment currently uses `p_filesz = data_len` and
  `p_memsz = data_len`.
- `self-hosted/native/codegen_x86_linux.sio` for current modular codegen address
  lowering and global handling.

## Suggested GLM Next Steps

1. Keep `build_modular_madaros.sh` bootstrap contract intact.
2. Trace where modular codegen represents globals and large static storage.
3. Add/confirm a modular BSS arena abstraction:
   - base virtual address,
   - aligned allocation,
   - global symbol offset table,
   - aggregate-return scratch,
   - large-local spill support if needed.
4. Teach the modular ELF writer to emit a pure BSS `PT_LOAD` when BSS size is
   non-zero:
   - `p_filesz = 0`,
   - `p_memsz = bss_size`,
   - RW flags,
   - page alignment,
   - virtual address matching codegen's absolute/global addresses.
5. Add small witness tests before attempting a self-host fixed point:
   - scalar global read/write,
   - array global read/write,
   - large global or large local spill,
   - struct/aggregate return scratch if relevant.
6. Only after those witnesses pass, retry the larger Madaros/self-host route.

## Current Coordination Note

This handoff intentionally does not edit source files. The shared checkout is
dirty with other agents' WIP, so source changes should happen in the owner's
isolated worktree/lane.
