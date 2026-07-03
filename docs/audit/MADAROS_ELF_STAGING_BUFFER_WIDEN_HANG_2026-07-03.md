<!-- docs:meta
topic_id: repo.docs.audit.madaros-elf-staging-buffer-widen-hang-2026-07-03
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-elf-staging-buffer-widen-hang-2026-07-03
-->

# Madaros forensic dispatch — widening the ELF staging buffer to 16 MiB fixes rc=13 but the rebuilt compiler hangs on the GPU driver

Date: 2026-07-03
Branch: `research/solver-ts3-parallel` @ `1dcdd37b9` (merge of #602)
Class: **ATTEMPTED FIX, REVERTED** — compiled clean, typechecked clean, but the rebuilt
compiler binary hangs (does not crash, does not progress) on the exact input it was meant
to unblock
Status: reverted; not landed; no code change from this dispatch is in the tree

> Third dispatch in this chain, in order:
> `docs/audit/MADAROS_GPU_KERNEL_IR_LOWER_TO_PTX_PTX_MODULE_COMBINATION_2026-07-02.md`
> (#585, fixed) → `docs/audit/MADAROS_GPU_MODULE_COMPILE_OOM_2026-07-03.md` (#602,
> root-caused, not fixed) → this one. After confirming the OOM dispatch's finding
> (`ulimit -v unlimited` gets past the segfault into `rc=13`, the already-documented
> `docs/audit/MADAROS_RC13_ELF_256K_CAP_2026-06-28.md` cap), this dispatch attempted that
> cap's proposed fix. It compiles and typechecks, but the **rebuilt compiler hangs**
> instead of fixing anything — a new, different failure mode, caught safely before
> touching the shared compiler binary.

## Context: the cap had already moved once, silently

`docs/audit/MADAROS_RC13_ELF_256K_CAP_2026-06-28.md` describes a 256 KiB (262144-byte)
ELF staging buffer cap and a precise, coupled two-cluster fix. By 2026-07-03 that doc was
already stale on one point: **someone had already widened the cap to 4 MiB (4194304
bytes)** — `NATIVE_V2_TEXT_BUF`, `NATIVE_V2_ELF_FILE_BUF`, and the `rc=13` guard in
`self-hosted/native/codegen_x86_linux.sio` were all already `4194304`, consistently, with
no dangling 256 KiB reference — undocumented anywhere found in this session. `souc build`
of `self-hosted/gpu/kretikos_emit_epistemic_wmma.sio` (the epistemic WMMA PTX-emitter
driver from #572/#585) still hits `rc=13` at this 4 MiB cap after 330 functions merge
(confirmed via `docs/audit/MADAROS_GPU_MODULE_COMPILE_OOM_2026-07-03.md`'s `ulimit -v
unlimited` repro) — so 4 MiB isn't enough for this driver's compiled output either.

## What was changed (and reverted)

**Critical prerequisite, easy to get wrong**: `4194304` is used in this file for **two
unrelated things** — the buffer-size cap (subject to this fix) and the x86-64 ELF default
load address `0x400000` (must never change; changing it corrupts how the OS maps the
binary). Both happen to equal 4194304 in decimal, which is a coincidence, not a
relationship. Before editing, every one of the 21 occurrences of `4194304` in
`self-hosted/native/codegen_x86_linux.sio` was individually read in context and classified:

- **6 sites are `base_addr`/`entry` arguments** (lines 7480, 9748, 9819, 9863, 9867 —
  the last explicitly commented `// 0x400000`, 10125) — **left untouched**.
- **15 sites are the buffer-size family** (declarations at lines 45, 66; bounds checks at
  1359, 1366, 1374, 1378, 1417, 1426, 9433, 9440; the `rc=13` guard itself at 9506; copy-loop
  bounds at 9567, 9580, 9586, 9594) — changed from `4194304` to `16777216` (16 MiB).

16 MiB was not an arbitrary choice: `self-hosted/native/codegen_x86_linux.sio` already
has two other buffers at exactly this size elsewhere in the same file —
`NATIVE_ELF_BUF: [i8; 16777216]` (line 40) and
`native_compile_result_ok_wide(bytes: [i8; 16777216], ...)` (line 920) — so it matches an
existing convention in this codebase rather than introducing a new magic number.

The separate, legacy `131072`-byte by-value `CodeBuffer`/`NativeCompileResult.bytes`
family (line 124 struct field, and its many consumers) was **deliberately left alone**,
per the RC13 dispatch's own explicit warning that widening that *by-value* struct
previously failed typecheck (large by-value struct copies are exactly the class of bug
`docs/audit/MADAROS_GPU_MODULE_COMPILE_OOM_2026-07-03.md` also documents) — the global
mirror buffers (`NATIVE_V2_TEXT_BUF`/`NATIVE_V2_ELF_FILE_BUF`) are what actually needed to
grow, not this struct.

## Verification steps taken

1. `souc check self-hosted/native/codegen_x86_linux.sio` — 169 `error[E175] function is
   private in its defining module` both **before and after** the edit (identical count,
   confirmed via `git stash`/`git stash pop` A/B comparison). This file cannot be checked
   standalone regardless of the buffer edit (same pre-existing condition as other files
   this session) — the edit introduced no *new* errors, but standalone `check` cannot
   positively confirm correctness here either.
2. **Rebuilt Madaros to an isolated path**, deliberately not overwriting the shared
   `artifacts/self-hosted/madaros`:
   ```bash
   bash scripts/ci/build_modular_madaros.sh /tmp/madaros-elf-cap-test
   ```
   Succeeded: `Madaros ready: /tmp/madaros-elf-cap-test (103429319 bytes)` — comparable
   size to the existing pinned binary (~103.9 MB), `bss=3481549304` (~3.4 GiB, driven by
   this codebase's many large static arrays, not something newly introduced here).
   Build output included pre-existing warnings (`assignment type mismatch` in
   `self-hosted/gpu/spirv.sio`, `match arm type does not match other arms` in
   `self-hosted/parser/exprs.sio`) — not investigated further, likely pre-existing and
   unrelated (not confirmed).
3. **Ran the rebuilt binary against the actual target**:
   ```bash
   ( ulimit -v unlimited; ulimit -s unlimited; timeout 500 \
     /tmp/madaros-elf-cap-test self-hosted/gpu/kretikos_emit_epistemic_wmma.sio \
     -o /tmp/out.elf )
   ```
   Printed through `imported_compile: load_done` and then **never printed
   `imported_compile: typecheck_begin`** (which every prior run, with the *unmodified*
   compiler, always printed within a few seconds). Ran for the full 500 s timeout with no
   further output, no crash, no ELF produced. This is a genuinely different symptom from
   both the SIGSEGV (#602) and the old `rc=13` (pre-widen): a **hang**, not a fast failure.

## What this rules out / narrows

- **Not the buffer-size numbers reaching an inconsistent state at runtime** — if the
  mismatch were, e.g., a stale cached size somewhere, it would more likely crash
  (out-of-bounds write) than hang cleanly at a module-loading boundary.
- **The hang happens before typecheck even starts** (between `load_done` and
  `typecheck_begin`), i.e. during module loading/merging of the raw source text — a stage
  that, per `docs/audit/MADAROS_GPU_MODULE_COMPILE_OOM_2026-07-03.md`, already had a
  measured ~61.5 GiB `VmPeak` footprint *before* this widening. Widening
  `NATIVE_V2_TEXT_BUF`/`NATIVE_V2_ELF_FILE_BUF` from 4 MiB to 16 MiB adds only ~24 MiB of
  static BSS (two buffers × 12 MiB delta each) — negligible next to 61.5 GiB — so the
  buffer size itself is an unlikely direct cause of a *new* hang at *this* stage (these
  buffers aren't even touched until the ELF-writing stage, well after typecheck). This
  suggests either (a) coincidental interaction with something else changing between builds
  in this actively-churning shared checkout (see "Caveats" below), or (b) an
  as-yet-unidentified real bug triggered by the widen that manifests earlier in the
  pipeline than expected. Not distinguished between (a) and (b) — see Next steps.

## Why this was reverted rather than debugged further

- The isolated-build discipline worked exactly as intended: the shared, pinned
  `artifacts/self-hosted/madaros` binary was never touched, so this dead end cost nothing
  beyond investigation time — reverting was a `git checkout <base> -- <file>` away, and
  the test binary/ELFs were local to `/tmp`.
- A hang is a worse failure mode to chase blind than a crash: no core dump, no stack
  trace, no clear "last known good line" the way a SIGSEGV at least gives an instruction
  pointer. Proper triage needs a debugger attached mid-hang (`gdb -p <pid>`, or repeated
  `/proc/<pid>/stack` / `py-spy`-equivalent sampling) — no such tooling was available in
  this environment (confirmed earlier in this session: no `gdb`, `lldb`, `strace`, or
  `ltrace` installed).
- Given two prior dispatches in this same chain already found genuinely large, separate,
  pre-existing problems (module-combination typecheck breakage, then ~61.5 GiB memory
  need), continuing to add more debugging surface on top of an already-uncertain shared,
  actively-churning checkout (see Caveats) risked spending more session budget without a
  tool that could actually observe *why* it hangs.

## Caveats

- This session's checkout is **shared with other concurrent, automated processes** — files
  in `self-hosted/` have been observed changing underfoot between checks multiple times
  this session (unrelated to this dispatch's own edits). The build/hang result here should
  be treated as a snapshot against whatever state the wider tree happened to be in at that
  moment, not a hermetic, fully-reproducible experiment. Re-run in a clean, isolated
  worktree before trusting this finding further.
- `VmPeak` (used in the sibling OOM dispatch) was not re-measured for this specific hang —
  it's possible the process is actually still making slow progress against a very large
  but finite allocation, and 500 s was simply not long enough. Not distinguished from a
  true infinite loop.

## Next steps (not attempted here)

1. Reproduce in a genuinely isolated worktree/container (not this shared checkout) to rule
   out cross-session interference.
2. Install `gdb` (or equivalent) and attach to the hung process to get an actual stack
   trace / instruction pointer, rather than guessing from stdout silence.
3. Bisect the 15-site edit itself: apply just the `NATIVE_V2_TEXT_BUF`/`NATIVE_V2_ELF_FILE_BUF`
   declaration widenings (lines 45, 66) without the bounds-check follow-through, rebuild,
   and see if the hang reproduces with *fewer* of the 15 sites changed — would help
   localize whether one specific bounds check (as opposed to the buffer size itself) is
   responsible.
4. Let the run continue well past 500 s (e.g. 30+ minutes) once in an isolated
   environment, to distinguish "hang" from "very slow" definitively.

## Cross-references

- `docs/audit/MADAROS_RC13_ELF_256K_CAP_2026-06-28.md` — the original dispatch this
  attempted to finish; its own "Cluster A/B must move together" warning was followed
  precisely (all 15 sites moved together), and its warning about the by-value `CodeBuffer`
  struct was heeded (left untouched).
- `docs/audit/MADAROS_GPU_MODULE_COMPILE_OOM_2026-07-03.md` — the immediately-preceding
  dispatch in this chain; its `ulimit -v unlimited` repro is what first surfaced `rc=13`
  as the next blocker after the SIGSEGV, and its VmPeak measurement is the baseline this
  dispatch's "negligible extra BSS" argument is measured against.
- `docs/audit/MADAROS_GPU_KERNEL_IR_LOWER_TO_PTX_PTX_MODULE_COMBINATION_2026-07-02.md`
  (#585) — the first blocker in this chain, already fixed; this dispatch's driver
  (`self-hosted/gpu/kretikos_emit_epistemic_wmma.sio`) is the same one that #585 unblocked
  at the typecheck stage.
