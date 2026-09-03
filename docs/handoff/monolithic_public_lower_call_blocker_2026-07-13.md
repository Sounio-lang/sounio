<!-- docs:meta
topic_id: repo.docs.handoff.monolithic-public-lower-call-blocker-2026-07-13
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.monolithic-public-lower-call-blocker-2026-07-13
-->

# Monolithic Public Lower Call Blocker

Date: 2026-07-13
Base: `3b6ad8e22a7e626fe7c7349ecad6a07eacc31ebd`
Review baseline commit: `6ceb0766f36567d4c8f1fea0cd5e92083575bd52`

## Claim Boundary

The supported claim is narrow: the native diagnostic Madaros process enters
the public `lower_program_to_ir` call and terminates with `SIGSEGV` before the
call-site exit marker. The current evidence does not identify a return ABI,
SRET, stack, global, BSS, or float-specific root cause.

`MONO-GLOBAL-F64-001` is blocked by
`BLK-20260713-monolithic-public-lower-call`. No semantic lowering fix was
applied.

## Primary Blocker

```text
Blocker-ID: BLK-20260713-monolithic-public-lower-call
Status: classified
Severity: B1
Class: compiler-semantics
Owner: Codex compiler_unblock lane
Lane: public monolithic lower_program_to_ir call
Worktree: /tmp/sounio-compiler-monolithic-f64-20260713
Branch: codex/compiler-monolithic-global-f64-20260713
Files-Owned: self-hosted/compiler/main.sio; self-hosted/ir/lower.sio; tests/compiler/fixtures/monolithic_public_lower_call/*.sio; tests/compiler/monolithic_public_lower_call_matrix.tsv; scripts/ci/madaros_monolithic_public_lower_call_matrix.sh; scripts/ci/madaros_monolithic_public_lower_call_gate.sh; docs/handoff/monolithic_public_lower_call_blocker_2026-07-13.md
Files-Read-Only: self-hosted/ir/ir.sio
Do-Not-Touch: boxed/flat semantic lowering implementations; sedenion algebra/oracles; bootstrap concatenated sources; primary checkout
Repro: SOUNIO_MADAROS_MONO_PUBLIC_LOWER_BIN=/tmp/sounio-mono-global-f64-baseline-20260713/madaros-before-fix SOUNIO_MADAROS_MONO_PUBLIC_LOWER_GATE_KEEP=1 bash scripts/ci/madaros_monolithic_public_lower_call_gate.sh
Observed: all eight fixtures emit the call-site lower_begin marker, then the native process terminates with SIGSEGV/rc139 before lower_done
Expected: lower_program_to_ir completes without a signal and the call site emits lower_done for all eight fixtures
Acceptance-Gate: SOUNIO_MADAROS_MONO_PUBLIC_LOWER_BIN=<current-source-elf> bash scripts/ci/madaros_monolithic_public_lower_call_gate.sh
Evidence-Level: E3
Evidence: /tmp/sounio-mono-public-lower-call-reviewfix-20260713/matrix/receipt.tsv; /tmp/sounio-mono-public-lower-call-reviewfix-20260713/gate/matrix/receipt.tsv
Fallback-Path: none
Legacy-Kept: yes; boxed and flat semantic paths remain unchanged
LLM-Offload: not-required
Next-Action: obtain final read-only review limited to exact-matrix enforcement and non-perturbative inline unwrap instrumentation; do not select a semantic fix from the boundary classes
```

## Dependent Blocker

```text
Blocker-ID: MONO-GLOBAL-F64-001
Status: proposed
Severity: B1
Class: evidence-gap
Owner: Codex compiler_unblock lane
Lane: public monolithic global BSS and typed-add preservation
Worktree: /tmp/sounio-compiler-monolithic-f64-20260713
Branch: codex/compiler-monolithic-global-f64-20260713
Files-Owned: tests/compiler/fixtures/monolithic_public_lower_call/bss_typed_adds.sio; scripts/ci/madaros_monolithic_public_lower_call_gate.sh
Files-Read-Only: self-hosted/ir/lower.sio; self-hosted/ir/ir.sio
Do-Not-Touch: boxed/flat semantic lowering implementations; sedenion algebra/oracles; bootstrap concatenated sources; primary checkout
Repro: SOUNIO_MADAROS_MONO_PUBLIC_LOWER_BIN=<current-source-elf> bash scripts/ci/madaros_monolithic_public_lower_call_gate.sh
Observed: BLK-20260713-monolithic-public-lower-call terminates the process before BSS and typed-add counts can be observed
Expected: bss_globals=3, bss_bytes=144, f64_adds=1, and i64_adds=1 through public lower_program_to_ir
Acceptance-Gate: SOUNIO_MADAROS_MONO_PUBLIC_LOWER_BIN=<current-source-elf> bash scripts/ci/madaros_monolithic_public_lower_call_gate.sh
Evidence-Level: E0
Evidence: dependent row tests/compiler/fixtures/monolithic_public_lower_call/bss_typed_adds.sio is versioned, but its semantic diagnostic is not yet reachable
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: keep blocked by BLK-20260713-monolithic-public-lower-call; do not apply the hypothesized preseed fix until the public call completes
```

## Matrix Contract

The versioned manifest contains exactly eight labels: empty, main-only, local
integer add, local float add, one integer global, one float global, BSS without
binop, and combined BSS plus typed adds. The launcher requires an explicit ELF
through `SOUNIO_MADAROS_MONO_PUBLIC_LOWER_BIN`; it never resolves a generic or
checked-in prebuilt. It records ELF size and SHA-256, identity exit and identity
SHA-256, and per row label, source, exit code, call begin/done, and last internal
boundary. Before execution, it compares the manifest against the exact eight
ordered label/source rows, so deletion, duplication, reordering, or path drift
fails closed.

## Baseline Receipt

The pre-instrumentation diagnostic ELF was built before any semantic
`self-hosted/ir/lower.sio` change. Build wall time was 3:09.71, maximum RSS was
513180 KB, size was 98411754 bytes, and SHA-256 was
`9c583ca2cee68ffaa16b1e5a5d1fe9f5bad1d660927b79dc48e43bee0264bd5b`.

The current-source diagnostic ELF was built from provisional tree
`5ee412796c0fba1b4ae050f7cf5c2e4611bfcc2a`. Build wall time was 3:09.70,
maximum RSS was 513180 KB, size was 98413295 bytes, and SHA-256 was
`37b8bbeea0bba06b6b9c6e7c574859aecfaa6dec5464138e45e1cdcdac6c525e`.
Its identity command exited zero and the identity output SHA-256 was
`c558884336518aafb594700b1346f0873ed1bfe50e7d6f15bdf0e9d785e79cc5`.

The first instrumented build from provisional tree `5ee41279` used an
intermediate local while separating boxed unwrap from result construction. Its
fine-grained final boundary was review-classified as perturbative and does not
support a causal inference.

The reviewed non-perturbative compiler source was rebuilt from provisional tree
`8111699f7fa99e19b55b08726ad37d6cd97fedc0`, with the original inline
`module: *boxed.module` expression restored. Build wall time was 3:10.53,
maximum RSS was 513180 KB, size was 98413103 bytes, and ELF SHA-256 was
`7e6203c08b254ea46cbae4094a0109317ac28063e4bb3b84209d51eeef2ae8fa`.
The identity command exited zero and its output SHA-256 was
`c558884336518aafb594700b1346f0873ed1bfe50e7d6f15bdf0e9d785e79cc5`.

All eight reviewed rows were `rc=139`, `begin=yes`, and `done=no`. Empty,
global-i64, and global-f64 recorded `boxed_unwrap_result_begin`; main-only,
both local-add rows, BSS without binop, and combined BSS/typed-adds recorded
`preseed_begin`. The two observed boundary classes are recorded without a
causal ordering or root-cause claim.

## Handoff Contract

```text
Current-SHA: this diagnostic commit (resolve exactly with git rev-parse HEAD); review baseline was 6ceb0766f36567d4c8f1fea0cd5e92083575bd52
Current-Branch: codex/compiler-monolithic-global-f64-20260713
Current-Worktree: /tmp/sounio-compiler-monolithic-f64-20260713
Dirty-Status: clean after the diagnostic amend; no semantic fix and no generated artifact in the worktree
Owned-Files: self-hosted/compiler/main.sio; self-hosted/ir/lower.sio; tests/compiler/fixtures/monolithic_public_lower_call/*.sio; tests/compiler/monolithic_public_lower_call_matrix.tsv; scripts/ci/madaros_monolithic_public_lower_call_matrix.sh; scripts/ci/madaros_monolithic_public_lower_call_gate.sh; docs/handoff/monolithic_public_lower_call_blocker_2026-07-13.md
Do-Not-Touch: primary checkout; bootstrap concatenated sources; boxed/flat semantic lowering; sedenion algebra/oracles
Last-Green-Gates: git diff --check; bash -n scripts/ci/madaros_monolithic_public_lower_call_matrix.sh scripts/ci/madaros_monolithic_public_lower_call_gate.sh; focused current-source Madaros build rc0
Failing-Gates: explicit current-source ELF public lower call matrix and gate fail because all eight rows terminate inside the call before lower_done; last_boundary is heterogeneous
Open-Blockers: BLK-20260713-monolithic-public-lower-call B1 E3; MONO-GLOBAL-F64-001 B1 E0 blocked-by primary
Artifacts: /tmp/sounio-mono-global-f64-baseline-20260713/ for the pre-instrumentation baseline; /tmp/sounio-mono-public-lower-call-current-20260713/ for the perturbative review run; /tmp/sounio-mono-public-lower-call-reviewfix-20260713/ for the reviewed build, identity, matrix, and gate receipts
Next-Command: SOUNIO_MADAROS_MONO_PUBLIC_LOWER_BIN=/tmp/sounio-mono-public-lower-call-reviewfix-20260713/madaros-diagnostic SOUNIO_MADAROS_MONO_PUBLIC_LOWER_MATRIX_DIR=/tmp/mono-public-lower-final-review-matrix bash scripts/ci/madaros_monolithic_public_lower_call_matrix.sh
```
