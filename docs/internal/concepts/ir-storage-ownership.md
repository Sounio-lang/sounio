<!-- docs:meta
topic_id: repo.docs.internal.concepts.ir-storage-ownership
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.ir-storage-ownership
-->

# IR Storage Ownership

Status: hypothesis; canonical graph materializer reaches the runtime witness; semantic assertion localization pending
Concept-ID: `SOUNIO-IR-STORAGE-OWNERSHIP`

This lane makes ownership of an out-of-line `IrModule` allocation explicit
without changing the canonical `IrModule`, `IrFunction`, `IrInstr`, DefId, or
SOIR representations. It exists to prove that a small live module can avoid an
inline value measuring hundreds of MiB in this stack and exceeding 1 GiB in
adjacent compiler snapshots, while the broader IR representation remains
unchanged.

```text
Semantic-Lane-ID: IR-HEAP-BRIDGE-V0
Owner: codex/ir-arena-storage-20260714
Concept-IDs: SOUNIO-IR-STORAGE-OWNERSHIP
Intent-Preserved: IR identity and scientific provenance must not be erased by a storage migration.
Transformation: selected IrModule construction and SOIR roundtrip use one explicit heap owner over the unchanged fixed IrModule ABI.
Types-Changed: adds state-machine IrModuleHeapBridge; does not change IrModule, IrFunction, IrInstr, or DefId.
Effects-Changed: heap construction now requires Alloc, Mut, Panic, and Div because the canonical MB-scale epistemic initializer is boxed; release requires Alloc and Mut; existing serializer effects are unchanged.
IR-Changed: storage location only; no opcode, field, identity, or interpretation changes.
Claims-Introduced: a bounded small live IrModule can be constructed and round-tripped without materializing IrModule on stack/BSS.
Claims-Forbidden: IrModuleHeapBridge is canonical, alias-safe, or production-ready for repeated use; all 2048 functions are materialized; managed graph handles are individually freed; IR capacity is unbounded; by-value IrModule paths are eliminated; this is a general arena allocator.
Assumptions: heap_alloc returns zeroed memory and heap_free accepts the unchanged user pointer; native-v2 stores its mapping header at ptr-8 while libc uses calloc/free directly; canonical aggregate storage is compiler-runtime-owned, retained by the bootstrap aggregate pool to process exit and by native-v2 RuntimeContext until an empty-frame whole-context reset or process exit.
Write-Set: self-hosted/ir/ir.sio, self-hosted/ir/heap_storage.sio, self-hosted/ir/soir_core.sio, self-hosted/ir/serialize.sio, self-hosted/test_ir.sio, self-hosted/compiler/main.sio, scripts/ci/ir_module_heap_bridge_gate.sh, scripts/bootstrap/bootstrap_concat.sh, docs/internal/concepts/*, docs/governance/topic-registry.v1.json, docs/governance/DOCS_AUTHORITY_MATRIX.md, docs/governance/DOCS_ACCEPTANCE_REPORT.md.
Read-Set: stdlib/mem/box.sio, native-v2 heap builtin implementation, SOIR v4/v5 tests.
Positive-Witness: one bounded shared internal witness used by T17 and source-fresh Madaros adds one materialized function, two canonical instruction nodes, and one string; checks exact function sentinels; round-trips v5 and v4; preserves state after a truncated decode; and observes one FREED transition across two release calls.
Negative-Witness: allocation failure is FAILED; invalid owner/index/capacity access fails closed; truncated staged decode preserves the live owner; second release is a FREED no-op; serializer capacity preflight remains authoritative; in-memory opcodes without stable SOIR tags leave serialized length zero.
Acceptance-Gate: source-fresh modular compile plus scripts/ci/ir_module_heap_bridge_gate.sh PASS from the raw ELF, shared T17 witness PASS, and scripts/ci/madaros_visibility_context_gate.sh PASS.
Integration-Target: stacked after draft #870 and current main.
Authoritative-Only-If: source-fresh compiler executes T17 and v4/v5 identity assertions with no fallback.
```

The reservation is two GiB of virtual address space, not a claim that two GiB
of physical memory is initialized. Linux anonymous mappings are demand-paged.
Runtime evidence in issue #882 shows that native-v2 lowers aggregate values as
managed handles, so a zeroed raw function slot is null. The bridge now installs
`ir_empty_function()` before its first field access; decoded and appended
instructions are likewise assigned through canonical values. No physical
offset is duplicated in the bridge. The same rule applies to the top-level
extension aggregates touched by codec preflight and publish: `epistemic` is
installed through its existing boxed canonical initializer and `algebras`
through its canonical empty-table initializer. The latter is now public solely
so the bridge can request that representation without duplicating its fields.

The transactional decoder holds a second virtual reservation, materializes its
function graph there, and adopts the staged pointer only after the complete
v5/v4 payload validates. A failed stage is raw-freed without modifying the live
owner. Its unreachable aggregate storage is retained by the active compiler
runtime. The source-fresh bootstrap ELF uses `lean_single`'s never-freed
aggregate pool; emitted native-v2 has no individual handle release, and its
current slow path only resets the whole managed heap at an empty unpinned
frame. Normal process exit leaves both paths to OS reclamation. The selftest
materializes exactly six function graphs across its live, successful, rejected,
truncated, v4, and unsupported-opcode cases. Therefore this witness is
intentionally bounded and issue #884 blocks any canonical repeated-use claim.

The bridge owner is an explicit runtime state machine with private fields.
`ir_module_heap_release(&! owner)` transitions `ALLOCATED` to `FREED`, nulls
the pointer and byte count, invalidates published instruction/function counts,
and only then passes the original raw user pointer to `heap_free`. Managed
handles are never passed to `heap_free`. A second release observes `FREED` and
cannot free again. This is exactly-once for the raw reservation per owner
instance, not a claim of alias-safe ownership: copying or aliasing
`IrModuleHeapBridge` is forbidden and remains unverified until issue #877 is
resolved.

```text
Semantic-Outcome: PARTIAL; canonical aggregate graph materialization removes the prior null-pointer crash and reaches the bounded runtime witness, but one or more semantic assertions remain false; repeated-use handle reclamation remains blocked by issue #884.
Concept-Status-Before: absent
Concept-Status-After: hypothesis with source-fresh compiler build and a gate-bound runtime blocker
Distinctions-Added: virtual reservation != touched physical memory; raw mmap ownership != RuntimeContext handle ownership; raw exactly-once free != managed handle reclamation; per-owner state machine != alias-safe linear ownership
Distinctions-Preserved: DefId provenance; SOIR v5 identity; SOIR v4 unknown-provenance compatibility; fail-closed capacity; unsupported writer opcode != invented wire tag
Distinctions-Erased: none
Evidence-Run: exact stacked source build rc=0 with out_status=present in 3:13.30, peak RSS 513180 KiB, ELF size 109087244 and SHA256 23264b857ceb4a5daa919cbf7572e14072a1ebc2cdcc4b98497653ce44637aaf; exact-SHA raw heap gate rc=1 with semantic_assertion and no segfault; #854, thin, and SRET gates were not run after the first acceptance gate failed
Fallback-Path: none
Legacy-Kept: canonical inline IrModule and all existing by-value paths
Conflicting-Lanes: none observed by semantic status scanner before edit
Next-Semantic-Interface: issue #882 owns bounded graph materialization validation; issue #884 owns canonical repeated-use handle lifecycle; issue #877 repairs linear owner state-threading; issue #878 separately owns unknown reader-tag fail-closed policy
```

```text
Blocker-ID: BLK-20260714-IR-HEAP-SERIALIZE-CLOSURE
Status: closed
Severity: B1
Class: bootstrap-runtime
Owner: codex/ir-arena-storage-20260714
Lane: IR-HEAP-BRIDGE-V0
Worktree: /tmp/sounio-soir-core-split-20260714
Branch: codex/soir-core-split-20260714
Files-Owned: self-hosted/ir/heap_storage.sio, self-hosted/ir/soir_core.sio, self-hosted/ir/serialize.sio, self-hosted/test_ir.sio, self-hosted/compiler/main.sio, scripts/ci/ir_module_heap_bridge_gate.sh, scripts/bootstrap/bootstrap_concat.sh, docs/internal/concepts/ir-storage-ownership.md, docs/internal/concepts/registry.tsv, docs/internal/concepts/bindings.tsv, docs/governance/topic-registry.v1.json, docs/governance/DOCS_AUTHORITY_MATRIX.md, docs/governance/DOCS_ACCEPTANCE_REPORT.md
Files-Read-Only: self-hosted/compiler/module_frontend.sio, self-hosted/native/codegen_x86_linux.sio
Do-Not-Touch: canonical IrModule/IrFunction/IrInstr layout and existing by-value paths
Repro: ulimit -s 65536; /usr/bin/time -v bash scripts/ci/build_modular_madaros.sh artifacts/ir-heap-bridge/madaros-source-fresh
Observed: extracting the bounded SOIR core reduces the heap dependency closure and the exact stacked source builds without the former seed exit 139
Expected: source-fresh Madaros ELF exists without activating the full epistemic/claim serializer closure
Acceptance-Gate: ulimit -s 65536; SOUNIO_BUILD_LOCK=/tmp/sounio-souc-build.lock /usr/bin/time -v bash scripts/ci/build_modular_madaros.sh <throwaway-output>
Evidence-Level: E3
Evidence: https://github.com/Sounio-lang/sounio/issues/879 records the original failure; corrective build rc=0, ELF size 109085048, SHA256 684593a1f43f676d47664a7660122d009b5632eaa45b9fa39339681ea67edb09
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: closed by the bounded SOIR core split; runtime ownership validation continues under BLK-20260714-IR-HEAP-POINTER-GRAPH-MATERIALIZATION in issue #882
```

```text
Blocker-ID: BLK-20260714-IR-HEAP-POINTER-GRAPH-MATERIALIZATION
Status: partial; runtime semantic assertion localization pending
Severity: B1
Class: compiler-semantics
Owner: codex/ir-arena-storage-20260714
Lane: IR-HEAP-BRIDGE-V0
Worktree: /tmp/sounio-ir-heap-graph-materializer-20260714
Branch: codex/ir-heap-graph-materializer-20260714
Files-Owned: self-hosted/ir/ir.sio, self-hosted/ir/heap_storage.sio, self-hosted/ir/soir_core.sio, scripts/ci/ir_module_heap_bridge_gate.sh, docs/internal/concepts/ir-storage-ownership.md
Files-Read-Only: self-hosted/native/*
Do-Not-Touch: canonical IR semantic fields, SOIR v5/v4 tags, unknown reader-tag policy
Repro: MADAROS_RAW_BIN=<exact-source-fresh-elf> IR_MODULE_HEAP_BRIDGE_EXPECT_SHA256=23264b857ceb4a5daa919cbf7572e14072a1ebc2cdcc4b98497653ce44637aaf bash scripts/ci/ir_module_heap_bridge_gate.sh
Observed: the prior null-slot rc=139 is removed; source-fresh ELF reaches the selftest and exits rc=1 with IR_MODULE_HEAP_BRIDGE_FAIL reason=semantic_assertion
Expected: stage-specific receipts identify the first false assertion, after which the raw gate must emit the exact PASS receipt without fallback
Acceptance-Gate: instrumented selftest identifies the failing stage; then a new source-fresh exact-SHA raw heap gate passes, followed by #854 resolved runtime and thin/SRET quick gates
Evidence-Level: E3
Evidence: https://github.com/Sounio-lang/sounio/issues/882; ptrace RIP 0x18783bb, fault_addr 0x88, rax=0; fn_count physical offset 0x4000 equals 2048 pointer slots
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: in a new stacked diagnostic lane, add unique stage/reason receipts for allocation, add-function sentinels, v5 serialization/deserialization, DefId, truncation preservation, v4 compatibility, and release before any semantic change
```

```text
Blocker-ID: BLK-20260714-NATIVE-V2-HANDLE-LIFECYCLE
Status: classified
Severity: B1
Class: compiler-semantics
Owner: unassigned
Lane: NATIVE-V2-MANAGED-HANDLE-LIFECYCLE
Files-Owned: none in this issue
Files-Read-Only: self-hosted/native/runtime_context.sio, self-hosted/native/gc.sio, self-hosted/native/codegen_x86_linux.sio, self-hosted/ir/heap_storage.sio
Do-Not-Touch: SOIR wire semantics, canonical IR fields, raw heap_alloc/heap_free pointer contract
Repro: repeatedly create, deserialize, and release IrModuleHeapBridge values in one native-v2 process
Observed: raw module mmaps are freed, but canonical aggregate handles append to RuntimeContext heap and handle table without per-owner release
Expected: managed-handle release or rooted runtime collection bounds handle_count and heap_cursor across repeated bridge use
Acceptance-Gate: repeated bridge create/roundtrip/release stress remains bounded while preserving live rooted graphs
Evidence-Level: E2
Evidence: https://github.com/Sounio-lang/sounio/issues/884; emitted allocation appends handles, slow path only performs empty-frame whole-context reset, normal Linux exit calls sys_exit directly
Fallback-Path: bounded one-shot bridge selftest only
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: specify rooted lifetime semantics, then wire release or collector behavior with a repeated-use witness
```
