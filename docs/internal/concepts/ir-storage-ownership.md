<!-- docs:meta
topic_id: repo.docs.internal.concepts.ir-storage-ownership
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.ir-storage-ownership
-->

# IR Storage Ownership

Status: hypothesis; prototype bridge pending executable validation
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
Effects-Changed: heap construction and release require Alloc; existing serializer effects are unchanged.
IR-Changed: storage location only; no opcode, field, identity, or interpretation changes.
Claims-Introduced: a bounded small live IrModule can be constructed and round-tripped without materializing IrModule on stack/BSS.
Claims-Forbidden: IrModuleHeapBridge is canonical or alias-safe; all 2048 functions are materialized; IR capacity is unbounded; by-value IrModule paths are eliminated; this is a general arena allocator.
Assumptions: heap_alloc returns zeroed memory and heap_free accepts the unchanged user pointer; native-v2 stores its mapping header at ptr-8 while libc uses calloc/free directly.
Write-Set: self-hosted/ir/heap_storage.sio, self-hosted/ir/soir_core.sio, self-hosted/ir/serialize.sio, self-hosted/test_ir.sio, self-hosted/compiler/main.sio, scripts/ci/ir_module_heap_bridge_gate.sh, scripts/bootstrap/bootstrap_concat.sh, docs/internal/concepts/*, docs/governance/topic-registry.v1.json, docs/governance/DOCS_AUTHORITY_MATRIX.md, docs/governance/DOCS_ACCEPTANCE_REPORT.md.
Read-Set: self-hosted/ir/ir.sio, stdlib/mem/box.sio, native-v2 heap builtin implementation, SOIR v4/v5 tests.
Positive-Witness: one shared internal witness used by T17 and source-fresh Madaros adds one sparse function, two instructions, and one string; round-trips v5 and v4; preserves state after a truncated decode; and observes one FREED transition across two release calls.
Negative-Witness: allocation failure is FAILED; invalid owner/index/capacity access fails closed; truncated staged decode preserves the live owner; second release is a FREED no-op; serializer capacity preflight remains authoritative; in-memory opcodes without stable SOIR tags leave serialized length zero.
Acceptance-Gate: source-fresh modular compile plus scripts/ci/ir_module_heap_bridge_gate.sh PASS from the raw ELF, shared T17 witness PASS, and scripts/ci/madaros_visibility_context_gate.sh PASS.
Integration-Target: stacked after draft #870 and current main.
Authoritative-Only-If: source-fresh compiler executes T17 and v4/v5 identity assertions with no fallback.
```

The reservation is two GiB of virtual address space, not a claim that two GiB
of physical memory is initialized. Linux anonymous mappings are demand-paged.
The bridge initializes top-level counts, but runtime evidence in issue #882
shows that native-v2 lowers array-of-struct fields as pointer tables and the
zeroed reservation does not materialize each live function node. The fixed
reservation and graph ownership must be revisited before this bridge can pass.
The transactional decoder is designed to hold two virtual reservations so it
can adopt the staged pointer on success and leave the live owner unchanged on
failure; runtime proof remains blocked by that missing materialization.

The bridge owner is an explicit runtime state machine with private fields.
`ir_module_heap_release(&! owner)` transitions `ALLOCATED` to `FREED`, nulls
the pointer and byte count, and only then passes the original user pointer to
`heap_free`. A second release observes `FREED` and cannot free again. This is
exactly-once per owner instance, not a claim of alias-safe ownership: copying
or aliasing `IrModuleHeapBridge` is forbidden and remains unverified until the
checker accepts linear borrowed or state-threaded owners.

```text
Semantic-Outcome: PARTIAL; the bounded SOIR core resolves source-fresh compiler emission, but runtime validation exposes an independent pointer-graph materialization blocker before the first live IrFunction is initialized.
Concept-Status-Before: absent
Concept-Status-After: hypothesis with source-fresh compiler build and a gate-bound runtime blocker
Distinctions-Added: virtual reservation != touched physical memory; storage ownership != IR semantic identity; per-owner exactly-once != alias-safe linear ownership
Distinctions-Preserved: DefId provenance; SOIR v5 identity; SOIR v4 unknown-provenance compatibility; fail-closed capacity; unsupported writer opcode != invented wire tag
Distinctions-Erased: none
Evidence-Run: exact stacked source build rc=0 in 3:13.20, peak RSS 513180 KiB, ELF size 109085048 and SHA256 684593a1f43f676d47664a7660122d009b5632eaa45b9fa39339681ea67edb09; #854 resolved runtime gate PASS; raw heap gate rc=139; ptrace fault_addr=0x88 at RIP 0x18783bb after loading a null function slot
Fallback-Path: none
Legacy-Kept: canonical inline IrModule and all existing by-value paths
Conflicting-Lanes: none observed by semantic status scanner before edit
Next-Semantic-Interface: issue #882 owns pointer-graph materialization for native-v2 array-of-struct storage; issue #877 repairs linear owner state-threading; issue #878 separately owns unknown reader-tag fail-closed policy
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
Status: classified
Severity: B1
Class: compiler-semantics
Owner: codex/ir-arena-storage-20260714
Lane: IR-HEAP-BRIDGE-V0
Worktree: /tmp/sounio-soir-core-split-20260714
Branch: codex/soir-core-split-20260714
Files-Owned: none; the future materializer lane must declare its write set before editing
Files-Read-Only: self-hosted/ir/ir.sio, self-hosted/ir/heap_storage.sio, self-hosted/native/*
Do-Not-Touch: canonical IR semantic fields, SOIR v5/v4 tags, unknown reader-tag policy
Repro: MADAROS_RAW_BIN=<exact-source-fresh-elf> IR_MODULE_HEAP_BRIDGE_EXPECT_SHA256=684593a1f43f676d47664a7660122d009b5632eaa45b9fa39339681ea67edb09 bash scripts/ci/ir_module_heap_bridge_gate.sh
Observed: raw ELF exits 139 in ir_module_heap_add_function; native-v2 loads functions[index] as an 8-byte pointer slot, obtains null from the zeroed reservation, then faults at defining_module_id offset 0x88
Expected: live function and instruction nodes are materialized and owned before field access, and the raw gate emits the exact PASS receipt
Acceptance-Gate: source-fresh exact-SHA raw heap gate plus #854 resolved runtime gate
Evidence-Level: E3
Evidence: https://github.com/Sounio-lang/sounio/issues/882; ptrace RIP 0x18783bb, fault_addr 0x88, rax=0; fn_count physical offset 0x4000 equals 2048 pointer slots
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: define compiler-owned materialization and destruction for pointer-lowered array-of-struct storage before another runtime attempt
```
