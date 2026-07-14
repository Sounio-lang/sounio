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
Write-Set: self-hosted/ir/heap_storage.sio, self-hosted/ir/serialize.sio, self-hosted/test_ir.sio, self-hosted/compiler/main.sio, scripts/ci/ir_module_heap_bridge_gate.sh, scripts/bootstrap/bootstrap_concat.sh, docs/internal/concepts/*, docs/governance/topic-registry.v1.json, docs/governance/DOCS_AUTHORITY_MATRIX.md, docs/governance/DOCS_ACCEPTANCE_REPORT.md.
Read-Set: self-hosted/ir/ir.sio, stdlib/mem/box.sio, native-v2 heap builtin implementation, SOIR v4/v5 tests.
Positive-Witness: one shared internal witness used by T17 and source-fresh Madaros adds one sparse function, two instructions, and one string; round-trips v5 and v4; preserves state after a truncated decode; and observes one FREED transition across two release calls.
Negative-Witness: allocation failure is FAILED; invalid owner/index/capacity access fails closed; truncated staged decode preserves the live owner; second release is a FREED no-op; serializer capacity preflight remains authoritative; in-memory opcodes without stable SOIR tags leave serialized length zero.
Acceptance-Gate: source-fresh modular compile plus scripts/ci/ir_module_heap_bridge_gate.sh PASS from the raw ELF, shared T17 witness PASS, and scripts/ci/madaros_visibility_context_gate.sh PASS.
Integration-Target: stacked after draft #870 and current main.
Authoritative-Only-If: source-fresh compiler executes T17 and v4/v5 identity assertions with no fallback.
```

The reservation is two GiB of virtual address space, not a claim that two GiB
of physical memory is initialized. Linux anonymous mappings are demand-paged;
the bridge explicitly initializes top-level counts and each inserted live
function. The fixed reservation must be revisited if IR caps or layout grow.
Deserialization temporarily holds two virtual reservations so it can adopt the
staged pointer on success and leave the live owner unchanged on failure.

The bridge owner is an explicit runtime state machine with private fields.
`ir_module_heap_release(&! owner)` transitions `ALLOCATED` to `FREED`, nulls
the pointer and byte count, and only then passes the original user pointer to
`heap_free`. A second release observes `FREED` and cannot free again. This is
exactly-once per owner instance, not a claim of alias-safe ownership: copying
or aliasing `IrModuleHeapBridge` is forbidden and remains unverified until the
checker accepts linear borrowed or state-threaded owners.

```text
Semantic-Outcome: BLOCKED before executable validation; implementation review is clean, but the source-fresh seed segfaults while emitting the newly activated full ir::serialize module closure.
Concept-Status-Before: absent
Concept-Status-After: hypothesis with reviewed prototype implementation
Distinctions-Added: virtual reservation != touched physical memory; storage ownership != IR semantic identity; per-owner exactly-once != alias-safe linear ownership
Distinctions-Preserved: DefId provenance; SOIR v5 identity; SOIR v4 unknown-provenance compatibility; fail-closed capacity; unsupported writer opcode != invented wire tag
Distinctions-Erased: none
Evidence-Run: exact base 00d3e5ac0 source-fresh build rc=0; integration build rc=139 with out_status=missing; writer-guard corrective build rc=139 with out_status=missing and no serializer exhaustiveness diagnostic
Fallback-Path: none
Legacy-Kept: canonical inline IrModule and all existing by-value paths
Conflicting-Lanes: none observed by semantic status scanner before edit
Next-Semantic-Interface: split a minimal SOIR core that the seed can emit, then rerun the source-fresh raw-ELF gate; issue #877 repairs linear owner state-threading; issue #878 separately owns unknown reader-tag fail-closed policy
```

```text
Blocker-ID: BLK-20260714-IR-HEAP-SERIALIZE-CLOSURE
Status: classified
Severity: B1
Class: bootstrap-runtime
Owner: codex/ir-arena-storage-20260714
Lane: IR-HEAP-BRIDGE-V0
Worktree: /tmp/sounio-ir-arena-storage-20260714
Branch: codex/ir-arena-storage-20260714
Files-Owned: self-hosted/ir/heap_storage.sio, self-hosted/ir/serialize.sio, self-hosted/test_ir.sio, self-hosted/compiler/main.sio, scripts/ci/ir_module_heap_bridge_gate.sh, scripts/bootstrap/bootstrap_concat.sh, docs/internal/concepts/ir-storage-ownership.md, docs/internal/concepts/registry.tsv, docs/internal/concepts/bindings.tsv, docs/governance/topic-registry.v1.json, docs/governance/DOCS_AUTHORITY_MATRIX.md, docs/governance/DOCS_ACCEPTANCE_REPORT.md
Files-Read-Only: self-hosted/compiler/module_frontend.sio, self-hosted/native/codegen_x86_linux.sio
Do-Not-Touch: canonical IrModule/IrFunction/IrInstr layout and existing by-value paths
Repro: ulimit -s 65536; /usr/bin/time -v bash scripts/ci/build_modular_madaros.sh artifacts/ir-heap-bridge/madaros-source-fresh
Observed: exact base build succeeds; adding the explicit heap-storage/serializer closure reaches serializer lowering then the seed exits 139 with no output artifact, including after the writer exhaustiveness guard removes its only serializer checker diagnostic
Expected: source-fresh Madaros ELF exists and executes --ir-heap-bridge-self-test with the exact PASS receipt
Acceptance-Gate: MADAROS_RAW_BIN=<absolute-source-fresh-elf> IR_MODULE_HEAP_BRIDGE_EXPECT_SHA256=<sha256> bash scripts/ci/ir_module_heap_bridge_gate.sh
Evidence-Level: E2
Evidence: https://github.com/Sounio-lang/sounio/issues/879 records the exact control and two integration build receipts; control ELF SHA e586cfd19b4fe5aa68512c962a48fe88793e59aea7f1a2b58f33505042c317a7
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: extract the small SOIR v5/v4 module encode/decode core from the 4,700-line serializer closure without changing tags or identity, then repeat the same source-fresh build and raw gate
```
