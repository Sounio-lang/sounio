<!-- docs:meta
topic_id: repo.docs.audit.madaros-box-autoderef-backend-information-refutation-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-box-autoderef-backend-information-refutation-2026-08-17
-->

# Madaros Box Auto-Deref Backend Information Refutation

Date: 2026-08-17

Status: **sixth cause refuted**. Runtime Box-descriptor inspection cannot repair
the auto-deref miscompile after the existing IR-to-Machine-IR boundary.

## Hypothesis

The native backend could inspect the allocation descriptor, recognize a Box,
load its payload, and then perform the source field read without changing
expression lowering.

## Instrument

- Source base: `9079afbac119214795b2e3508fc90d962b355b63`
- Source-built Madaros SHA-256:
  `2dcaf1594465e67f07858f4e178409b8c642fec2e02a000f095e8f7cf505745e`
- Source-derived `lean_single` instrument compiler SHA-256:
  `87ebb6ecb4dc03b173cb5ae83e0633a0acff2e8624740917113d554ae83f3105`
- Boundary witness:
  `tests/native-v2/box_backend_information_boundary_witness.sio`
- Prior end-to-end witness: `tests/run-pass/box_all_read_forms.sio` in PR
  #1814, built from source at `c5754c0c84`

The boundary witness constructs an `IrFieldGet` with all information that could
plausibly identify the operation: destination 9, base 4, field index 52, field
name `tag`, and `label_id=1`. It then calls `native_v2_pseudo_from_ir` directly
and prints the resulting `MachineInstr`.

The source-derived compiler built the witness successfully. The witness ELF
has SHA-256
`f34c037d43daa2ef011f9604dc38097161cf0e8e2e15fb509686f56441ee38b2`
and executed with `rc=0`.

## Measured Boundary

`native_v2_pseudo_from_ir` preserves only:

```text
opcode=13 dst=9 base=4 idx=52 name_len=0 cond=0 arg_index=-1
BOX_BACKEND_INFORMATION_ERASED
```

The resulting `MachineInstr.name` is empty. Its remaining generic metadata
slots retain their defaults (`cond=0`, `arg_index=-1`), so `label_id=1` is also
absent. Legalization then emits one `MIR_OP_FIELD_LOAD(base, 52)`.

The direct native path has more IR metadata than Machine IR, but it still sees
the already-wrong index. The prior source-built trace measured:

| Source | IR entering native codegen |
|---|---|
| `b.tag` | one `field_get(base=b, field_idx=52, label_id=0)` |
| `(*b).tag` | `field_get(base=b, field_idx=0)`, then `field_get(field_idx=2)` |

## Why The Runtime Descriptor Is Insufficient

The emitted descriptor table distinguishes Box (`descriptor_id=8`) from a
generic struct (`descriptor_id=0`). It does not encode the boxed type `Ex`, its
field names, or the mapping `tag -> 2`. Resolving a handle yields only the
object base.

Even a new runtime test for `descriptor_id=8` would be ambiguous. The explicit
Box payload operation is itself `field_get(base, 0)` and must load exactly one
payload. An unconditional backend auto-deref would apply to both the missing
auto-deref and the intentional explicit dereference. Machine IR carries no bit
that distinguishes them, and it carries no inner-layout identity with which to
repair index 52.

## Verdict

**REFUTED:** "The backend can recover Box auto-deref from the runtime Box
descriptor without changing lowering or the IR contract."

This is an information-erasure boundary, not an omitted backend branch. A
backend repair would first require a new explicit operand contract carrying at
least operation kind plus inner-layout identity. Reconstructing that contract
from generic runtime descriptors is impossible in the current representation.
The smallest repair remains upstream: retain `T` at binding time and emit the
missing `field_get(base, 0)` before the field index resolved against `T`.

## Evidence Boundary

The source-built compiler at the stated base was produced successfully through
`scripts/ci/build_modular_madaros.sh`, which serializes its compiler invocations
through `scripts/dev/souc-build-lock.sh`. The source-derived instrument compiler
and the boundary witness were also built through that lock.
Its build emitted existing diagnostics, including the
`self-hosted/resolve/imports.sio:384` exhaustive-match diagnostic, but still
produced the ELF. This note does not claim that the full repository gate is
green, nor does it modify `self-hosted/ir/lower.sio`.
