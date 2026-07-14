<!-- docs:meta
topic_id: repo.docs.internal.compiler.raw-field-terminal-contract
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.compiler.raw-field-terminal-contract
-->

# Raw Field Terminal Contract

Status: implementation checkpoint
Concept-ID: `SOUNIO-IR-STORAGE-OWNERSHIP`

This checkpoint is the final extension of legacy field-address lowering. After
it is proven, the legacy compiler remains a differential oracle; new address,
ownership, or place semantics belong in the successor IR rather than additional
`label_id` conventions.

## Semantic Boundary

The supported raw projection is exactly:

```text
(*identifier_raw_pointer_to_named_struct).field
```

Reads accept `*const T` and `*mut T`. Writes accept only `*mut T`. The lowerer
must know the pointer kind, the named pointee, the exact named field, and the
compiler-owned physical word offset. Unknown pointees, missing fields, dynamic
array extents, layout overflow, and non-identifier pointer expressions fail
closed. A typed non-identifier raw pointer expression fails with `E230`:
`raw field projection requires an identifier pointer operand`. The terminal
legacy projection accepts only fields marked
`raw_projectable` whose declared storage width is exactly one word. Scalars,
pointers, references, and named-aggregate handle slots are projectable. Inline
arrays are never projectable, including a one-element array. Fixed arrays still
determine the offsets of later fields, but projecting the array field itself
requires an aggregate/place result and therefore fails closed with `E229`:
`raw inline aggregate field projection requires Place IR`. There is no
field-name hash or cross-struct fallback.

`IrFieldGet` and `IrFieldSet` use an opcode-local address mode:

```text
0 managed handle
1 typed reference to a managed handle slot
2 raw address
```

Raw mode carries a physical word offset in `field_idx`. Managed and reference
modes retain their ordinal field index. SOIR already serializes both fields, so
this checkpoint does not change the SOIR version or wire width.

Because `label_id` is a shared physical slot, CFG transforms must call
`ir_opcode_has_cfg_label` before scanning or renumbering it. Inlining,
instrumented cloning, and normalization preserve field modes and every other
opcode-local `label_id` payload verbatim.

The modular native backend implements raw mode as `[base + offset_words * 8]`.
It must not resolve a handle, add a managed-object header, or first dereference
the pointee word. Machine IR preserves the same mode in its field-op-specific
payload.

## Physical Layout Authority

The lowerer computes each declared field's final `offset_words`,
`storage_words`, and `raw_projectable` category once while registering the named
struct. Fixed arrays contribute their literal element count recursively; every
non-array value contributes one managed/native word.
This matches the existing modular aggregate representation: fixed arrays are
tables of slots, while scalars, references, pointers, and named aggregates are
one slot. Storing the final offset prevents later consumers from reconstructing
or reinterpreting the layout.

## Proof Surface

The positive witness must distinguish same-named fields in different structs,
place a fixed `[f64; 2]` before a scalar field, place a named aggregate handle
before another field, cover raw const reads and raw mutable writes across a
call, and preserve `f64` classification. Existing managed-field and typed-ref
field witnesses remain green.

The negative witnesses must reject a field write through `*const T`, inline
aggregate projections, and raw reads or writes whose dereference operand is not
an identifier. The static gate rejects modes outside `0..2`, any new field-mode
label convention, CFG passes that renumber opcode-local payloads, raw lowering
that emits `OpDeref`, global/hash fallback, handle resolution, or a
managed-object header on the raw backend path.

## Claims Forbidden

This checkpoint does not claim general pointer arithmetic, C ABI struct layout,
packed layout, dynamic-array field layout, arbitrary raw pointer expressions,
raw indexing, provenance recovery, bounds safety, null safety, or canonical
arena ownership. It does not change heap storage, the SOIR writer, or the
default serializer.
