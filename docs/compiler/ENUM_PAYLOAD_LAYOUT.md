<!-- docs:meta
topic_id: repo.docs.compiler.enum-payload-layout
authority: repo_only
audience: contributors
last_validated: 2026-10-30
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.enum-payload-layout
-->

# Enum payload layout in Madaros (KL-20)

**Status**: implemented on `fix/kl20-20261030` (native Madaros lowering, x86-64)
**Date**: 2026-10-30
**Scope**: `self-hosted/ir/lower.sio`, `self-hosted/check/check.sio`, `self-hosted/check/defs.sio`
**Closes**: KL-20 for non-generic user enums ([KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md#kl-20--enum-payload-variants-p13))

## Context

Madaros represented every enum value as a bare discriminant. A variant with a
payload (`Circle(f64)`, `Rect { w: i64, h: i64 }`) had nowhere to keep it, so
since `974c29864` the checker refused every construction and payload pattern
with E264. The docs left the layout open. This note records the choices made
where real alternatives exist.

## Decision

### 1. Two representations, chosen per enum

* An enum **none** of whose variants declares a payload keeps the bare
  discriminant (`i64`, variants numbered from 0 in declaration order). Nothing
  about these enums changes: same values, same ABI, same `==`, same
  `E::V as i64`.
* An enum with **at least one** payload variant is a *payload enum*. Every value
  of it, unit variants included, is a pointer to a **tagged cell**.

Alternative rejected: a mixed representation (unit variants as small integers,
payload variants as pointers, told apart at run time by magnitude). It saves an
allocation per unit value but makes every tag test a two-way branch and relies
on heap addresses never being small integers. The uniform cell is one load and
one compare.

### 2. Cell layout

```
word 0       discriminant (declaration index of the variant)
word 1..n    the variant's payload fields, in declaration order
size         8 * (1 + maxf) bytes, maxf = largest field count over the enum's variants
```

Every payload field occupies one 8-byte word, which is how the lowerer already
carries every value: an integer or `bool`, the bits of an `f64`/`f32`, or the
handle of a `string`, struct, array, `Box<T>`, `Option<T>` or another enum.
Alignment is therefore always 8, and no field needs padding. All cells of one
enum have the same size (discriminant + the largest payload), so a future
in-place reassignment of a payload-enum place cannot overflow its cell.

A struct variant's fields keep their declaration order (`Rect { w, h }` puts `w`
in word 1, `h` in word 2) whatever order a literal or pattern names them in. A
tuple variant's fields are the parser's positional `_0`, `_1`, ...

Field types that are not one word are refused (E-level lowering refusal, never
truncated): tuples, `f128`/`f256`, and `Knowledge<T>` (which carries variance
side state).

### 3. Value semantics

A payload enum value is immutable after construction; nothing in the language
assigns into a payload. Copying the value copies the pointer, which is
observationally a copy. An array payload is stored as its handle, exactly as
`Some(arr)` stores it (`lower_option_some_cell`); binding it to a `var` copies
it, as for any array.

### 4. No niche optimization

`Option<PayloadEnum>` is an Option cell holding the enum's cell pointer
(two allocations). Packing `None` into a null enum pointer would save one
allocation but needs the Option lowering to know the payload's representation
at every site; the existing Option lowering is already correct and is reused
unchanged. Revisit only with a measured cost.

### 5. Equality and exhaustiveness

* `==`/`!=` on a payload enum is refused (**E279**): on cells it would compare
  addresses, and the enum-vs-integer comparison accepted for bare enums would
  compare an address with a discriminant. Structural equality is a separate
  feature.
* A `match` over a payload enum must cover every variant (**E280**), because a
  value no arm matches yields the `0` the match's end label loads, which for a
  payload enum is a null cell handed on as a value. The check is conservative:
  a variant counts as covered by an unguarded arm whose sub-patterns are all
  irrefutable (bindings, `_`, tuples of those), by a binding/`_` arm, or by an
  or-pattern alternative that does. A variant split across arms by refutable
  sub-patterns needs a final `_`. Payload-free enums keep their existing
  (unchecked) behaviour; changing that is a separate decision because existing
  code relies on it.
* Constructor arity is **E277**, pattern arity **E278**; a constructor argument
  of the wrong type is E009, an unknown struct-variant field E012.

### 6. Generic enums: not in this change

`enum Opt<T> { S(T), N }` does not parse in Madaros (the enum header has no
type-parameter list), so there is no generic user enum to lay out yet. The
layout above is independent of the payload types, so a type-erased generic enum
would use the same cell. What it additionally needs is (a) the parser and
`EnumDef` to keep type parameters, (b) checker instantiation (`Opt<i64>`) and
inference at construction (`Opt::S(5)`, and `Opt::N` with no payload to infer
from), and (c) the lowerer to know, at each payload binding, whether a `T`
field holds an `f64` (the lowerer tracks float-ness per register, not per
type). Builtin `Option<T>` is unaffected and composes with payload enums in
both directions (`Some(Shape::Circle(r))`, `V(Option<i64>)`).

## Consequences

* `Shape::Circle(2.0)`, `Shape::Rect { w: 3, h: 4 }`, `Shape::Empty` on a payload
  enum each allocate one cell; passing and returning them moves one word.
* Patterns bind payloads with the metadata a parameter of the declared field
  type gets (float/integer kind, struct type, `Box` inner layout, f64-array
  elements, fixed array length, Option payload kind), so `p.x` on a struct
  payload and `*b` on a `Box<Self>` payload lower as they do for parameters.
* A scrutinee reached through `&E` (a `p: &Proposition` parameter) reads the
  tag and payload words through the reference, like struct patterns do.
