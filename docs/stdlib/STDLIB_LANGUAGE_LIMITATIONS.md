<!-- docs:meta
topic_id: repo.docs.stdlib.stdlib-language-limitations
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.stdlib.stdlib-language-limitations
-->

# Stdlib Module Language Limitations

This page describes language constructs that stdlib modules cannot use today,
and what to write instead. Everything below was measured against the committed
compiler (`bin/madaros-linux-x86_64`, Madaros v0.80.0) with
`SOUNIO_STDLIB_PATH` pinned to this tree's `stdlib/`. Reproduce with:

```bash
bash scripts/dev/language_limitation_sweep.sh
```

The sweep writes one row per file to `artifacts/audit/`. The figures quoted
here come from `artifacts/audit/language_limitation_sweep_20260903.tsv`.

## Scale

Over `stdlib/`, `examples/` and `tests/run-pass/` — 4539 files:

| tree | files | accepted by `souc check` |
|---|---|---|
| `tests/run-pass` | 1912 | 1760 (92%) |
| `stdlib` | 1611 | 1176 (73%) |
| `examples` | 1016 | 621 (61%) |

**Most rejections are not language limitations.** The two largest error classes
in the tree are wiring and dead code, not constructs the compiler refuses to
support:

- **E137 (`use of undeclared variable`)** — 323 files. 191 of them contain no
  `use` statement at all while calling functions from other modules (2947 of
  3623 occurrences). Of the 676 occurrences in files that *do* import
  something, 529 name a function that is defined nowhere in the tree and 147
  name one that exists but is not imported. None name a function that exists
  **and** is imported, which is the only shape that would implicate name
  resolution.
- **E035 (`effect not declared in function signature`)** — was 199 files. The
  dominant cause was a single over-declared effect row on a stdlib
  constructor, since corrected; the current figure is far lower. Effects are
  reported faithfully — the errors were real, the annotation they propagated
  was not.

Read a large error count as a claim about that file, not about the language.

## Actual language limitations

### Slices are fixed-length; `Seq<T>` is the growable one

**Status**: by design.

`[T]` supports `.len()` and nothing else. `.push()`, `.pop()`, `.insert()` and
`.remove()` on a slice receiver are rejected with **E019** (`method calls are
not supported for this type`).

```sio
// Rejected — E019
var xs: [i64] = []
xs.push(1)

// Write this instead
var xs: Seq<i64> = seq_new()
xs.push(1)
let n = xs.len().unwrap("len")
```

`Seq<T>` carries `.push`, `.get`, `.set` and `.len`, and nests: `Seq<Seq<i64>>`
checks clean. See `tests/run-pass/seq_methods.sio`.

Note the third idiom in the tree: `stdlib/collections/vec.sio` offers `IntVec`
and `FloatVec`, fixed-capacity 256-element structs with their own `push`. They
work because struct methods work, not because slices grew one.

**Scope**: 46 stdlib files use `.push(` across 349 call sites; 40 of them are
rejected. The six that pass do not push onto a slice — they push onto their own
struct type or onto a `Seq`. `stdlib/stats/validation.sio` is the clearest
precedent: its header records that it was written against fixed `&[f64; 256]`
buffers with an explicit length argument precisely because the imported
multi-module path rejects `[f64].push()` with E019.

### Character literals break outside a simple binding

**Status**: open defect, frontend.

A character literal is accepted as the whole right-hand side of a binding, and
as the operand of an unparenthesised `as`. Anywhere a bracket encloses it, the
parser fails:

```sio
let c = '0'              // OK
let d = '0' as i64       // OK

let e = ('0')            // parse error
let f = g('0')           // parse error
let h = ['0']            // parse error
```

The token reported after the literal varies with the character, so the failure
is in tokenising or advancing past the literal rather than in the grammar for
any one bracket form.

Separately, `char` does not combine with `i64`:

```sio
if c >= '0' { }   // E004: expected i64, found char
```

`as` is the documented conversion, but only outside brackets, so the practical
form is a separate binding:

```sio
let zero = '0' as i64
if c >= zero { }
```

Most of the stdlib avoids the construct entirely and compares ASCII codes
returned by `str_char_at`, as in `stdlib/darwin_pbpk/io/observed_csv.sio` and
`stdlib/genomics/io/fasta.sio`. That is the idiom to follow.

### Type arguments on method calls are accepted and ignored

**Status**: open.

`x.m::<T>()` parses, and the type arguments are then discarded — the checker
does not read them and code generation has no notion of generics. Nothing
warns. Do not write it; it means nothing.

There is no `parse` method on `string` and no `FromStr` trait. Use the
type-specific free functions: `str_to_i64` and `str_to_f64` in
`stdlib/str/lib.sio`, or `parse_i64` / `parse_f64` in `stdlib/data/csv_loader.sio`.

### One instantiation per generic template

**Status**: bounded.

Turbofish at a call site works — `f::<T>()`, `f::<T, U>()`, `f::<A, B, C>()` —
but a template may be instantiated with only **one** type-argument list per
compilation unit. Distinct scalar arguments still behave (`pick::<i64>` and
`pick::<f64>` together return the right values); with any non-scalar argument
the compilation is refused rather than silently mis-specialised. Nested type
arguments are fine on their own: depth was never the constraint.

See `docs/compiler/KNOWN_LIMITATIONS.md` for the compiler-side detail and the
undocumented caps (4 type parameters, 256 generic functions, 256 generic
structs).

## The three modules this page used to be about

This page previously stated that three modules "cannot yet be implemented"
because of generic type parameters in call position, method calls with type
parameters, and deeply nested generic return types. **All three claims were
wrong.** Turbofish works, nesting depth was never the issue, and none of the
three modules contains the constructs it blamed — they use `[[usize]]`, not
`Vec<Vec<Vec<usize>>>`, and they parse integers with hand-written
`parse_i64` / `parse_f64` rather than `.parse::<T>()`. The modules were
rewritten into real Sounio syntax in early 2026 and this page was never
updated.

Measured today:

| module | verdict | why |
|---|---|---|
| `stdlib/stats/effect_sizes.sio` | rejected | 2× E019 — `.push()` on a slice |
| `stdlib/graph/nulls/configuration.sio` | rejected | 22× E019 + 5× E009 |
| `stdlib/data/csv_loader.sio` | rejected | 16 parse errors — character literals |

None of that is about generics.

The wiring matters as much as the compilation:

- `stdlib/stats/lib.sio` does not export `effect_sizes`, and nothing imports
  it. It is also largely superseded by `stdlib/stats/effect_size.sio` and
  `stdlib/stats/effect_convert.sio`, which are self-testing `//@ run-pass`
  modules. Cliff's delta is the part with no successor.
- `stdlib/graph/lib.sio` does not reach `nulls`, so
  `graph::nulls::configuration` is unreachable through the `graph` entry point.
  Its configuration-model code has no successor anywhere in the tree.
- `stdlib/data/csv_loader.sio` is the only one with a live test
  (`tests/stdlib/data/test_csv_e2e.sio`) and keeps a distinct role — graph edge
  lists — beside `stdlib/data/csv.sio`, `stdlib/data/csv_reader.sio` and
  `stdlib/csv/`.

None of the three is inside the `stdlib.surface` support contract checked by
`scripts/ci/sounio_stdlib_surface_support_gate.sh`.

## See Also

- [KNOWN_LIMITATIONS.md](../compiler/KNOWN_LIMITATIONS.md) — compiler-side detail
- [MINIMUM_VIABLE_SOUNIO.md](../guide/MINIMUM_VIABLE_SOUNIO.md) — current language capabilities
- `scripts/dev/language_limitation_sweep.sh` — the measurement behind this page
