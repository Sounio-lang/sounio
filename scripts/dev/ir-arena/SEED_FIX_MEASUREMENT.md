# The seed fix for #1678, measured on `origin/main`

One opcode. It subsumes all four source workarounds, and it un-sticks the
self-test suite past the point where it had been crashing.

## The defect

Arrays of aggregates in `lean_single` are stored **pointer-per-slot**, not
inline — the compiler's own store-path comments say so. The two paths disagreed:

| path | emits | meaning |
|---|---|---|
| value | `mov rax, [rcx+rdx*8]` | **loads** the element pointer — correct |
| borrow (`type_is_array_ref`) | `lea rax, [rcx+rdx*8]` | address of the pointer **cell** — wrong |

`0x8d` where it should be `0x8b`. The callee then reads the pointer bits as the
first field, which is precisely the plausible-looking wrong number this whole
investigation chased.

`origin/main` `self-hosted/compiler/lean_single.sio:16045`. Patch in
`1678_seed_fix.patch`; `elem_ty` and `elem_hash` are already in scope four lines
above, so the guard is local.

This is the unfinished half of **#740**, which fixed the *type* of
`&!module.functions[i]` — ref rather than value — and left the address
arithmetic alone.

## Every witness, before and after

Seed derived from **pristine `origin/main`** with only that patch.

| witness | stock seed | patched seed | expected |
|---|---|---|---|
| `repro_boxed_element_ref` | `hoisted=5 inplace=8154823786712` | `hoisted=5 inplace=5` | 5 5 |
| `pkgshape` (local / ref-param / bound-`let`) | all garbage | `5 5 77 5` | `5 5 77 5` |
| `subshape` (`&elem` and `&elem.field`) | both garbage | `77 5 5 77` | `77 5 5 77` |

One opcode covers local containers, reference parameters, `Box`, `&elem`,
`&elem.field`, and the bound-`let` variant.

## The measurement that closes the argument

The four source hoists were **reverted**. `git status` showed exactly one
modified file, `lean_single.sio`. Madaros was then built from those pristine
sources with the patched seed.

```
T19 OK: sprof promotion
```

`T19` is the pre-existing self-test that asserts profile-guided promotion
reaches `AGGRESSIVE`. It reads OK with **no source change anywhere** — so the
compiler fix provably subsumes `ir/profile.sio`, `ir/layout.sio` ×2 and
`compiler/main.sio`'s receipt.

## It also moved the self-test crash — and refutes what I published in #1680

Deterministic, 3 runs of 3:

| | OK | FAIL | dies after |
|---|---|---|---|
| `origin/main`, stock seed | 20 | 7 | T23 |
| `origin/main`, **patched seed** | **22** | **6** | T24 |

`T24 layout sort hot first` — *the test that was segfaulting* — now passes, and
the crash moved on to T25.

**So the leading hypothesis I filed in #1680 is wrong.** I attributed the T24
crash to address-space exhaustion: `layout_sort_by_profile` takes `IrModule` by
value and returns a `LayoutResult` that embeds another, ~2.08 GB each, with a
nested `layout_reorder_functions` call on top. That reasoning was tidy and it
was not the cause. T24 crashed because `layout_sort_by_profile:183` takes
`&module.functions[si].name` — the wrong address — and dereferencing it walked
off. Same bug as T19, different symptom.

Worth keeping as method: the size argument explained the T23/T24 contrast well
enough to feel settled, and the thing that killed it was changing one opcode and
looking, not thinking harder.

## Regression check

Same two programs, same pristine sources, both compilers:

| | stock seed | patched seed |
|---|---|---|
| `flt.sio -O` | `v=9223372036854775808.000000` | identical |
| `min1.sio -O` | `rc=182` | identical |
| `min1.sio` no `-O` | `hoisted=5 inplace=5` | identical |

Both `-O` results are pre-existing `origin/main` defects — the first is #1669,
which this branch fixes separately — and neither is touched by the seed patch.
Madaros built at the 6-error baseline.

## The three open items, now closed

### 1. Broad suite — bit-identical

`scripts/dev/run_sio_test_suite.sh`, **2948 tests**, run twice from pristine
`origin/main` sources: once with a Madaros built by the stock seed, once by the
patched seed. Both compilers built at the 6-error baseline and differ in md5, so
the patch did change generated code.

```
Pass 377   Fail 1213   Known failures 143   Skip 1215   Total 2948
```

**`diff` of the two full outputs: 0 lines. Same md5, `9443281c…`.** Not one test
moves in either direction.

Stated honestly: with 1213 failing and 1215 skipped, this harness has limited
discriminating power when pointed at a raw Madaros ELF — it is normally driven
against `bin/souc`. The real signal is the 377 that pass, and none of them
changed. It is evidence of no regression, not proof of safety.

### 2. Bootstrap fixed point — on `origin/main`'s own source this time

| chain | result |
|---|---|
| patched | `gen2 == gen3 == 25fb229c2bf68e94b17ba5d9bc79174f` |
| unpatched control | `gen2 == 3a7a17a0f62e97e771c96c029ccd18b3` |
| **shipped `bin/souc-lean-single-x86_64`** | **`3a7a17a0f62e97e771c96c029ccd18b3`** |

The unpatched control **reproduces the shipped seed bit-exactly**, so a refresh
is an auditable operation rather than an act of faith. The earlier attempt could
not show this because it patched a copy of `lean_single.sio` that was 715 lines
behind `origin/main`.

### 3. `&!` mutable borrows — worse than reading

`repro_mut_borrow_element.sio`. The element holds `tag=101 bits0=8`; a mutable
borrow of it, in place, through a reference parameter, then reads back:

```
stock seed     A_hoisted tag=101 bits0=8 | B_inplace tag=0   bits0=0
patched seed   A_hoisted tag=101 bits0=8 | B_inplace tag=201 bits0=15
```

The write did not merely land at the wrong address — **it zeroed the element**.
Every earlier witness only read, so until now the demonstrated harm was wrong
numbers. This is silent destruction of state, and it raises the severity of
#1678.

## Still not established

- The three `ir/egraph.sio` sites remain unaudited beyond "private, zero
  external callers". They are moot if the seed fix lands.
- No measurement of whether other `lean_single` codegen paths share the
  pointer-per-slot confusion; only the `type_is_array_ref` borrow path was
  examined.

Refs #1678, #1680, #740, #1669
