# #1678 — the seed fix, handed to the Madaros fixed-point lane

One opcode in `self-hosted/compiler/lean_single.sio`. Everything here is
measured; nothing is predicted.

## The defect

`lean_single` stores arrays of aggregates **pointer-per-slot**, not inline — its
own store-path comments say so. The two access paths disagreed:

| path | emits | meaning |
|---|---|---|
| value | `mov rax, [rcx+rdx*8]` | **loads** the element pointer — correct |
| borrow (`type_is_array_ref`) | `lea rax, [rcx+rdx*8]` | address of the pointer **cell** — wrong |

The callee then reads the pointer bits as the first field, which is why the
symptom is a plausible-looking wrong number rather than a crash.

This is the unfinished half of **#740**, which corrected the *type* of
`&!module.functions[i]` — ref rather than value — and left the address
arithmetic alone.

## Bootstrap evidence

Measured on `origin/main`'s own `lean_single.sio` (an earlier attempt used a
copy 715 lines behind, which is why its numbers could not be compared to the
shipped binary at all):

| chain | md5 |
|---|---|
| patched | `gen2 == gen3 == 25fb229c2bf68e94b17ba5d9bc79174f` |
| unpatched control | `gen2 == 3a7a17a0f62e97e771c96c029ccd18b3` |
| **shipped `bin/souc-lean-single-x86_64`** | **`3a7a17a0f62e97e771c96c029ccd18b3`** |

The unpatched control **reproduces the shipped seed bit-exactly**. That is what
makes a refresh an auditable operation rather than an act of faith.

The seed derived from *this branch* is `397b88a38d12402f27afd30ec8b075de`, byte
for byte the one every measurement below was taken with — the explanatory
comment added here does not perturb the binary.

## What it fixes

Four reproducers, all in this directory, run under `lean_single`:

| file | stock seed | patched seed | expected |
|---|---|---|---|
| `repro_aggregate_element_ref.sio` | `hoisted=5 inplace=8154823786712` | `hoisted=5 inplace=5` | `5 5` |
| `repro_container_kinds.sio` | all garbage | `5 5 77 5` | `5 5 77 5` |
| `repro_elem_vs_field.sio` | both garbage | `77 5 5 77` | `77 5 5 77` |
| `repro_mut_borrow_element.sio` | `B_inplace tag=0 bits0=0` | `B_inplace tag=201 bits0=15` | `201 15` |

The last one is the severe case. A **mutable** borrow does not merely write to
the wrong address — it **zeroes the element it was supposed to modify**. Every
other witness only reads.

## Regression evidence

- `scripts/dev/run_sio_test_suite.sh`, **2948 tests**, run twice from pristine
  `origin/main` — stock-seed Madaros vs patched-seed Madaros. The two compilers
  differ in md5, so the patch did change generated code. The two suite outputs
  do not: **`diff` is 0 lines, same md5 `9443281c…`**.
  `Pass 377 / Fail 1213 / Known 143 / Skip 1215`, identical both ways.
  Honest caveat: that harness discriminates poorly against a raw Madaros ELF, so
  this is evidence of no regression, not proof of safety.
- Madaros builds at the 6-error `origin/main` baseline.

## The demonstration that closes the argument

`madaros --self-test` on **pristine** `origin/main` sources — the four source
hoists in PR #1681 **reverted**, `git status` showing one modified file:

```
T19 OK: sprof promotion
T24 OK: layout sort hot first
```

Both were red before: T19 failing, T24 the point where the suite **segfaulted**.
So the seed fix subsumes all four hoists in #1681, and also un-sticks the
self-test suite (#1680) — where I had filed the T24 crash as address-space
exhaustion. That hypothesis was wrong; T24 died on `ir/layout.sio:183`
dereferencing the wrong address.

## Not established

- Only the `type_is_array_ref` borrow path was examined. Whether other
  `lean_single` codegen paths share the pointer-per-slot confusion is unmeasured.
- `scripts/ci/lean_single_fixed_point_gate.sh` compares `stage1 == stage2` while
  the Makefile compares `gen2 == gen3`. This fix passes either way, but the gate
  is wrong in general and will reject the next legitimate codegen change.

## The refreshed seed ELF is IN this branch, and it passes the canonical gate

`bin/souc-lean-single-x86_64` is replaced here with the fixed point of the
patched source, **`25fb229c2bf68e94b17ba5d9bc79174f`** (the shipped one is
`3a7a17a0f62e97e771c96c029ccd18b3`).

This is not optional. `scripts/ci/canonical_compiler_gate.sh` — the CI step
"Canonical lean_single fixed point" — **fails** when the committed ELF is not
the byte-identical self-reproducing fixed point of `lean_single.sio`. Patching
the source without refreshing the ELF would land a red gate.

Run on this branch:

```
[canonical-compiler] bin/souc md5     = 25fb229c2bf68e94b17ba5d9bc79174f
[canonical-compiler] self-compile md5 = 25fb229c2bf68e94b17ba5d9bc79174f
[canonical-compiler] PASS: bin/souc IS the canonical self-reproducing fixed point
```

That it would fail otherwise is arithmetic on measured values, not an assertion:
the shipped seed is `3a7a17a0…`, and that seed compiling this branch's source
produces `397b88a3…`. The gate compares those two for equality.

**Note that this gate was NOT RUNNING when the handoff was first delivered.** The
`Contracts` job died at an earlier step from 2026-08-06 09:33Z until PR #1684,
taking 30 later steps with it — this one included. So the first version of this
handoff was verified against a gate that was switched off.

## Ownership

This branch is a **handoff, not a claim**. The Madaros fixed-point lane declared
ownership in `0e6a294ac8`; the seed refresh belongs there. Cherry-pick, rewrite
or discard as suits that lane.

Refs #1678, #1680, #1681, #740
