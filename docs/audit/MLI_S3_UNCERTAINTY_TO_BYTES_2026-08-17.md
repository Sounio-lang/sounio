<!-- docs:meta
topic_id: repo.docs.audit.mli-s3-uncertainty-to-bytes-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: cursor-2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.mli-s3-uncertainty-to-bytes-2026-08-17
-->

# MLI S3 — Uncertainty in the instruction stream

**Date:** 2026-08-17
**Lane:** `cursor-2` / `mli-s3b-knowledge-to-hardware`
**Status:** research result. Both answers were publishable; this one is the first.
**Gates:** `self-hosted/mli/s3_gate_runner.sio`, `self-hosted/mli/s3b_gate_runner.sio`

## 1. The question

Every production compiler erases uncertainty before machine code. LLVM, GCC
and Cranelift lower a value and drop its variance, because no backend has a
notion of an operand that carries its own error. MLI was designed so
`Knowledge<T>` stays a KIND all the way down. S3 is where that either
becomes the first backend that lowers uncertainty as a first-class operand,
or is revealed as ornament.

The same question applies to Cayley–Dickson multiplication. Octonion `cd_mul`
as a compiler primitive with associator-aware lowering is unclaimed ground,
and the associator is a positive graded invariant in this project's own
theory (`stdlib/algebra/octonion.sio`, `stdlib/epistemic/uncertain_octonion.sio`).

## 2. What S3 now does

The sanctioned route is still expand-then-legalize (Option C, design §2.3 /
§8.2). That is not a library above the backend: `expand_k` / `expand_cd`
are compiler passes that rewrite first-class kinds into scalar MLI, and
`legalize_x86` (S3b) emits those scalars — including the branch diamonds
the confidence lane requires — as measured x86-64 bytes.

An unexpanded `Knowledge` or `CD` operand that reaches `legalize_x86` is
still refused with `MLI_L_KNOWLEDGE_UNEXPANDED`, checked before any shape
check. Stripping the variance here would be one `if` cheaper. That `if`
is the difference between a kind and decoration.

## 3. The named temptation — Knowledge

### 3.1 `k_add` — the second `addsd`

After `expand_k` rewrites

```
kc = k_add(ka, kb)
ret k_extract_var(kc)
```

the first float instruction in the stream is

```
addsd xmm0, xmm1    ; f2 0f 58 c1     val = a.val + b.val
```

**That is the instruction at which erasure becomes tempting.** A production
backend would `ret` the first `addsd`. Keeping the variance is exactly one
more `addsd` of the variance slots, plus their load/store:

```
addsd xmm0, xmm1    ; f2 0f 58 c1     var = a.var + b.var
                    ; GUM: Var(X+Y) = Var(X)+Var(Y)   (JCGM 100:2008 §5.1.2)
```

Cost to keep, measured: 4 bytes of encoding + the two slot loads and one
store already in the mimicry shape. Gate X7 requires `addsd` count ≥ 2
even when the function returns the *value* lane — unused-lane DCE of the
variance `addsd` would be erasure, and is therefore a future-pass
forbidden transformation unless a dedicated "variance-dead" analysis
exists.

### 3.2 `k_mul` — the first `mulsd`

```
mulsd xmm0, xmm1    ; f2 0f 59 c1     val = a.val * b.val
```

**That is the product-form temptation.** GUM (delta method, stdlib
`ep_mul`, JCGM 100:2008 §5.1.2) requires four more `mulsd` and one
`addsd`:

```
Var(XY) ≈ Y²·Var(X) + X²·Var(Y)
```

The σx²σy² Gaussian-product term is **not** added. That omission is the
stdlib oracle (`knowledge.sio`), documented in `knightian.sio` as a
standard GUM convention — not a guess. Gate X13 requires `mulsd` count ≥ 5
and `addsd` count ≥ 1.

Confidence decay is `*99/100` for add and `*98/100` for mul, matching
`ep_add` / `ep_mul`. The conf lane still costs three branch diamonds
(9 extra blocks) because MLI R0 has no `select` and `Bool` does not
participate in integer arithmetic. That is the *second* temptation, named
in S3 and now paid: S3b encodes `test %rax,%rax; je rel32`.

`k_div` / `k_fma` remain refused. Division by an uncertain Y needs its
own documented derivation; it is not guessed here.

## 4. The named temptation — Cayley–Dickson

### 4.1 `cd_mul` — the first `mulsd` of `a0*b0`

A backend that treated the algebra as a real would emit `a0*b0` and
return. Keeping the product costs, measured against today's walls:

| dim | algebra | ops | exploded args | expand | legalize |
|---:|---|---:|---:|---|---|
| 2 | complex | 4 fmul + 1 fsub + 1 fadd | 4 | yes | yes (4 `mulsd` + 1 `addsd` + 1 `subsd`) |
| 4 | quaternion | 16 fmul + 12 fadd/fsub | 8 | yes (fits `MAX_INSTRS=32`) | **refuses** — measured GPR convention is 6 registers |
| 8 | octonion | 64 fmul + 56 fadd/fsub | 16 | **`MLI_XC_OCTONION_WALL`** | never reached |
| 16 | sedenion | larger | 32 | same wall | never reached |

The octonion overflow is the named wall. Raising `MAX_ARGS` / `MAX_INSTRS`
is the deferred module-arena decision, triggered now by a corpus (this
gate), not guessed in advance.

### 4.2 `cd_associator` — not an instruction below dim 8; a wall at 8

The S1 kind model already encodes the algebra grade: `cd_associator`
with `dim < 8` is `MLI_VERR_OPERAND_RULE`. An identically-zero
associator is not an instruction. That is associator-aware at the
verifier, not at emit.

For dim ≥ 8 the associator is a positive graded invariant. Emitting
zero — or emitting `a0*b0*c0` — would be a semantic miscompile of this
project's own theory. `expand_cd` refuses with
`MLI_XC_ASSOCIATOR_WALL` before exploding args, so the refusal is
named as the associator, not as a generic capacity error.

Computing the octonion associator is four `oct_mul` (two triple
products) plus 8 subtractions: well above both walls. The op is
legal MLI and cannot reach bytes today. That named inability is the
contribution.

## 5. What was *not* claimed

- Bit-identity of the Knowledge / CD sequences against the pinned
  native-v2 emitter. The golden `add1` bit-identity gate is unchanged
  (pin O5). The Knowledge / CD sequences have no pinned-emitter twin:
  the pinned emitter has never carried these kinds.
- Executed ELF parity of the GUM bytes against a host oracle. The
  interpreter (host = Sounio under Madaros, single semantic clock)
  matches `ep_add` / `ep_mul` / complex and quaternion products. Wrapping
  the legalized bytes in a runnable ELF is the next measured tranche,
  not this one.
- `k_div`, `k_fma`, CD dim=8 expansion, a by-ref CD ABI, or DCE of
  unused variance lanes.

## 6. Files

| path | role |
|---|---|
| `self-hosted/mli/legalize_x86.sio` | S3b: multi-arg, integer, `je rel32` / `jmp rel32` |
| `self-hosted/mli/expand_k.sio` | `k_add` + `k_mul` GUM, shared conf diamonds |
| `self-hosted/mli/expand_cd.sio` | complex / quaternion mul; associator vanishing; named walls |
| `self-hosted/mli/s3_gate_runner.sio` | X1–X13 |
| `self-hosted/mli/s3b_gate_runner.sio` | C1–C8 |
