<!-- docs:meta
topic_id: repo.tests.vectors.f128-f256-v0d.f256-only-traps
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: grok-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.tests.vectors.f128-f256-v0d.f256-only-traps
-->

# What f256 softfloat requires that f128 does not (structurally)

**Date:** 2026-08-17  
**Oracle:** MPFR 4.2.1, RNE  
**Corpus:** `f256_only_traps.jsonl` (28 vectors)  
**Generator:** `gen/f256_only_traps_gen.c`

This is not “f256 is bigger so mul is slower.” It is: **which incorrect algorithms produce wrong bits only when the destination format is wider than binary128.**

## Limb geometry

| Format | P | Trailing bits | Significand limbs (64-bit) | Full schoolbook product limbs |
|---|---:|---:|---:|---:|
| binary128 | 113 | 112 | **2** | **4** |
| binary256 | 237 | 236 | **4** | **8** |

A correct IEEE mul must form (or simulate) the full product and reduce with
guard / round / **sticky**. Dropping the low half of the product without sticky
is incorrect at any width; at f256 the natural mistaken implementation is a
**4×4→4** limb schoolbook (keep high half only). That is the 4-limb error
class — f128’s mistaken form is 2×2→2, a different code shape.

## Trap classes found (measured)

### 1. `f128_cascade_mul` — intermediate IEEE format is binary128

```
direct  = RNE_f256( a_f256 * b_f256 )          # full product, one RNE
cascade = RNE_f256( RNE_f128(a) * RNE_f128(b) )  # operands+product through f128
```

**Measured:** paths differ for inputs with significand mass past bit 113
(e.g. `0x1.00000000000000000000000000001p+0` squared: direct low limb `16`,
cascade `0`).

**Why no f128 analogue of the same shape:** the cascade intermediate is
**binary128**. An f128 implementation’s only narrower IEEE binary cascade is
**binary64** (already catalogued as double-rounding / widen-f64). There is no
IEEE binary format between f128 and f256 other than… nothing — so
**“round through f128” is an f256-only architectural mistake.**

### 2. `triple_round_mul` — three successive IEEE roundings on a product

```
direct = RNE_f256(a*b)
triple = RNE_f256( RNE_f128( RNE_f64(a)*RNE_f64(b) ) )
```

**Measured:** differs on the same past-f128 significand family.

**Why f256-only as a *triple*:** f128 admits at most a **two-step** cascade
(f64→f128). The third stage exists only when the destination is wider than
f128.

### 3. `schoolbook_no_sticky` — 4-limb product truncate without sticky

```
direct = RNE_f256(a*b)                         # sticky from full product
wrong  = encode( trunc_toward_zero_to_p(a*b) ) # MPFR_RNDZ to p=237
```

**Measured:** e.g. `0.1*0.1` — low limb off by 1 (`…6193` vs `…6194`).

**Structural note:** this is the multiprecision bug “discard low product limbs.”
It is *more severe* at 4-limb significands (8-limb products) than at 2-limb
(f128). Vectors mark `why_f256_only` as the 4-limb algorithm class.

## What we are *not* claiming

- That f128 softfloat is easy (it is not).
- That schoolbook is always wrong if implemented with full sticky (it is fine,
  only slower).
- That every f256 mul must avoid all f128 use for *non-finite* or exact cases
  (many dyadic cases agree on both paths).

## Consumer rule for V0-D f256

Any softfloat mul that:

1. multiplies via binary128 intermediates, or  
2. rounds a×b through f64 then f128 then f256, or  
3. keeps only the high 237 product bits with sticky≡0  

**must fail** the corresponding rows in `f256_only_traps.jsonl` against
`result_direct`.

## Regeneration

```sh
cd tests/vectors/f128_f256_v0d/gen
./run_f256_only_traps.sh
```
