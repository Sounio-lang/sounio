<!-- docs:meta
topic_id: repo.tests.vectors.f128-f256-v0d.generation-receipt
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: grok-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.tests.vectors.f128-f256-v0d.generation-receipt
-->

# V0-D arithmetic hard-case corpus — generation receipt

**Date:** 2026-08-17  
**Agent:** grok-cli1 · lane `ws-g-v0d-arith-vectors`  
**Constraint:** corpus only — no softfloat implementation.

## Relation to prior corpora

| Corpus | Kept? | Role |
|---|---|---|
| `tests/vectors/f128_f256/f128.jsonl` (minimax `8ce767f33b`) | **untouched** | broad add/sub/mul/div/cmp |
| `tests/vectors/f128_f256/literal_boundary_*.jsonl` (Wave 3) | **untouched** | V0-B literals / double-round |
| **this directory** | **new** | structured hard cases for V0-D |

Sibling path `f128_f256_v0d/` chosen so writes do not collide with
`grok-cli3` claim on `tests/vectors/f128_f256/**` while remaining next to
the literal/arithmetic corpora.

## Toolchain

| Component | Value |
|---|---|
| Tool | **MPFR** |
| Version | **4.2.1** (`mpfr_get_version()` / `pkg-config --modversion mpfr`) |
| GMP | linked `-lgmp` |
| Rounding | **MPFR_RNDN** (IEEE-754 round-ties-to-even) |
| Extended precision | **4096** bits for exact op evaluation before encode |
| Host double (Rump only) | C `double` + libm, then `mpfr_set_d` + encode |
| OS / compiler | Ubuntu 24.04 pod, gcc + `-lmpfr -lgmp -lm` |

Build / run:

```
cd tests/vectors/f128_f256_v0d/gen
gcc -O2 -Wall -Wextra -Wno-unused-function \
  -o arith_hard_gen arith_hard_gen.c -lmpfr -lgmp -lm
./run.sh
```

## Exact invocation (per op family)

Binary ops (`add`/`sub`/`mul`/`div`):

```
mpfr_init2(a|b|r, 4096);
/* build a,b via mpfr_set_ui / mpfr_set_ui_2exp / mpfr_add of powers of two */
mpfr_{add,sub,mul,div}(r, a, b, MPFR_RNDN);
/* encode r → IEEE binary128 or binary256 wire (sign, biased exp, trailing, limbs) */
```

Sqrt:

```
mpfr_sqrt(r, a, MPFR_RNDN);  /* domain: a < 0 → NaN */
```

Rump 1988 (`a=77617`, `b=33096`):

```
exact:  evaluate
  333.75*b^6 + a^2*(11*a^2*b^2 - b^6 - 121*b^4 - 2) + 5.5*b^8 + a/(2*b)
  entirely in MPFR at p=4096, MPFR_RNDN; then RNE-encode to binaryN
f64_result: same AST in IEEE binary64 (C double), then mpfr_set_d + encode
```

Generator path recorded on every vector:
`tests/vectors/f128_f256_v0d/gen/arith_hard_gen.c`.

## Coverage

| Family | f128 | f256 | Ops exercised |
|---|---:|---:|---|
| halfway_tie_even | 6 | 6 | add, mul |
| sticky_bit | 6 | 6 | add, mul, div, sub |
| **catastrophic_cancel** | **31** | **30** | sub, add, mul, div (expanded 2026-08-17) |
| overflow_underflow | 3 | 2 | mul |
| sqrt_hard | 6 | 5 | sqrt |
| rump | 1 | 1 | closed-form expression |
| **Total** | **53** | **50** | |

Cancellation expansion (2026-08-17): eps ladder `2^-k` for k∈{1,2,4,…,80},
signed cancel, opposite-sign add, Sterbenz at exp 10/40/80/100/200, large
integers past f32 span, product `(1±e)` forms, near-equal subnormals, div near-1.

Spot checks (this generation):

| Case | Expectation | Result |
|---|---|---|
| `1 + ulp/2` (f128 add) | RNE → exactly `1.0` (exp=16383, trail=0) | PASS |
| Rump exact | negative, magnitude ~0.827… | sign=1, exp≈16382 |
| Rump host double | magnitude ~1.18e21 (wrong) | `f64_bits_differ=true` |
| Provenance fields | tool/version/rounding/invocation on all rows | complete |
| Cancel count parity | ≥20 per format | 31 / 30 |

## Output hashes (post-cancellation expansion)

```
arith_hard_f128.jsonl  53 lines
  md5    = 2daa40def9d6ba64bac9151332994293

arith_hard_f256.jsonl  50 lines
  md5    = b4026ca0fe4ef157fd90b486fb1a3149
```

```
md5sum    tests/vectors/f128_f256_v0d/arith_hard_*.jsonl
sha256sum tests/vectors/f128_f256_v0d/arith_hard_*.jsonl
```

## What this is not

- Not a softfloat implementation.
- Not a replacement for minimax random/special op corpora.
- Not V0-B literal vectors (those remain under `f128_f256/literal_boundary_*.jsonl`).
