<!-- docs:meta
topic_id: repo.tests.vectors.f128-f256-v0d
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: grok-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.tests.vectors.f128-f256-v0d
-->

# f128 / f256 V0-D arithmetic hard-case corpus (MPFR oracle)

External-oracle vectors for **softfloat arithmetic** (add, sub, mul, div, sqrt)
at cases where a naive implementation is right on random inputs and wrong on
structure. Produced **before** any Sounio softfloat exists — the corpus is the
judge, not a peer of the implementation.

Sibling of `tests/vectors/f128_f256/` (literals + general arithmetic from
minimax-cli3 / Wave 3). This directory does **not** replace those files.

| Sibling corpus | Role |
|---|---|
| `f128_f256/literal_boundary_*.jsonl` | V0-B literal / double-round traps |
| `f128_f256/f128.jsonl` / `f256.jsonl` | Broad random+special op coverage |
| **`f128_f256_v0d/arith_hard_*.jsonl`** | **Structured hard cases for V0-D** |

## Why not only the minimax op corpus?

`f128.jsonl` is large and useful, but mostly random + specials. V0-D must also
be graded on cases chosen because of **IEEE structure**:

| Family | Failure mode if wrong |
|---|---|
| `halfway_tie_even` | Tie-to-even ignored → off-by-1 ulp on midpoints |
| `sticky_bit` | Sticky ignored → wrong direction when bits sit below half-ulp |
| `catastrophic_cancel` | Guard digits lost → totally wrong small results |
| `overflow_underflow` | Gradual underflow / overflow mishandled |
| `sqrt_hard` | Non-square / subnormal sqrt mis-rounded |
| `rump` | Ill-conditioned poly; host `double` is wrong by ~10^21 |

## Files

```
tests/vectors/f128_f256_v0d/
├── README.md
├── GENERATION_RECEIPT.md
├── arith_hard_f128.jsonl    # 53 vectors (31 cancellation)
├── arith_hard_f256.jsonl    # 50 vectors (30 cancellation)
└── gen/
    ├── arith_hard_gen.c
    ├── run.sh
    └── .gitignore
```

## Schema (one JSON object per line)

```jsonc
{
  "id": "f128_arith_0001",
  "format": "binary128",          // or binary256
  "op": "f128_add",               // f{128,256}_{add,sub,mul,div,sqrt,rump1988}
  "family": "halfway_tie_even",
  "arity": 2,                     // 1 for sqrt
  "a": { "class", "sign", "exponent", "trailing_hex", "limbs": [/* LE i64 */] },
  "b": { ... } | null,            // null for sqrt
  "result": { ... },
  "rounding": "rne",
  "f64_sign_differs": false,      // true if host-double path flipped sign
  "provenance": {
    "tool": "MPFR",
    "version": "4.2.1",
    "rounding_mode": "MPFR_RNDN",
    "extended_precision_bits": 4096,
    "invocation": "…",
    "generator": "tests/vectors/f128_f256_v0d/gen/arith_hard_gen.c",
    "notes": "…"
  }
}
```

Rump rows also carry `f64_result`, `f64_bits_differ`, `f64_host_double`,
`expression`, and `provenance.citation`.

`limbs` match `IrWideNumericPayloadPool` (2 limbs f128, 4 limbs f256, little-endian).

## Consumer rules (V0-D)

1. Implement softfloat **without** reading these files into the algorithm.
2. For each vector, run the named op on `a` (and `b`) and require bit-identity
   with `result` under RNE.
3. For `family=rump`, the oracle is `result` (MPFR exact path), **not**
   `f64_result` (host double is intentionally wrong).
4. Do not use Sounio output as a second oracle.

## Regenerate

```sh
cd tests/vectors/f128_f256_v0d/gen
./run.sh
```

## Provenance

See `GENERATION_RECEIPT.md`: tool, version, rounding, exact invocation, hashes.
