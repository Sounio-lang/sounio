<!-- docs:meta
topic_id: repo.tests.vectors.f128-f256
authority: repo_only
audience: users
last_validated: 2026-08-16
validated_by: minimax-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.tests.vectors.f128-f256
-->

# f128 / f256 MPFR Test-Vector Corpus

External-oracle test vectors for IEEE-754 binary128 and binary256, generated
from **MPFR** (not Sounio). Two families:

| Family | Files | Role |
|---|---|---|
| **Arithmetic ops** (Wave 1, minimax-cli3) | `f128.jsonl`, `f256.jsonl` | add/sub/mul/div/cmp vs MPFR — gates V0-D softfloat |
| **Literal / boundary** (Wave 3, grok-cli1) | `literal_boundary_f128.jsonl`, `literal_boundary_f256.jsonl` | source string → bit pattern; **double-rounding traps** for V0-B/V0-C probes |

A probe that checks Sounio only against Sounio cannot detect a systematically
wrong implementation. These corpora are the external ground truth.

## Files

```
tests/vectors/f128_f256/
├── README.md                      # this file
├── GENERATION_RECEIPT.md          # provenance + hashes (both generators)
├── f128.jsonl                     # 4414 arithmetic vectors (minimax-cli3)
├── f256.jsonl                     # 4411 arithmetic vectors
├── literal_boundary_f128.jsonl    # 53 literal/boundary vectors (Wave 3)
├── literal_boundary_f256.jsonl    # 49 literal/boundary vectors
└── gen/
    ├── mpfr_vector_gen.c          # arithmetic corpus generator
    ├── run.sh
    ├── literal_boundary_gen.c     # literal + double-round generator
    ├── run_literal_boundary.sh
    └── .gitignore
```

## Relationship to minimax-cli3 (`8ce767f33b`)

**Reused, not replaced.** The arithmetic corpora and `mpfr_vector_gen.c` are
kept byte-identical to commit `8ce767f33b`. Wave 3 **extends** the directory
with a second generator and two new JSONL files aimed at V0-B literal probes
(especially double-rounding traps that a widen-from-f64 parser shortcut
passes while still being wrong).

## Quickstart — arithmetic (V0-D)

```python
import json
with open("tests/vectors/f128_f256/f128.jsonl") as f:
    for line in f:
        v = json.loads(line)
        op = v["op"]                       # e.g. "f128_add"
        a, b, r = v["a"], v["b"], v["result"]
```

## Quickstart — literal boundary (V0-B / V0-C)

```python
import json
with open("tests/vectors/f128_f256/literal_boundary_f128.jsonl") as f:
    for line in f:
        v = json.loads(line)
        lit = v["source_literal"]          # e.g. "0.1", "0x1p-16494"
        expected = v["expected"]           # direct string→binaryN (MPFR RNE)
        via_f64 = v["via_f64"]             # string→f64 RNE→widen
        if v["double_rounds_differs"]:
            # a correct f128 parse must match expected, NOT via_f64
            ...
```

Families in the literal corpora:

| `family` | Meaning |
|---|---|
| `exactly_representable` | Dyadic / integer values with unique encoding |
| `provably_not_representable` | Non-dyadic decimals (need RNE) |
| `subnormal` | Min subnormal and near-range hexfloat |
| `min_normal` | Smallest positive normal |
| `max_finite` | Largest finite magnitude |
| `ulp_neighbors` | 1.0 ± ulp and ulp-as-value |
| `double_rounding_trap` | Direct vs via-f64 encodings **differ** |
| `literal_boundary` | Spelling variants (1e0, 0X1.0P+0, …) |

Every vector carries a `provenance` object: tool, version, rounding mode,
invocation sketch, generator path, notes. Magic constants without derivation
are forbidden.

`limbs` is little-endian `i64` payload matching
`IrWideNumericPayloadPool` (2 limbs f128, 4 limbs f256).

## Regenerating

```sh
cd tests/vectors/f128_f256/gen
./run.sh                     # arithmetic corpora
./run_literal_boundary.sh    # literal/boundary corpora
```

## Why MPFR

MPFR is the standard extended-precision reference. Arithmetic vectors compute
at high precision then apply IEEE-754 RNE. Literal vectors parse the source
string at 2048-bit precision then RNE to binary128/256; the via-f64 path
rounds to 53-bit precision first. Neither path uses Sounio.

## Lane / coordination

- Wave 1 arithmetic: `minimax-cli3` / `ws-g-mpfr-vectors` (`8ce767f33b`)
- Wave 3 literal extension: `grok-cli1` / `ws-g-ref-vectors`
- **V0-C wire/limbs (sibling dir):** `tests/vectors/f128_f256_v0c/` — accept/reject encodings
- **V0-D hard-case arithmetic (sibling dir):** `tests/vectors/f128_f256_v0d/` —
  halfway / sticky / cancellation / Rump / sqrt; does not replace this tree