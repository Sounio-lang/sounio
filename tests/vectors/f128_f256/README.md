<!-- docs:meta
topic_id: repo.tests.vectors.f128-f256
authority: repo_only
audience: users
last_validated: 2026-08-16
validated_by: minimax-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.tests.vectors.f128-f256
-->

# f128 / f256 MPFR Test-Vector Corpus

External-oracle test vectors for IEEE-754 binary128 and binary256 arithmetic
and comparison, generated from MPFR (the Multiple Precision Floating-Point
Reliable Library). Used to gate Sounio's future softfloat f128/f256
implementation (WS-G V0-D in the Madaros focus plan).

This directory contains **vectors only** — no Sounio arithmetic. The
generator lives in `gen/`; the corpora are `f128.jsonl` and `f256.jsonl`.

## Files

```
tests/vectors/f128_f256/
├── README.md                # this file
├── GENERATION_RECEIPT.md    # how every vector was produced, with hashes
├── f128.jsonl               # 4414 vectors, one per line
├── f256.jsonl               # 4411 vectors, one per line
└── gen/
    ├── mpfr_vector_gen.c    # the generator (C, MPFR + GMP)
    ├── run.sh               # rebuild + regenerate wrapper
    └── .gitignore           # excludes the compiled binary
```

## Quickstart for consumers

```python
import json
with open("tests/vectors/f128_f256/f128.jsonl") as f:
    for line in f:
        v = json.loads(line)
        op = v["op"]                       # e.g. "f128_add"
        a, b, r = v["a"], v["b"], v["result"]
        # a, b, r are dicts with class/sign/exponent/trailing_hex/limbs
        # for cmp_* ops, r is a plain bool
```

For Sounio consumers: each vector's `limbs` field is exactly the
little-endian `i64` sequence an `IrWideNumericPayloadPool` entry takes
(2 limbs for f128, 4 limbs for f256) — see
`self-hosted/compiler/f128_f256_numeric_payload_probe.sio`.

## Regenerating

```sh
cd tests/vectors/f128_f256/gen
./run.sh
```

Records tool versions, seeds, and hashes to stderr; updates
`f128.jsonl` and `f256.jsonl` in the parent directory. Output is
deterministic for a given toolchain — see `GENERATION_RECEIPT.md` for the
fixed PCG seed and the build command.

## Why MPFR and not "another f128 library"

MPFR is the standard, well-tested arbitrary-precision reference; it gives
us effectively infinite precision for the computation step, and we then
apply IEEE-754 RNE rounding manually to land on the target format. This
isolates the rounding-logic correctness question (which is what bites
implementations) from the question "is MPFR correct at large precision"
(whose answer is "yes, with extreme thoroughness" — see the MPFR
correctness proofs).

A *separate* cross-check against a non-MPFR oracle (e.g. SoftFloat,
libbid) is a wave-3 follow-up; see the receipt's "Cross-validation
against an independent oracle (future work)" section.

## Lane / coordination

Lane `ws-g-mpfr-vectors`, agent `minimax-cli3`, per
`docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md` §1 WS-G
and §3 Wave 1.