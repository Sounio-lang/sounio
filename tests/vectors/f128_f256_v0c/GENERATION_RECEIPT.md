<!-- docs:meta
topic_id: repo.tests.vectors.f128-f256-v0c.generation-receipt
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: grok-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.tests.vectors.f128-f256-v0c.generation-receipt
-->

# V0-C wire / limb encoding corpus — generation receipt

**Date:** 2026-08-17  
**Agent:** grok-cli1 · lane `ws-g-v0c-wire-vectors`  
**Constraint:** corpus only — no limb-pool implementation.

## Provenance (every vector)

| Field | Value |
|---|---|
| tool | `structural-ieee754` |
| version | **IEEE-754-2008** (binaryN layout §3.6) |
| rounding_mode | `n/a-encoding` (no arithmetic) |
| invocation | hand-constructed from format parameters (see generator) |
| generator | `tests/vectors/f128_f256_v0c/gen/wire_encoding_gen.c` |
| citation | IEEE 754-2008 §3.6; same K/P/bias/emax as `f128_f256_format_descriptor_probe.sio` |

Build:

```
gcc -O2 -Wall -Wextra -Wno-format -o wire_encoding_gen wire_encoding_gen.c
./run.sh
```

## Format parameters used

| | binary128 | binary256 |
|---|---|---|
| storage | 128 | 256 |
| K / P | 15 / 113 | 19 / 237 |
| trailing bits | 112 | 236 |
| bias | 16383 | 262143 |
| emax | 16383 | 262143 |
| all-1s exp | 32767 | 524287 |
| limbs (LE i64) | 2 | 4 |

Class derivation (accept/reject oracle for `class` field):

| class | exp | trailing |
|---|---|---|
| zero | 0 | 0 |
| subnormal | 0 | ≠0 |
| normal | 1..2·emax | any |
| inf | 2·emax+1 | 0 |
| nan (qNaN) | 2·emax+1 | quiet bit set |
| snan | 2·emax+1 | quiet bit clear, trail ≠0 |

## Coverage

| Family | f128 | f256 |
|---|---:|---:|
| valid_edge | 12 | 11 |
| limb_boundary | 4 | 5 |
| class_consistent | 1 | 0 |
| malformed_reject | 14 | 8 |
| **Total** | **31** | **24** |

Reject reasons exercised: `limb_count`, `truncated_trailing_hex`, `missing_field`,
`illegal_sign`, `exponent_range`, `class_mismatch`, `non_hex_trailing`,
`empty_limbs`, `limbs_trailing_disagree`, `overwidth`.

## Hashes

```
wire_f128.jsonl  31 lines  md5=b65edaea57f8f7e588b83c75d9573c37
wire_f256.jsonl  24 lines  md5=e04c1b4607226021260f2acf4d0e063a
```

## Not in scope

- Softfloat ops (V0-D: `f128_f256_v0d/`)
- Literal parse (V0-B: `literal_boundary_*.jsonl`)
