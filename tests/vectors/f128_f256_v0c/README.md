<!-- docs:meta
topic_id: repo.tests.vectors.f128-f256-v0c
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: grok-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.tests.vectors.f128-f256-v0c
-->

# f128 / f256 V0-C wire-format / limb-pool corpus

Encodings, not values. This is the judge for **V0-C** (wire format + limb pools)
before any limb decoder/encoder exists.

| Ladder stage | Corpus |
|---|---|
| V0-B literals | `tests/vectors/f128_f256/literal_boundary_*.jsonl` |
| **V0-C wire/limbs** | **`tests/vectors/f128_f256_v0c/wire_*.jsonl`** |
| V0-D softfloat hard cases | `tests/vectors/f128_f256_v0d/arith_hard_*.jsonl` |

## Families

| Family | Verdict | Meaning |
|---|---|---|
| `valid_edge` | accept | +0/−0, min/max subnormal, min normal, ±1, max finite, ±inf, qNaN, sNaN |
| `limb_boundary` | accept | single bits at LSW/MSW and cross-limb edges |
| `class_consistent` | accept | `class` matches (exp, trailing) derivation table |
| `malformed_reject` | **reject** | wrong limb arity, truncated hex, illegal sign/exp, class mismatch, empty limbs, overwidth |

## Schema

```jsonc
{
  "id": "f128_wire_0001",
  "format": "binary128",
  "family": "valid_edge",
  "verdict": "accept" | "reject",
  "reject_reason": null | "limb_count" | "truncated_trailing_hex" | ...,
  "encoding": { "class", "sign", "exponent", "trailing_hex", "limbs": [...] },
  "provenance": {
    "tool": "structural-ieee754",
    "version": "IEEE-754-2008",
    "rounding_mode": "n/a-encoding",
    "invocation": "...",
    "generator": "tests/vectors/f128_f256_v0c/gen/wire_encoding_gen.c",
    "citation": "IEEE 754-2008 §3.6 …",
    "notes": "..."
  }
}
```

Counts (this generation): **31** binary128 + **24** binary256; **33** accept + **22** reject.

## Consumer rules (V0-C)

1. **accept** → decoder yields the stated class/sign/exp/trailing; re-encode must match `limbs` / `trailing_hex`.
2. **reject** → fail closed; do not produce a silent default value.
3. Do not implement the decoder by hard-coding these IDs — use the rules in `GENERATION_RECEIPT.md`.

## Regenerate

```sh
cd tests/vectors/f128_f256_v0c/gen && ./run.sh
```
