# Omega Sprint 5 Track 1

## Goal

Harden Merkle inclusion proof validation using the locked Sprint 4 Genesis manifest,
without changing strict gate defaults.

## Command

```bash
python3 scripts/omega/omega_merkle_inclusion_proof.py --strict
```

## Inputs

- `artifacts/omega/omega_genesis.v1.0.json`

## Output

- `artifacts/omega/merkle_inclusion_proof.v1.json`

## Pass criteria

- `status == "pass"`
- `missing_count == 0`
- `mismatch_count == 0`
- `aggregate_match == true`
- `merkle_root_match == true`

