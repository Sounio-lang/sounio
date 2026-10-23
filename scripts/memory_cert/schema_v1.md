# Memory Certificate v1

Machine-checkable allocation contract for a Madaros (or lean_single) generation.

## Object shape (JSON)

```json
{
  "version": 1,
  "generation": 0,
  "kind": "measured",
  "source_tree": "<git sha or 'unknown'>",
  "elf_sha256": "<hex or empty>",
  "created_utc": "2026-10-22T00:00:00Z",
  "phases": [
    {
      "id": "Parse",
      "bound_bytes": 536870912,
      "bound_kind": "Estimated",
      "confidence": 0.5,
      "provenance": "founder_fixed_arena_2GiB_split",
      "measured_peak_bytes": null
    }
  ],
  "notes": "Gen 0 seed: declared arenas, not proved."
}
```

## `bound_kind`

| Kind | Meaning |
|------|---------|
| `Exact` | Lean/SMT proved; actual ≤ bound always |
| `UpperBound` | Proved upper; may be loose |
| `Estimated` | Profiled μ (optionally with confidence) |
| `Unknown` | No bound — verify must fail unless allow_unknown |

## Verification rules

1. JSON parses; `version == 1`
2. Every required phase id present: `Parse`, `Check`, `Lower`, `Codegen`
3. Each `bound_bytes` ≥ 0 and finite
4. No `Unknown` unless `--allow-unknown`
5. If `elf_sha256` non-empty and `--elf PATH` given: hashes match
6. Fixed-point: two certs equal on `(phases[].id, bound_bytes, bound_kind)` and elf hashes match when both set
