<!-- docs:meta
topic_id: repo.tests.vectors.f128-f256.v0b-probe-consumption
authority: repo_only
audience: internal+codex
last_validated: 2026-08-17
validated_by: grok-cli3
source_of_truth: tests/vectors/f128_f256/V0B_PROBE_CONSUMPTION.json
-->

# V0-B probe consumption of MPFR literal_boundary corpora

**Date**: 2026-08-17  
**Emitter**: `scripts/dev/ws_g_v0b_emit_literal_probes.py`  
**Gate**: `scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0b`  
**Oracle**: [PR #1761](https://github.com/Sounio-lang/sounio/pull/1761) (GREEN / mergeable; MPFR external; **not** Sounio-derived).  
Tip reconciled 2026-08-17 against `d9ee9312ca` (literal boundary + V0-D hard-case add-on).  
In-tree `literal_boundary_*.jsonl` are **byte-identical** to that PR head.  
**Two corpora on #1761 — only the V0-B set is wired into these probes.**

| #1761 corpus family | Path | V0-B probes |
|---|---|---|
| **V0-B literal boundary** | `tests/vectors/f128_f256/literal_boundary_f{128,256}.jsonl` | **Consumed** |
| Bulk arithmetic (Wave 1) | `tests/vectors/f128_f256/f{128,256}.jsonl` | Not consumed (V0-D) |
| **V0-D hard cases** (halfway / sticky / tie-even / subnormal / Rump) | `tests/vectors/f128_f256_v0d/arith_hard_f{128,256}.jsonl` (27+25) | **Not consumed** (V0-D) |

## Consumed (wired into V0-B probes)

| Corpus | Role | Rows | Embedded as source literals | Limb-oracle only |
|---|---|---:|---:|---:|
| `literal_boundary_f128.jsonl` | V0-B source → bits | 53 | 48 | 5 (unary `-`) |
| `literal_boundary_f256.jsonl` | V0-B source → bits | 49 | 45 | 4 (unary `-`) |

**Double-rounding traps** (`double_rounds_differs=true`):

| Format | Total traps | Embedded as source | Limb-only (unary minus) |
|---|---:|---:|---:|
| f128 | 18 | 17 | 1 (`-0x1.f…p+16383`) |
| f256 | 16 | 16 | 0 |

Every embedded trap appears as `let v_N: fXXX = <source_literal>` plus
`ORACLE_<id>_EXPECTED` / `ORACLE_<id>_VIA_F64` limb tables (LSW-first i64)
copied from the JSONL — **MPFR ground truth, not Sounio**.

Notable embedded traps: `0.1`, `0.2`, `0.3`, `1.1`, long π/e digit strings,
`1e-20`, `1.0000000000000002`, hexfloat midpoints
`0x1.00000000000008p+0` / `…01p+0` / `…001p+0`, `9.999999999999999e-1`.

Probes:

- `tests/run-pass/f128_v0b_literal_smoke.sio`
- `tests/run-pass/f256_v0b_literal_forms.sio`

## Not consumed at V0-B (and why)

| Asset | Why unused at V0-B |
|---|---|
| `f128.jsonl` / `f256.jsonl` (4414 + 4411 rows) | **Arithmetic** ops (add/sub/mul/div/cmp). Ladder defers ops to **V0-D** softfloat. |
| `tests/vectors/f128_f256_v0d/arith_hard_f128.jsonl` (27) + `arith_hard_f256.jsonl` (25) | V0-D **hard cases** (halfway results, sticky-bit, tie-to-even, subnormals, Rump sign-inversion under short precision). Present in-tree from #1761 for the softfloat lane; **not** V0-B literals. |
| `gen/mpfr_vector_gen.c` / `f128_f256_v0d/gen/arith_hard_gen.c` | Generators for arithmetic corpora — V0-D. |
| Source spellings with leading `-` (e.g. `-0`, `-1`, `-0x1p-16494`) | **Sounio has no unary minus** (`0 - x` only). Rows remain in the JSONL and as `ORACLE_*` limb tables in the probe, but are **not** emitted as source literals. |
| Live bit-identity assert (`literal bits == expected.limbs`) | Requires limb extraction / run path after E249 lifts. Tables are embedded now so a widen-f64 implementer has the external expected vs via_f64 pair in-tree; gate today only checks embedding + E249. |

## Gate behaviour (V0-B green on Madaros as of 2026-09-06)

1. Verify sha256 of both literal_boundary JSONL files against `GENERATION_RECEIPT.md`.
2. Verify every `double_rounds_differs` row has `expected.limbs != via_f64.limbs`.
3. Verify probes embed those source strings + oracle tables.
4. Positive control `hello.sio` → `check: OK`.
5. Positive probes → must be `check: OK` without `error[E249]` to pass stage.
6. Negatives (arith/cast/implicit) → must not `check: OK`.

Under Madaros V0-B, steps 4–6 pass. Live limb-identity assert remains V0-D.

## Regenerate probes after vector updates

```bash
python3 scripts/dev/ws_g_v0b_emit_literal_probes.py
bash scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0b
```
