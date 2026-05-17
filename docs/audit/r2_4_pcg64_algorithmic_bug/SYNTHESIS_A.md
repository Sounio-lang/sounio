# Phase A Synthesis — Reference oracle (2026-05-17)

## Status

**PENDING EXTERNAL CROSS-CHECK.** Streams generated in-workspace; the dispatch's §3.A.4 step (cross-check against PCG's published C reference, outside the workspace) is the operator's call before Phase B can use these as canonical.

## Algorithm chosen

**PCG-XSH-RR-64/32-LCG (PCG32), output doubled to 64-bit.** Per O'Neill 2014.

Why not PCG64-XSL-RR-128/64: that variant needs true 128-bit state arithmetic, which on i64 requires either 32-bit-half decomposition or careful carry emulation across `state_hi`/`state_lo`. The complexity of doing that correctly in Sounio (without a 128-bit primitive) outweighs the quality difference for our use case. PCG32 doubled has period ~2^62 per stream which is comfortably above the dissertation's Monte Carlo budget (max ~10^7 draws).

The dispatch §3.A.1 explicitly authorizes "the simpler PCG-XSH-RR for 64-bit state if the algorithm-cost trade-off matters." Cost matters here.

## Implementation

`reference/pcg64_reference.sio` — 95 lines, Sounio-native, no Python.

Core primitives:
- `lshr(x, n)` — logical right shift on signed i64 via mask `(x >> n) & ((1 << (64-n)) - 1)`. Strips Sounio's arithmetic-shift sign extension.
- `pcg32_step(rng) -> (Pcg32, i64)` — single PCG32 advance, output is 32 bits in the low half of an i64. Uses LCG with `mult = 6364136223846793005`, `inc = 1`.
- `pcg64_step(rng) -> (Pcg32, i64)` — calls `pcg32_step` twice and concatenates `(hi32 << 32) | lo32`.
- `pcg32_init(seed)` — O'Neill's canonical pattern: `state=0; step; state += seed; step`. With `initseq = 0` so `inc = 1`.

## Outputs

Four oracle streams, 256 samples each, one i64 per line:

| File | Size | First sample | sha256 |
|---|---|---|---|
| `reference/oracle_seed_0.txt` | 5227 B | `-1963209312182704874` | `b2a3d36f…` |
| `reference/oracle_seed_1.txt` | 5211 B | `-2145630622996943051` | `a50b1034…` |
| `reference/oracle_seed_31415.txt` | 5216 B | `6192612575141146067` | `b2177f60…` |
| `reference/oracle_seed_20260516.txt` | 5231 B | `-721871931154621307` | `3aa75419…` |

Total: 1024 samples across 4 seeds.

## How to regenerate

```bash
./bin/souc compile docs/audit/r2_4_pcg64_algorithmic_bug/reference/pcg64_reference.sio -o /tmp/pcg_ref.elf
/tmp/pcg_ref.elf > /tmp/oracle_all.txt
# Streams are delimited by single-line seed markers; split with awk.
# Each stream: 256 i64 samples, one per line.
```

## External verification spec for the operator

To certify these streams as canonical, the operator runs an external PCG32 reference (e.g. O'Neill's `pcg-cpp` `pcg_random::pcg32`) configured as:

- **Variant:** PCG-XSH-RR-64/32-LCG.
- **Multiplier:** `6364136223846793005`.
- **Increment:** `1` (corresponds to `initseq = 0`).
- **Init pattern:** `state = 0; step_once_discard; state += seed; step_once_discard;`
- **Output coupling for 64-bit:** call `pcg32_random_r()` twice per 64-bit sample. First call's u32 → high 32 bits; second call's u32 → low 32 bits. Concatenate as `((u64)hi << 32) | lo`. Reinterpret as i64 for printing (so the high bit of u64 becomes the sign bit of i64).

Expected: first sample from `seed = 31415` should be `6192612575141146067` (= `0x55F0AA10AB2EF553` as u64).

If external reference disagrees on the very first sample, the in-workspace implementation has a mismatch worth investigating before Phase B touches `distributions.sio`.

If external reference agrees, declare the oracle canonical and proceed to Phase B.

## Out of scope for Phase A

- Statistical sanity (mean / variance) — that's Phase C.
- Comparing oracle to current buggy `dst_pcg64` — they will obviously disagree (the whole point of R.2.4).
- Reformatting oracle files to binary or different separators — text-i64-per-line is the simplest test-harness contract.

## Halt point

Operator confirms external cross-check passes, then Phase B (apply B2 or B3 to `distributions.sio`) is authorized. If the external check disagrees, return to Phase A with the discrepancy.

Wall-clock spent on Phase A: ~25 min (write reference + generate + split + verify + writeup).
