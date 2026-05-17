<!-- docs:meta
topic_id: repo.docs.audit.r2-4-pcg64-algorithmic-bug.synthesis-a
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-4-pcg64-algorithmic-bug.synthesis-a
-->

# Phase A Synthesis — Reference oracle (2026-05-17)

## Status

**CERTIFIED CANONICAL (2026-05-17).** Operator ran an external PCG-cpp PCG32 reference and reported first 8 individual 32-bit samples for seed 31415. Sounio implementation matches bit-exact across all 8 — fingerprint cross-check documented at end of this file. Oracle is canonical; Phase B authorized.

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

Expected: first 64-bit sample from `seed = 31415` should be `6192612575141146067` (= `0x55F09453C4E7ADD3` as u64; high32=`0x55F09453`=1441829971 from pcg32 call 1, low32=`0xC4E7ADD3`=3303517651 from pcg32 call 2). The hex `0x55F0AA10AB2EF553` printed in earlier draft was a transcription error — corrected here.

If external reference disagrees on the very first sample, the in-workspace implementation has a mismatch worth investigating before Phase B touches `distributions.sio`.

If external reference agrees, declare the oracle canonical and proceed to Phase B.

## Out of scope for Phase A

- Statistical sanity (mean / variance) — that's Phase C.
- Comparing oracle to current buggy `dst_pcg64` — they will obviously disagree (the whole point of R.2.4).
- Reformatting oracle files to binary or different separators — text-i64-per-line is the simplest test-harness contract.

## Fingerprint cross-check (operator's external reference vs Sounio impl)

Operator's external PCG-cpp PCG32 reference for seed 31415 produced these first 8 individual 32-bit samples. Sounio's `pcg32_step` (compiled via `bin/souc` HEAD) emits identical values via `reference/fingerprint_probe.sio`:

| idx | external u32 | external hex | Sounio output | match |
|---|---|---|---|---|
| 0 | 1441829971 | `0x55F09453` | 1441829971 | ✓ |
| 1 | 3303517651 | `0xC4E7ADD3` | 3303517651 | ✓ |
| 2 | 117924292  | `0x070761C4` | 117924292  | ✓ |
| 3 | 106048764  | `0x06522CFC` | 106048764  | ✓ |
| 4 | 3207584603 | `0xBF2FDB5B` | 3207584603 | ✓ |
| 5 | 3398509113 | `0xCA912239` | 3398509113 | ✓ |
| 6 | 516904547  | `0x1ECF5663` | 516904547  | ✓ |
| 7 | 4032317196 | `0xF058470C` | 4032317196 | ✓ |

8/8 bit-exact. Combined first 64-bit sample = `(0x55F09453 << 32) | 0xC4E7ADD3` = `0x55F09453C4E7ADD3` = `6192612575141146067` as i64, matching `head -1 oracle_seed_31415.txt`.

The 256-sample streams in `reference/oracle_seed_*.txt` are now declared canonical by external verification.

## Phase A complete — Phase B authorized

Operator authorized Phase B based on this fingerprint match. The fix in `stdlib/random/distributions.sio` should be validated against the byte-level oracle (256 samples per seed, all 4 seeds) and the statistical-sanity probe described in DISPATCH §3.C.

Wall-clock spent on Phase A: ~25 min (write reference + generate + split + verify + writeup) plus ~5 min for fingerprint cross-check addendum.
