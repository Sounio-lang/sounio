<!-- docs:meta
topic_id: repo.docs.audit.r2-4-pcg64-algorithmic-bug.synthesis-a
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-4-pcg64-algorithmic-bug.synthesis-a
-->

# Phase A Synthesis — Reference oracle, TRUE PCG64-XSL-RR-128/64 (2026-05-17)

## Status

**PENDING EXTERNAL CROSS-CHECK (TRUE B2 pivot).** The earlier PCG32-doubled oracle (certified canonical) was discarded after operator directive "TRUE B2 — real serious PL, no tech debt." This synthesis records the rewritten PCG64-XSL-RR-128/64 reference. The PCG32 work is preserved verbatim at `reference/pcg32_attempt/` for historical record.

The new oracle requires fresh external cross-check against pcg-cpp's `pcg64` class.

## Algorithm chosen

**PCG-XSL-RR-128/64-LCG (true PCG64).** Per Melissa O'Neill 2014, §4.4. The same algorithm pcg-cpp exposes as `pcg_random::pcg64` and the named contract of the existing `DstPcg64` struct (4 i64 fields = full 128-bit state). Period 2^128. Quality matches the PCG paper's reference.

128-bit arithmetic emulated on Sounio i64 via 32-bit-half decomposition (`umul64_high`) and unsigned-semantics helpers (`u_lt`, `lshr`).

## Implementation

`reference/pcg64_reference.sio` — 138 lines, Sounio-native.

Core 128-bit primitives:
- `u_lt(a, b)` — unsigned less-than via top-bit XOR + signed compare.
- `lshr(x, n)` — logical right shift on signed i64 via mask.
- `umul64_high(a, b)` — high 64 bits of u64 × u64 by 32-bit-half decomposition + carry.
- `u128_add(hi, lo, hi, lo)` — 128-bit addition with carry detected via `u_lt`.
- `u128_mul(hi, lo, hi, lo)` — 128-bit multiplication mod 2^128: `res_lo = a_lo*b_lo`, `res_hi = umul64_high(a_lo, b_lo) + a_hi*b_lo + a_lo*b_hi`.
- `rotr64(x, rot)` — 64-bit right rotation.

PCG64 step:
- `pcg64_step(rng)` — state = state * MULT_128 + INC_128 (mod 2^128); output = `rotr64(state_hi_old XOR state_lo_old, (state_hi_old >> 58) & 63)`.
- `pcg64_new(seed)` — O'Neill's canonical pattern with `inc = PCG_DEFAULT_INCREMENT_128`: state=0; step; state += (0, seed); step.

Constants (verbatim from pcg-cpp `pcg_extras.hpp`):

| Constant | Decimal | Hex |
|---|---|---|
| PCG_DEFAULT_MULTIPLIER_HIGH_64 | 2549297995355413924 | 0x2360ED051FC65DA4 |
| PCG_DEFAULT_MULTIPLIER_LOW_64  | 4865540595714422341 | 0x4385DF649FCCF645 |
| PCG_DEFAULT_INCREMENT_HIGH_64  | 6364136223846793005 | 0x5851F42D4C957F2D |
| PCG_DEFAULT_INCREMENT_LOW_64   | 1442695040888963407 | 0x14057B7EF767814F |

INC_LOW low byte is `0x4F` (odd) → INC_128 is odd → satisfies the PCG odd-increment requirement. ✓

## Self-test of 128-bit primitives

Before generating the oracle, every helper was verified against hand-computed expected values in `reference/helpers_selftest.sio`. All 18 assertions bit-exact:

```
lshr(-1, 1)                    = 9223372036854775807    (0x7FFF...)
lshr(0x4000_0000_0000_0000,62) = 1
umul64_high(2^32, 2^32)        = 1                       (2^64 / 2^64)
umul64_high(2^33, 2^33)        = 4                       (2^66 / 2^64)
umul64_high(2^60, 2^60)        = 72057594037927936       (2^56)
umul64_high(2^63, 2)           = 1                       (2^64 / 2^64)
u128_add((0,-1), (0,1))        = (1, 0)                  (carry from low to high)
u128_add((0,1), (0,2))         = (0, 3)
u128_mul((0,1), MULT)          = MULT                    (identity)
u128_mul((0,2), (0,2^63))      = (1, 0)                  (carry path)
rotr64(1, 1)                   = -2^63                   (bit 0 → bit 63)
rotr64(2^63, 1)                = 2^62
rotr64(1, 0)                   = 1                       (no-op)
u_lt(0, -1)                    = true                    (-1 as u64 = MAX)
u_lt(-1, 0)                    = false
u_lt(1, 2)                     = true
```

Run via `./bin/souc compile reference/helpers_selftest.sio -o /tmp/st.elf && /tmp/st.elf`. Output pairs (expected then actual) match on every line.

## Outputs

Four oracle streams, 256 samples each, one i64 per line, regenerated under TRUE PCG64:

| File | Lines | First sample | sha256 |
|---|---|---|---|
| `reference/oracle_seed_0.txt` | 256 | `5591422465364813936` | `ce750f543285…` |
| `reference/oracle_seed_1.txt` | 256 | `6213437932455367576` | `dd76065adf35…` |
| `reference/oracle_seed_31415.txt` | 256 | `-5645831904906215705` | `f4d9059d2c73…` |
| `reference/oracle_seed_20260516.txt` | 256 | `379010195096432901` | `b83f61e17ab6…` |

Total: 1024 samples across 4 seeds. ELF-print emits as decimal i64; sign bit of i64 corresponds to high bit of u64 output.

## How to regenerate

```bash
./bin/souc compile docs/audit/r2_4_pcg64_algorithmic_bug/reference/pcg64_reference.sio -o /tmp/pcg64_ref.elf
/tmp/pcg64_ref.elf > /tmp/pcg64_all.txt
# Seed-marker lines (one per seed) delineate the 4 streams; split with awk
# (markers at NR 1, 258, 515, 772 in the 1028-line output).
```

## External verification spec for the operator

To certify the new oracle, the operator runs the **true** PCG64 in pcg-cpp:

```cpp
#include "pcg_random.hpp"
#include <iostream>
#include <cstdint>
int main() {
    pcg64 rng(static_cast<uint64_t>(31415));
    for (int i = 0; i < 8; ++i) {
        // Print as signed int64 to match Sounio i64 view:
        std::cout << static_cast<int64_t>(rng()) << "\n";
    }
}
```

Compile and run; compare line-by-line against the Sounio fingerprint emitted by `reference/pcg64_fingerprint_probe.sio`.

Sounio's first 8 outputs per seed (from `pcg64_fingerprint_probe.sio` run on `bin/souc` HEAD):

| seed | i | Sounio output (i64 dec) |
|---|---|---|
| 0 | 0 | `5591422465364813936` |
| 0 | 1 | `74029666500212977` |
| 0 | 2 | `8088122161323000979` |
| 0 | 3 | `-1924914382715075334` |
| 0 | 4 | `-7632739411327113122` |
| 0 | 5 | `9052198920789078554` |
| 0 | 6 | `7381380909356947872` |
| 0 | 7 | `-7485149332228263313` |
| 1 | 0 | `6213437932455367576` |
| 1 | 1 | `-2200603052647351302` |
| 31415 | 0 | `-5645831904906215705` |
| 31415 | 1 | `-2825318976064776997` |
| 20260516 | 0 | `379010195096432901` |
| 20260516 | 1 | `-3515258213216447036` |

(Full 8-sample tables for all 4 seeds available via `pcg64_fingerprint_probe.sio` run.)

If pcg-cpp agrees on seed=31415 first sample = `-5645831904906215705` (as i64, equivalently `0xB1AE6A8C70F32127` as u64 = `12800912168803335975` as u64) → oracle is canonical, proceed to Phase B.

If pcg-cpp disagrees → return to Phase A with the discrepancy. Most likely failure modes: (a) wrong constants in the Sounio reference (verify against pcg_extras.hpp), (b) bug in `umul64_high` carry chain (helpers self-test would have caught most, but a specific edge case might slip through).

## Out of scope for Phase A

- Statistical sanity (mean / variance) — that's Phase C.
- Comparing oracle to the current buggy `dst_pcg64` — they will obviously disagree.
- Re-deriving PCG_DEFAULT_INCREMENT_128 from seq — using the published default suffices for our seed-only API.

## Predecessor (PCG32-doubled, now superseded)

The earlier PCG32-doubled oracle was operator-certified on 2026-05-17 (8/8 fingerprint match vs pcg-cpp `pcg32`). It was discarded same-day after the "TRUE B2 — no tech debt" directive. Preserved at `reference/pcg32_attempt/`:

```
pcg32_attempt/
  pcg32_reference.sio          (95 lines, PCG-XSH-RR-64/32 doubled)
  pcg32_fingerprint_probe.sio
  oracle_seed_0.txt
  oracle_seed_1.txt
  oracle_seed_31415.txt
  oracle_seed_20260516.txt
```

If a quality-vs-cost reassessment ever points back at PCG32 (e.g., on resource-constrained targets), the work is salvageable from there.

## Phase A complete — pending external re-verification

Wall-clock spent on Phase A (TRUE PCG64): ~45 min (write 128-bit helpers + self-test + new reference + regenerate + writeup) on top of the ~30 min sunk on the PCG32 path. Total Phase A: ~75 min. Halt for operator's external cross-check before Phase B.
