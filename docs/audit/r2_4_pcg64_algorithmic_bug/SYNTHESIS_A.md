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

**CERTIFIED CANONICAL (2026-05-17).** Operator directive: "TRUE B2 — real
serious PL, no tech debt." Sounio reference implementation of
PCG-XSL-RR-128/64-LCG (pcg-cpp's `pcg64`) cross-checked against pcg-cpp HEAD
on 4 seeds × 8 samples = **32/32 bit-exact** (see fingerprint table below).
Full 256-sample × 4-seed oracle streams regenerated and committed.

## Algorithm chosen

**PCG-XSL-RR-128/64-LCG (canonical PCG64)**, O'Neill 2014 §6.3.

- 128-bit state emulated on two `i64` as `(state_hi, state_lo)`.
- LCG: `state ← state * MULT_128 + inc (mod 2^128)`.
- `MULT_128 = (2549297995355413924 << 64) | 4865540595714422341`.
- Output function XSL-RR: `xorshifted = (state >> 64) ^ state` (low 64 bits),
  `rot = (state >> 122) & 63`, `output = rotr64(xorshifted, rot)`.
- **Output coupling:** matches `pcg-cpp`'s `setseq_base` template default
  `output_previous = (sizeof(itype) <= 8)`; for `pcg64` itype is `pcg128_t`
  (sizeof = 16) ⇒ `output_previous = false` ⇒ `operator()` returns
  `output(base_generate())` (NEW state after advance).

The PCG32-doubled approach previously certified is preserved at
`reference/pcg32_attempt/` for historical record but is **not canonical**.

## Implementation

`reference/pcg64_reference.sio` — 138 lines, Sounio-native, no Python.

Primitives (all verified bit-exact via 18-assertion `helpers_selftest.sio`):
- `u_lt(a_hi, a_lo, b_hi, b_lo)` — unsigned 128-bit `<`.
- `lshr(x, n)` — logical right shift on i64.
- `umul64_high(a, b)` — high 64 of u64 × u64 via 32-bit half decomposition.
- `u128_add` / `u128_mul` — mod-2^128 with carry detection via `u_lt`.
- `rotr64(x, rot)` — 64-bit right rotation.

Step + init:
- `pcg64_step(rng) -> (DstPcg64, i64)` — advance then output from NEW state.
- `pcg64_new(seed)` — pcg-cpp `seed(seed, stream)` semantics with stream=0.

## Outputs

Four oracle streams, 256 samples each, one i64 per line:

| File | First sample | sha256 |
|---|---|---|
| `reference/oracle_seed_0.txt`        | `74029666500212977`     | `7c34eb2c266a5fd213fdcefc430a289b3a5e85204d8c0aec5930dd9155c82c6b` |
| `reference/oracle_seed_1.txt`        | `-2200603052647351302`  | `32a1acd8bf9bbf6ac9c219343a0ae4facba407b39c3f255d828f996d1c48f50e` |
| `reference/oracle_seed_31415.txt`    | `-2825318976064776997`  | `b715e064ac45e24dd4a6d71aaf85bc23804b8d55a7ae60fbc2e85fafb8876975` |
| `reference/oracle_seed_20260516.txt` | `-3515258213216447036`  | `56d072e8f470aa83126103d810e43096d4e3cd88011a1a04aab3482655c61524` |

Total: 1024 samples across 4 seeds.

## How to regenerate

```bash
./bin/souc compile docs/audit/r2_4_pcg64_algorithmic_bug/reference/pcg64_oracle_gen.sio -o /tmp/oracle_gen
/tmp/oracle_gen > /tmp/all_oracle.txt
# Streams delimited by single-line seed markers (0, 1, 31415, 20260516).
awk '
  /^0$/        {seed="0"; next}
  /^1$/        {seed="1"; next}
  /^31415$/    {seed="31415"; next}
  /^20260516$/ {seed="20260516"; next}
  {print > ("oracle_seed_" seed ".txt")}
' /tmp/all_oracle.txt
```

## External verification (pcg-cpp HEAD)

```cpp
#include "pcg-cpp/pcg_random.hpp"
pcg64 rng(seed);
for (int i = 0; i < 8; ++i)
    std::cout << static_cast<int64_t>(rng()) << "\n";
```

Diff against Sounio fingerprint probe output:

```
$ ./verify_pcg64 > /tmp/pcgcpp_out.txt
$ /tmp/sounio_fp_probe > /tmp/sounio_out.txt
$ diff /tmp/pcgcpp_out.txt /tmp/sounio_out.txt && echo MATCH
MATCH
```

### Fingerprint table (first 8 samples per seed)

| seed | i | pcg-cpp | Sounio | match |
|---|---|---|---|---|
| 0 | 0 | 74029666500212977 | 74029666500212977 | ✓ |
| 0 | 1 | 8088122161323000979 | 8088122161323000979 | ✓ |
| 0 | 2 | -1924914382715075334 | -1924914382715075334 | ✓ |
| 0 | 7 | -5944627205623820838 | -5944627205623820838 | ✓ |
| 1 | 0 | -2200603052647351302 | -2200603052647351302 | ✓ |
| 1 | 7 | 2161154022852409228 | 2161154022852409228 | ✓ |
| 31415 | 0 | -2825318976064776997 | -2825318976064776997 | ✓ |
| 31415 | 7 | 3656666283514925126 | 3656666283514925126 | ✓ |
| 20260516 | 0 | -3515258213216447036 | -3515258213216447036 | ✓ |
| 20260516 | 7 | 2460099662311570537 | 2460099662311570537 | ✓ |

All 32/32 samples bit-exact. (Table abridged; full diff is empty.)

## Diagnostic: the off-by-one bug found and fixed

Initial draft of `pcg64_step` computed the XSL-RR output from the **old**
state (pre-advance). pcg-cpp's `operator()` for `pcg64` outputs from the
**new** state because `setseq_base<itype=pcg128_t>` selects
`output_previous=false`. The Sounio first sample was therefore pcg-cpp's
"sample −1" (output of the seeded state before any advance), shifting the
whole stream by one.

Fix: compute `xorshifted` and `rot` from `(add.hi, add.lo)` (the new state
returned by the LCG advance), not from `(old_hi, old_lo)`. One-location
change in `pcg64_reference.sio:pcg64_step`.

## Out of scope for Phase A

- Statistical sanity (mean / variance) — Phase C.
- Patching `stdlib/random/distributions.sio` — Phase B.
- Removing deprecation header from `random/distributions.sio` — Phase D.

## Phase A complete — Phase B authorized

The 256-sample × 4-seed oracle in `reference/oracle_seed_*.txt` is canonical
by external pcg-cpp cross-check. Phase B should rewrite the dst_pcg64 path
in `stdlib/random/distributions.sio` to the algorithm above and validate
byte-equal against these oracle files.
