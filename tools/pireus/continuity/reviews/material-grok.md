CLEAR

Static review only (no execution claim). Admitted plans only: odd stride 1..15, offset 0..15, load/layout 0|1, unroll 1|2|4|8|16, 16-thread block, grid.x = vector count.

- Inverse lane map: `src = (j - offset) * inv(stride) & 15` recovers the thread that loaded `b[k]` with `k = (tid*stride+offset)&15`; u32 wrap is ≡ mod 16.
- Shuffle: `shfl.sync.idx` clamp 15 + membermask 65535 is the 16-lane CTA; both f64 halves, same src lane; uniform control flow with the pre-loop `b` load.
- Signs: `pireus_signs[j]` bit `k` is `cd_sigma(k^j,j)<0`; const `j*4` addressing; `selp` ±1.0 then separate `mul.rn`/`add.rn`.
- Addressing: layout0 `vec*16+coef`, layout1 `coef*nctaid+vec`; `mul.wide` byte offset; driver AoS/SoA transpose matches; `nctaid≤2^24` keeps u32 indices in range.
- Unroll: `j = r4+u` with `r4 += unroll` (or fully unrolled 16) is ascending `j=0..15`; `i=k XOR j`; no FMA; +0.0 init.
- Numeric/parity: fixtures and kernel share that sequence; `finite-bits-nan-class-v1` is exact bits for non-NaNs (inf and signed zero included), NaN payloads not compared; Python only transports bits.
