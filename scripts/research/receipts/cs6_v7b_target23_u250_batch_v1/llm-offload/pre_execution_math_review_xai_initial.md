VERDICT FAIL

1. [CRITICAL] Python // vs C trunc-toward-zero on signed qdiv
   Python `a//d` floors; FPGA/C truncates toward zero. Any negative numerator in RK4 state or bisection produces bit mismatch.
   Correction: Python model must emulate truncdiv: `int(a/d)` (or `(a^d)<0 ? (a+sign(d)*(abs(d)-1))//d : a//d`).

2. [HIGH] qmul arithmetic right-shift on negative products
   `(ap_int<128>)a*b >> 40` sign-extends correctly only if the 128-bit product is interpreted as signed; synthesis may widen differently. Python must cast to signed 128-bit before `>>40`.

3. [HIGH] Bisection midpoint `(lo+hi)/2`
   In Q24.40 two's-complement, `lo+hi` overflows int64 when both are near INT64_MIN/MAX; FPGA may wrap or saturate while Python does not. Replace with `lo+(hi-lo)/2`.

4. [MEDIUM] RK4 final scale `/6144`
   6144 = 6<<10, but the four adds and the division are performed sequentially with rounding after each qmul/qdiv. Accumulated rounding error differs from a single fused multiply; Python must replicate the exact sequence of four qmuls.

5. [LOW] Event return uses `hi_state` after 24 bisections
   Final state is taken from the last accepted hi, not re-integrated at the refined time; this is coherent for detection but silently widens the reported crossing interval by up to 2^-10 * 2^-24. Document or correct if a true refined state is required.
