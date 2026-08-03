[OK] Mean kernel time `0.449751859 s` per 331-orbit batch yields `735.961383` orbits/s.
  $331 / 0.449751859 = 735.961383246...$, matches to the stated precision.

[OK] The largest absolute difference (`~1.99e-18`) and Arb separations (`~1.64e-18`) are strictly less than the smallest retained CAPD margin (`~5.07e-15`).
  $1.99 \times 10^{-18} \ll 5.07 \times 10^{-15}$, confirming that hardware fixed-point discretization error does not violate the enclosing CAPD intervals.

[OK] The input/output data footprint satisfies the stated parameters (331 inputs, 2648 output words).
  $331 \times 8 = 2648$. This implies a fixed 8-word payload per orbit (e.g., standard state variables, event crossings, determinant), which is dimensionally consistent with Q24.40 (64-bit) arithmetic.

[OK] Integration step parameter `2^-10`.
  Standard power-of-two step size scaling, allowing exact representation in fixed-point arithmetic without floating-point conversion overhead.
