# Pre-execution review disposition

The first focused prompt abbreviated `rk4` as if the propagation step were
fixed during event localization.  That prompt was incomplete: the actual
function receives a Q24.40 step argument, and bisection recomputes one RK4 step
with that fractional step size.  The corrected prompt and the implementation
are retained separately.

Useful finding applied:

- The post-processing constants are now frozen as binary64 hexadecimal values
  in both Python and C++, eliminating expression-order differences in `q0`.

Findings checked and intentionally not applied:

- Python `qdiv` does not use signed `//`; it explicitly truncates the absolute
  magnitude and restores the sign, matching Vitis C++ division.
- Signed `ap_int<128>` right shift is the specified HLS operation.  Python uses
  the same arithmetic shift, and Vitis CSim matched all 2,648 frozen output
  words exactly.
- The event state is deliberately the strictly nonnegative right endpoint of
  the final bracket.  This is a deterministic numerical transcript, not a
  claim that the endpoint is the exact root.
- Intermediate overflow is absent on the frozen 331-orbit corpus: an overflow
  disagreement between bounded `ap_int` CSim and the unbounded Python model
  would have appeared in the bit-exact comparison.
- Truncation in the RK4 update is part of the frozen fixed-point method.  The
  result is compared to retained Decimal and CAPD evidence; it is not promoted
  to a rigorous interval integration certificate.

DeepSeek returned `Insufficient Balance`; the failed provider response is
retained.  XAI and Z.AI responses, including the invalid-prompt attempt, remain
available for audit rather than being presented as unanimous approval.
