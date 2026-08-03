[OK] S1.I31.F96 intervals are defined by 128-bit endpoints
  1 sign bit + 31 integer bits + 96 fractional bits = 128 bits exactly.

[OK] Multiplication of S1.I31.F96 values requires an exact 256-bit intermediate
  The product of two 128-bit scaled integers (128 bits × 128 bits) yields a 256-bit intermediate exactly, which is sufficient for subsequent directed rounding back to F96.

[TIGHTENABLE] Exact 256-bit intermediates applied to addition/subtraction and division
  Addition/subtraction requires at most 129 bits (32 integer bits + 96 fractional + 1 sign) to prevent overflow, and division of a 128-bit integer by a standard positive integer requires at most 128 bits. 256 bits is exact but over-allocated for these specific operations.

[OK] Fail-closed overflow condition for 128-bit endpoint bounds
  The sum of two valid S1.I31.F96 numbers can exceed the maximum 31-bit integer bound (requiring up to 32 integer bits), meaning legal interval additions can mathematically overflow the 128-bit domain and must trigger refusal.

[OK] Validated ODE proof obligations (Picard self-map, strict contraction, event transversality, bounded Poincare determinant)
  The listed certification criteria are mathematically sound, well-defined, and standard requirements in rigorous numerics for bounding continuous dynamical systems.
