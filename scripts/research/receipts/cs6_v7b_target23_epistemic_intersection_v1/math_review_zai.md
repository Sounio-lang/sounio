[OK] det(q0_xy) = (unstable_x*stable_y - stable_x*unstable_y) * radius_u * radius_s
  The determinant of the 2x2 matrix formed by columns [radius_u*unstable_x, radius_u*unstable_y] and [radius_s*stable_x, radius_s*stable_y] exactly factors as stated.

[OK] Resident and Homogeneous reconstruction scaling
  Factoring out the column norms before propagation and restoring them later is sound because the determinant of a 2x2 matrix is bilinear: det([a*u, b*s]) = a*b*det([u, s]).

[OK] Liouville enclosure formula (exp(∫div f) * v_i/v_f * det(q0_xy))
  Standard Poincaré-map section area scaling correctly accounts for the normal velocity ratio and divergence integral.

[OK] Claim: I6 certifies d < 0, intersection supplies cross-method consistency
  If I1...I6 rigorously enclose d, and I6 ⊂ (-∞, 0), then d ∈ I6 guarantees d < 0. The intersection ∩I_i is nonempty (since d is in all) and is bounded by I6, making it strict-negative. The intersection supplies cross-method consistency but does not "refine" the strict-negativity of I6, since I6 is already strict-negative.

[OK] Bounded claim limited to 200 retained attempts is sound
  Verifying a mathematical or statistical property over a finite set (N=200) is logically sound and valid, provided the bounds of the claim explicitly restrict the conclusion to those 200 attempts (no global generalization).

[OK] Parsing binary64 hex endpoints via float.fromhex and as_integer_ratio is exact
  float.fromhex perfectly recovers the exact IEEE 754 binary64 float, and as_integer_ratio exactly converts this float into its irreducible rational numerator and denominator, allowing for rigorous and exact interval arithmetic.
