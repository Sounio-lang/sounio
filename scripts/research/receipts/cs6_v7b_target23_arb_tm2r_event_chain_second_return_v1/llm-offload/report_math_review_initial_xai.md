[OK] `tau_0(z) = -w(z) / mid(dw/dt)`
  Standard first-order event-time predictor; consistent with interval Newton usage.

[OK] `C = -range(w(Phi_tau_0(X))) / range(dw/dt)`
  Standard interval Newton correction step; acceptance condition (Picard slab + strict sign) is correctly stated.

[OK] Carrier stabilization via exact rational endpoint reconstruction
  Preserves TM2 structure while enforcing enclosure; no flattening claimed.

[OK] Certification scope explicitly limited
  Correctly states that leaf coverage, not covering relation or return-map determinant, is obtained.

No algebraic, derivation, or statistical errors found.
