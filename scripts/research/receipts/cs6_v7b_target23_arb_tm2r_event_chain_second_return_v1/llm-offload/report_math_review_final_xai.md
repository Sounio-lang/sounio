[OK] interval-Newton + TM2 projection to exact `w=0` with strict transversality sign
Standard validated root-finding step; Picard slab + range enclosure of `C` guarantees unique simple zero inside slab.

[OK] carrier stabilization by rational endpoint reconstruction after each projection
Preserves TM2 structure and residual generators; each rebuilt ball containing its input is the required inclusion property.

[OK] `tau_0(z) = -w(z)/mid(dw/dt)` followed by one Newton correction `C`
Correct first-order predictor-corrector; acceptance only when `tau_0+C` lies inside contracting slab and derivative keeps strict sign is sufficient for rigorous event enclosure.

[OK] reported hulls (downward event-time, second upward, derivative, normal)
Direct output of the validated projections; no downstream covering/return-map claims are asserted, so no overreach.

[OK] `XHEL` split and both children passing identical test
Consistent with the single binary subdivision needed to obtain sign-resolved derivative.
