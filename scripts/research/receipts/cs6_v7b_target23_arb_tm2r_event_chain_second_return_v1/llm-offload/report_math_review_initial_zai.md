[OK] Event-time Newton predictor `tau_0(z) = -w(z)/mid(dw/dt)` is the correct first-order linearization of `w(Φ_t(X))=0`.
[OK] Interval correction `C = -range(w(Φ_{tau_0}(X)))/range(dw/dt)` is a standard Krawczyk/Newton contractor; acceptance on `tau_0+C` inside the Picard slab plus strict sign of `dw/dt` is sound.
[OK] Subset consistency: projected normal hull `[54.3440, 54.3722]` ⊂ derivative hull `[53.4950, 54.9793]`; both endpoints enclosed.
[TIGHTENABLE] "Generators cover discarded coefficient uncertainty using `|z^alpha| <= 1`."
  Hypothesis `|z| ≤ 1` (and `alpha ≥ 0`) is unstated; without it the generator bound fails. State the working domain or the bound on `|z|`.
[OVERREACH] "16,383 derivative-directed split nodes at depth 18 produced no sign-resolved leaf."
  `16383 = 2^14 − 1`; on a binary split this is the full-tree count at depth 14, not 18. Either "depth ≤18" or describe the pruning rule; as written the depth/leaf count pair is internally inconsistent.
[OK] Carrier stabilization invariant "each rebuilt Arb ball must contain its input" is the correct containment condition for a non-flattening TM2 rebuild; preserving exact rational endpoints and re-enclosing is sound.
[OK] Acceptance of projected carrier with `w` set exactly to zero while retaining source/residual symbols is consistent with a section-crossing certificate.
[OK] Scope honesty: no claim of return-map determinant, covering relation, HPG result, or chaos; downstream mathematical targets (degree certificate, face images, entry/exit inequalities) are correctly listed as not yet established.
[OK] Per-child re-test after the `XHEL` split using the same Newton test preserves the per-tile independence of the certificate.

Highest-impact items to fix first: (1) state the `|z|≤1` (or equivalent) hypothesis behind `|z^alpha|≤1`; (2) reconcile the 16,383 / depth-18 counts.
