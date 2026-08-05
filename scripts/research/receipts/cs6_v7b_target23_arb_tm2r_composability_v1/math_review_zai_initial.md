```
[OK] Chart M = [[nx,ny],[-ny,nx]] is a scaled rotation; columns (nx,ny),(-ny,nx) are orthogonal with equal norm √(nx²+ny²).
[OK] det(M) = nx²+ny² (>0 for (nx,ny)≠0, so invertible). No division-by-zero risk unless source config degenerates.
[OK] Inverse (x0,y0) = M⁻¹(Uc,Sc): x0 = (nx·Uc − ny·Sc)/det, y0 = (ny·Uc + nx·Sc)/det. Matches.
[OK] Linear block: M⁻¹·(Ur,0)ᵀ = (nx·Ur/det, ny·Ur/det); M⁻¹·(0,Sr)ᵀ = (−ny·Sr/det, nx·Sr/det). Matches x_xi,y_xi,x_eta,y_eta. Source/worker chart is consistent end-to-end; no sign or inverse error.
[OK] Horizons 2000→2600 and slab growth 1/256 + 1/128 = 3/256 are budget/envelope changes. With the four strict gates (closed Picard boxes, strict sign of Df, predictor strictly inside, interval-Newton bracket strictly inside) the enlarged slab admits more candidates but cannot admit a false positive: rigor is gated by the interval-Newton bracket, not by the slab size. Not a blocker.
[TIGHTENABLE] "radius 1/128 after the old maximum 1/256" — ambiguous whether final radius is 3/256 (additive) or 1/128 (replacement). State explicitly: it is an additive margin, total = 1/256 + 1/128 = 3/256.
[OVERREACH] "splits the pre-event TM2R state in the dominant source/residual variable" chosen from the post-projection carrier, then "preserves the corresponding global xi,eta subdomain." Coverage preservation is *asserted*, not derived. For a true covering relation you must show two facts that are not stated:
   (a) the union of the children's pre-event partitions equals the parent's pre-event tile (no holes in x0,y0,x_xi,…,y_eta space); and
   (b) the union of the children's (xi,eta) subdomains equals the parent's, i.e., the post-projection images of the split pre-event pieces are jointly exhaustive.
   Choosing the split variable using post-projection state does not by itself break coverage, but it does mean the pre-event partition is data-dependent; without an explicit proof that the partition is a refinement (or at least a cover) of the parent pre-event box, downstream "covering relation" claims rest on an unproven lemma. Required: record, per split, the parent pre-event box, the chosen variable, the cut value, and the two child pre-event boxes; then assert (and ideally interval-check) that child_A ∪ child_B = parent.
[OK] Negative certification in result_report.md ("neither complete tile support, B→C, recurrence, nor chaos is certified") is conservative given that no JSON tile receipt was emitted and only 5/... subbranches finished. Not an overclaim — if anything it is slightly *under*-claiming if the 10 internal carriers were themselves already rigorous, but that is the safe direction.
[OVERREACH] Calling the 10 produced objects "internal carriers" without a JSON tile receipt is fine terminologically, but the report should state that they are *provisional* and not admissible as witnesses to any covering relation until the split-tree coverage proof above is filled in. As written, a downstream reader could mistake "carrier" for "certified tile".
[TIGHTENABLE] Provenance fields required before these (or future) receipts can support a covering relation:
   • source chart params: nx, ny (as exact rationals), Uc, Sc, Ur, Sr, with units;
   • P² specification: which Poincaré section, integrator, step count (2600 horizon), and a content hash of the integrator code;
   • Picard closure tolerance (absolute/relative) and the exact closing predicate;
   • interval-Newton contraction factor / acceptance predicate;
   • split tree: parent id, split variable, cut, child ids, with assertion of coverage (a)/(b);
   • RNG seed (if any sampling entered the slab search);
   • wall-clock and machine fingerprint for reproducibility.
[OK] Strict-sign + interval-Newton + predictor-strictly-inside conjunction is the right gate; no leap.
[OK] "Emits a tile receipt only after every child has completed" is the correct atomicity rule given the split semantics; no leak.
```

**Bottom line.** No blocker in the chart/inversion math. One MAJOR issue: the split-driven retry claims coverage preservation implicitly, but the report and contract must (i) prove parent = child_A ∪ child_B in *pre-event* coordinates, and (ii) prove the post-projection (xi,eta) subdomains are jointly exhaustive. Without those two clauses the words "covering relation" cannot be supported by any receipt this worker emits.
