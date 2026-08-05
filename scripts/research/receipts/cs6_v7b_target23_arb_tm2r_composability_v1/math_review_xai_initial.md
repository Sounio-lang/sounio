[OK] source chart + inverse reconstruction
  Linear map and explicit inverse match (det = nx²+ny², signs agree with adjugate).

[OK] slab enlargement + acceptance predicates
  1/128 radius does not relax the four strict enclosure/sign/Newton conditions; soundness preserved.

[OVERREACH] splitting pre-event state from post-projection carrier
  Selection of split variable after Newton projection can miss branches whose images land in the same slab; coverage claim requires explicit pre-image argument.

[TIGHTENABLE] partial-run report
  Statement “neither … certified” is correct but should add explicit “coverage incomplete after five sub-branches.”

[WRONG] missing provenance
  Tile receipts require at minimum (nx,ny,det,slab radius,Newton tolerance,seed) before they can underwrite any covering relation; absent fields block downstream use.
