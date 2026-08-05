[OK] affine reparam (c=-1/2,r=1/2) + inverse (c=1,r=2) recovers parent exactly
  substitution y=1+2z yields x=z; symmetric for right branch.

[OK] global rational split [C-R,C] / [C,C+R] matches normalized TM2R children
  identical to the two affine maps when C=0,R=1.

[OK] terminal verifier + L/H children ⇒ exhaustive cover of root tile
  standard binary partition induction on axis-aligned splits; any variable works.

[OK] post-projection heuristic does not affect partition soundness
  split_state operates on pre-event branch; injectivity never invoked.

No remaining soundness flaw in affine bridge, inverse reconstruction, trace, or induction. Prior auditability issue is closed.
