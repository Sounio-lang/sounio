[OK] Depth-4 partition yields 16x16=256 disjoint cells.
  Arithmetic is correct, and dyadic grid partitioning is a disjoint complete cover.

[OK] 231 retained cells + 25 rejected cells = 256 initial cells.
  Arithmetic is correct; accounts for the entire parent rectangle.

[OK] Replacing 25 rejected cells with four dyadic children forms a disjoint complete cover of the rejected regions (Q1).
  Standard 2D dyadic refinement (depth $n$ to $n+1$) partitions a rectangle into 4 mutually exclusive and collectively exhaustive sub-rectangles.

[OK] Total adaptive leaves = 231 + 25*4 = 331.
  Arithmetic is correct (231 + 100 = 331).

[OK] Expected selected attempts = 331 * 2 = 662 (Q3).
  Arithmetic is correct (2 carriers per leaf).

[OK] Leaf rule logic is sound (Q2).
  If $x \in I_i$ for all six interval enclosures, then $x \in \bigcap I_i$. If the intersection is nonempty and strictly negative ($\bigcap I_i \subset (-\infty, 0)$), then $x < 0$.

[OK] Exact rational conversion of binary64 hex endpoints.
  Binary64 floats are dyadic rationals; converting to exact rationals prior to intersection eliminates rounding errors during interval intersection logic.

[OK] Bounding the result to "retrospective, target-23-only, not global HPG, not V7-B, not prospective" avoids overclaim (Q4).
  The logic strictly limits mathematical conclusions to the evaluated geometric domain (target 23) using specific retained data, correctly refraining from asserting global properties (e.g., V7_B_ELIGIBILITY=false, GLOBAL_HPG_CERTIFICATE=false).
