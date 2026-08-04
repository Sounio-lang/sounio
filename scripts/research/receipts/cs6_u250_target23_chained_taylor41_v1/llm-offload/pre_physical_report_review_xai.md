**Acknowledged.** This is a clean, self-contained two-return validated integration certificate for the frozen target-23 box. The chain (1 686 steps, degree-40 Taylor + order-41 Lagrange, `h = 2^{-8}`, signed S1.I31.F192) has been reconstructed and checked by an independent verifier; both zero-crossings of `w` are isolated inside `2^{-50}` brackets with strictly positive normal velocity, and all local radii remain comfortably below the contraction bounds.

The reported resource figures after multiplier/function sharing (≈23 % DSP, 136.99 MHz estimated) are consistent with the single-card HLS flow; the dual-card physical result is correctly marked false until a second U250 appears under XRT.

### Immediate next mathematical step (as you noted)
Replace the scalar-radius initial enclosure by a tight covering of the target leaf (or a dependency-tracked set representation) and lift the return map to that covering. The existing Picard/Taylor infrastructure already supplies the local step operator; the missing piece is a global propagation of the set enclosure across the two-return chain while preserving the strict sign conditions on the section.

### Immediate next machine step
Once the second U250 is enumerated, the split at step 843 lets both partitions run concurrently with no hidden state conversion; the only additional verification required is the concatenation check at the boundary checkpoint (already stored explicitly).

If you would like a compact formal statement of the two-return claim (in the style of a CAP or a Coq/Isabelle lemma skeleton), a suggested format for the leaf-covering set, or help drafting the next checkpoint header once the spatial covering is in place, just say the word.
