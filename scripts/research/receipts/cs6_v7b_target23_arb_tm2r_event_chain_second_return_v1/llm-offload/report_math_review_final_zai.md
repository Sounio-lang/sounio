[OK] `tau_0(z) = -w(z) / mid(d w / dt)`
  Standard first-order Taylor expansion predictor for the root of `w(Phi_t(X)) = 0` using the midpoint of the derivative interval.

[OK] `C = -range(w(Phi_tau_0(X))) / range(d w / dt)`
  Standard interval Newton corrector step `-f(t)/f'([t])` used to rigorously enclose the root within the Picard slab.

[OK] "bounds the order-13 temporal remainder" for an "order-12 flow polynomial"
  Taylor series truncation at order $N=12$ inherently yields an $N+1=13$ temporal remainder term.

[OK] "even the constant coefficient of the section derivative enclosed both signs. Spatial subdivision could not shrink that constant interval"
  Correct: spatial subdivision shrinks the dependency on spatial variables (higher-order monomial terms), but cannot shrink a constant coefficient if its uncertainty is fundamentally driven by non-spatial integration residuals.

[OK] "The node count is a search budget, not the size of a complete depth-18 binary tree."
  A complete depth-18 binary tree requires $2^{19}-1 = 524,287$ nodes, vastly exceeding the stated 16,383-node budget ($2^{14}-1$).

[TIGHTENABLE] "every degree-two source monomial therefore satisfies `|z^alpha| <= 1`"
  The property $|z^\alpha| = |\xi|^a |\eta|^b \le 1$ holds for monomials of *any* degree over $[-1, 1]^2$, not just degree two.
