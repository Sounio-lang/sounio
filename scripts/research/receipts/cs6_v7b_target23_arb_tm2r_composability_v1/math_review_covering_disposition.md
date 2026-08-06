# Disposition of initial covering review

No numerical covering claim exists yet; the four support jobs are still
running. The review was applied to the method and implementation.

## Accepted hardening

The analyzer now independently walks each terminal split tree on the induced
global `xi=-1` and `xi=+1` boundary. At an `XI` bisection it follows the unique
outer child; at every other bisection it requires both children. The aggregate
records positive recursive check counts for both faces, and the verifier
requires them.

The aggregate now states the row-coordinate convention, the section-normal
velocity definition, and the Poincare determinant formula explicitly.

## Rejected findings

- Fixing local `xi` does not evaluate only two points. It fixes one coordinate
  and ranges the other five local variables over `[-1,1]`, including the full
  Taylor-model remainder. Exact affine bisections map the selected outer local
  face onto the corresponding global face.
- In unstable dimension one, left-negative/right-positive boundary values give
  degree `+1` without monotonicity. A convex boundary-preserving homotopy takes
  the unstable component to `A*xi`, `A>1`, while the stable component contracts
  to zero. Interior oscillations do not change Brouwer degree.
- `target_hset()` uses row coordinates
  `u=a*x+b*y`, `s=-b*x+a*y`. Its inverse has columns
  `(a*U_RADIUS,b*U_RADIUS)/D` and
  `(-b*S_RADIUS,a*S_RADIUS)/D`, `D=a^2+b^2`, so its physical
  chart determinant is exactly `U_RADIUS*S_RADIUS/D`.
- `initial_normal` and `final_normal` are not bare normals. They are interval
  enclosures of the signed normal velocity `dw/dt=x*y-ZS` on `w=0`.
- Losing correlation by interval multiplication can widen an enclosure but
  cannot make a valid outward interval enclosure unsound. The certificate only
  needs strict sign exclusion.

The determinant claim remains conditional on `ell` retaining the worker's
integrated-divergence convention. This is checked against the existing worker
implementation and the already used first-edge analyzer before finalization.

## Recheck disposition

The xAI recheck accepted the explicit boundary recursion, face restriction,
chart determinant, and one-dimensional degree argument. It correctly requested
that `ell` enclose the integrated divergence; the worker ODE sets
`ell'=x-y-(w+ZS)/2-1`, exactly the divergence of its `(x,y,w)` vector field.

Z.AI accepted the face recursion, charts, and Poincare formula, but interpreted
the degree as a six-dimensional degree and therefore objected to contracting
the stable image to zero. In an h-set covering relation the degree is instead
the Brouwer degree of the terminal unstable map `A: R^u -> R^u`, here `u=1`.
The h-set has one unstable and one stable dimension; `rho0..rho3` are rigorous
Taylor-model enclosure parameters, not additional h-set dimensions. The
aggregate and verifier now state and enforce those dimensions explicitly.
