# Math recheck with explicit construction

Recheck the method below. No numerical certificate is claimed yet.

## Face restriction

Each terminal model has six local variables. For the global left face, the
analyzer selects exactly the terminal domains whose exact rational global
`xi` lower bound is `-1`, substitutes only local variable zero with `-1`, and
ranges the other five variables over `[-1,1]`. The right face is analogous with
local `xi=+1`. Every split is exactly
`old=-1/2+new/2` or `old=1/2+new/2`.

The new recursive boundary-cover check starts at each relevant source tile. At
an internal `XI` split it follows the unique outer child; at any other variable
it requires both children; a terminal node is accepted only when its exact
global xi bound touches the requested face.

## Degree

The analyzer requires, for every remaining source coordinate,
`u_C(P2(-1,*))<-1`, `u_C(P2(+1,*))>1`, and
`s_C(P2(xi,*)) in (-1,1)` on the whole support. The claimed homotopy contracts
the stable component to zero and takes the unstable component by convex
homotopy to `A*xi`, `A>1`. On the two exit faces, both endpoints of the convex
combination remain strictly on the same outside side. The terminal map has
Brouwer degree `+1`.

## Charts

Rows are `u=a*x+b*y`, `s=-b*x+a*y`, with `D=a^2+b^2>0`. The inverse chart is

```
x=(a*u-b*s)/D
y=(b*u+a*s)/D
```

For `u=U_CENTER+U_RADIUS*xi`, `s=S_CENTER+S_RADIUS*eta`, its physical Jacobian
has determinant `U_RADIUS*S_RADIUS/D > 0`.

## Poincare determinant

The section is `w=0`; the signed normal velocity is interval-evaluated as
`dw/dt=x*y-ZS`. Under the existing worker convention, `ell` is the integrated
divergence, so Liouville plus the section correction gives

```
det(DP_section)=exp(ell)*initial_normal/final_normal
det(DP_normalized)=det(DP_section)*det(source_chart)/det(target_chart)
```

All factors are outward Arb intervals on each terminal carrier and acceptance
requires a strictly positive lower bound.

Identify any remaining mathematical or fail-closed gap. In particular, say
whether the explicit boundary recursion resolves the face-cover concern and
whether monotonicity is necessary for degree `+1` under the stated sign pairing.
