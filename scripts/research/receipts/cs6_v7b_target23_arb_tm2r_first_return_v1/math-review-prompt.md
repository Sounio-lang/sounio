# Hostile mathematical audit: Arb TM2R first full-leaf return

Return BLOCKER for any possible under-enclosure. Audit the following carrier,
not its software style.

Each state component is represented as a total-degree-2 polynomial in six
variables plus an Arb interval remainder:

```text
X_i = P_i(xi,eta,rho_1,...,rho_4) + R_i,
(xi,eta,rho) in [-1,1]^6.
```

The two source variables `xi,eta` are never reparameterized. Multiplication
retains all degree-at-most-2 terms and puts every higher-degree monomial into
the interval remainder using its exact parity range on the cube: `[-1,1]` if
any exponent is odd, `[0,1]` when every exponent is even and the monomial is
nonconstant. Products with existing interval remainders use

```text
range(P)*S + range(Q)*R + R*S.
```

Each step has a validated Picard tube `B`, strict `h*L(B)<1`, normalized
autonomous-flow coefficients `a_k=D_t^k phi(0,q)/k!`, an order-12 polynomial,
and the componentwise Lagrange remainder `range(a_13(B))*h^13`. The raw TM
endpoint range must be contained in `B`.

After every step the raw TM is reconditioned. Pure `xi,eta` monomials are
copied unchanged. Every coefficient vector `g` of a monomial containing a
residual variable becomes a zonotope generator:

- if any exponent is odd, its monomial range is `[-1,1]`, so use generator `g`;
- if all exponents are even, its range is `[0,1]`, so add `g/2` to the retained
  constant vector and use generator `g/2`;
- each scalar remainder interval contributes its midpoint to the retained
  constant and its radius as a separate coordinate-axis generator.

A real Gram-Schmidt basis is derived from generator midpoints, rounded to
finite decimal strings, and then interpreted as an exact rational matrix `Q`.
Its inverse is computed by exact rational Gauss-Jordan elimination. For each
Arb interval generator vector `g`, compute `c=Q^-1*g` with outward rounding,
verify componentwise that outward-rounded `Q*c` contains `g`, and accumulate

```text
d_j = sum_over_generators sup(abs(c_j)).
```

The new residual carrier is `Q*diag(d)*rho`, `rho in [-1,1]^4`. There is no
claim that `Q` is exactly orthogonal; only exact invertibility and generator
reconstruction are used.

This process completed 617 steps. The full pre-step `w` interval was strictly
negative; the full post-step `w` interval was strictly positive. A Picard tube
for that step enclosed every trajectory and `x*y-zs` was strictly positive on
the tube. The claim is only: every initial point in the exact leaf has one
negative-to-positive `w=0` crossing in that time step, with positive normal.

Audit separately:

1. TM multiplication and parity tail enclosure;
2. zonotope conversion of odd/even residual monomials and scalar remainders;
3. exact-rational `Q`, interval `Q^-1*g`, summed coordinate radii, and whether
   this encloses the entire Minkowski sum despite coefficient dependencies;
4. repeated reconditioning and time integration;
5. whether strict endpoint signs plus a Picard tube and positive normal prove
   existence and orientation of at least one crossing for every trajectory;
6. whether they prove uniqueness of the crossing in the step, and whether the
   stated first-return wording needs an additional no-earlier-crossing argument.

Do not promote second return, determinant, covering relation, or chaos.
