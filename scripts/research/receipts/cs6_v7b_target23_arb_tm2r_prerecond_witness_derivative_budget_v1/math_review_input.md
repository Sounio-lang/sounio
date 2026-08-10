# Independent math review request

Review the following exact-rational diagnostic for mathematical or logical
errors. Focus on interval soundness, the TM2R multiplication decomposition,
the split substitution, and whether the stated decision follows from the data.

## Construction

There are six normalized variables on `[-1,1]`. Each state component is a
degree-2 polynomial with interval coefficients plus one interval remainder.
For `D = u*v - w - z_s`, multiplication retains monomials of total degree at
most two and adds all other contributions to the remainder as:

1. polynomial products of degree greater than two;
2. `P_u * R_v`;
3. `P_v * R_u`;
4. `R_u * R_v`;
5. `-R_w`.

Each polynomial monomial is enclosed on `[-1,1]^6`: a nonconstant monomial
with any odd exponent uses `[-1,1]`; a nonconstant all-even monomial uses
`[0,1]`. Interval multiplication uses all four endpoint products. Interval
addition is Minkowski addition.

For a one-level split of one normalized variable, the polynomial is transformed
exactly by substituting `old = -1/2 + new/2` for the left child and
`old = +1/2 + new/2` for the right child. The inherited component remainders
are unchanged. The derivative model is then rebuilt from the child components.

## Terminal exact results

- Endpoint derivative enclosure: `[-297.32166943797415, 340.8269847254264]`.
- Total width: `638.1486541634006`.
- Interval remainder width: `638.1389110524358`, or
  `99.99847322235952%` of total width.
- Remainder decomposition by width:
  - `R_u * R_v`: `282.4911829348361` (`44.26797646126046%`).
  - `P_u * R_v`: `261.3522700594368` (`40.95538847935282%`).
  - `P_v * R_u`: `81.73073732267902` (`12.80767179482701%`).
  - `-R_w`: `12.56472073495388` (`1.968963264476665%`).
  - discarded degree greater than two: `5.299600161515271e-10`.
- The first three sources sum to `98.03103673544029%` of remainder width.
- The best one-level split is `rho3`, reducing the worse child radius by only
  `0.000912330373203929%`; neither child has a positive lower bound.
- The observed terminal time tube, which is separate from the endpoint model,
  requires radius contraction greater than `14.668741391572123` to be strictly
  positive if its midpoint is held fixed.

All stored quantities are exact fractions; decimals above are only summaries.
An independent verifier recomputes interval ranges, widths, decomposition sums,
variable attributions, split domains, split factors, and classification. It
passes, and 29 negative mutations are rejected.

## Claimed conclusion

Another isolated symbolic bisection is not a credible route in the current
representation. The next experiment should transport and recondition the `u`
and `v` remainder directions with a rigorous QR doubleton/tripleton carrier,
then repeat the whole time-tube event test. This diagnostic does not claim that
all deeper subdivision fails, that QR must succeed, or that any crossing,
covering relation, recurrence, chaos, or open problem has been certified.

Return `PASS` or `REJECT`, then list any blocker or required wording correction.
