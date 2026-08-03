# Result-stage mathematical audit: target-23 Arb validated center orbit

Audit the exact scope and mathematical soundness of this retained result. Distinguish a valid center-orbit enclosure from any leaf-wide or global claim.

Frozen method: 256-bit python-flint Arb balls; order-40 Taylor polynomial; fixed step 2^-8; per-step Picard inclusion X0+[0,h]F(B) subset B; explicit Banach bound h*L_inf(B)<1; order-41 interval remainder; global state-radius propagation by exp(mu_inf*h); independently isolated two negative-to-positive Poincare crossings with event-segment Picard boxes; transversality from a positive lower bound of x*y-zs; Liouville determinant exp(ell(T))*nu(0)/nu(T)*det(DQ0). All decimal constants enter as exact base-10 rationals. CAPD output is not read by the worker and is used only by the verifier for strict containment comparison.

Slurm job 8548, Python 3.12.3, python-flint 0.8.0, source commit 62c4deb0d49c1f0457c2b0ea62c745631cff1e13.

Observed exact obligations:
- 1793 Picard calls and 1793 containments; maximum Picard iterations 5.
- max h*L_inf upper bound = 1504044378330017502109337803399842957758855755619123208914565054859747160249 / 7237005577332262213973186563042994240829374041602535252466099000494570602496, approximately 0.2078268922496043 < 1.
- maximum propagated global state-radius upper bound approximately 3.354759306896197e-15.
- second event-time width = 2^-50; initial and final normal lower bounds approximately 142.8566 and 54.3578, both positive.
- validated determinant interval approximately [-2.8398635728038477e-11, -2.8398635728037957e-11], exact width 338587319/649037107316853453566312041152512, approximately 5.216763651615146e-25.
- the full determinant interval is strictly negative and strictly inside both retained CAPD carrier intervals.
- independent verifier passes locally and on the cluster; 14/14 semantic/provenance mutations are rejected.

Requested verdict:
1. Is it mathematically justified to call this an independently implemented validated enclosure of the exact frozen center trajectory and its Liouville/oriented-return determinant?
2. Identify any concrete flaw in the stated Picard/Taylor/global-error/event/transversality/Liouville chain.
3. Confirm or reject the strict boundary: this does not certify the full leaf, a full independent interval engine, global H-PG, V7-B eligibility, novelty/priority, an open problem, or FPGA execution.
4. Return PASS only if the narrow wording is supported; otherwise give a concrete counterexample or missing obligation.
